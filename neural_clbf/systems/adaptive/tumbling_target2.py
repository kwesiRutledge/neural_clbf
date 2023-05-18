"""
tumbling_target2.py
Description:
    Define an instance of class ControlAffineParameterAffineSystem for the system of
    a spacecraft attempting to land in a specific region of a target vehicle that is tumbling
    with an unknwon rate of rotation in T seconds.
    This is a modified version of the original system where I change the units of certain states
    to make the numerical issues go away.
"""
from abc import (
    ABC,
    abstractmethod,
    abstractproperty,
)
from typing import Callable, Tuple, Optional, List

from matplotlib.axes import Axes
import numpy as np
import torch
from torch.autograd.functional import jacobian

from neural_clbf.systems.utils import (
    Scenario,
    ScenarioList,
    lqr,
    robust_continuous_lyap,
    continuous_lyap,
)

import polytope as pc

from neural_clbf.systems.adaptive.control_affine_parameter_affine_system import (
    ControlAffineParameterAffineSystem
)

import cvxpy as cp

class TumblingTarget2(ControlAffineParameterAffineSystem):
    """
    Represents a spacecraft attempting to land in a specific region of a target vehicle that is tumbling
    with an unknwon rate of rotation in T seconds.

    The system has state defined in the dynamical form written in:
    - "A Spacecraft Benchmark Problem for Hybrid Control and Estimation" [IEEEXplore: https://ieeexplore-ieee-org.libproxy.mit.edu/stamp/stamp.jsp?tp=&arnumber=7798765]

        x = [ p_x , p_y , p_z, phi_d, v_x, v_y, v_z]

    where p_x, p_y, and p_z are the x-, y-, and z- position of the Chaser (spacecraft trying to dock with target) in
    free space, v_x, v_y, and v_z are the x-, y- and z- velocities of the Chaser in free space.

    The control inputs are:

        u = [ F_x, F_y, F_z ]

    representing the force (in Newtons) of the robot end effector in the three axes of motion.
    """

    # Number of states, controls and parameters
    N_DIMS = 7
    N_CONTROLS = 3
    N_PARAMETERS = 1

    # State Indices
    P_X = 0
    P_Y = 1
    P_Z = 2
    PHI_D = 3  # Orientation of the Target
    V_X = 4
    V_Y = 5
    V_Z = 6

    # Control indices
    F_X = 0
    F_Y = 1
    F_Z = 2

    # Parameter indices
    OMEGA_0 = 0

    def __init__(
        self,
        nominal_scenario: Scenario,
        Theta: pc.Polytope,
        dt: float = 0.01,
        controller_dt: Optional[float] = None,
        use_linearized_controller: bool = True,
        scenarios: Optional[ScenarioList] = None,
        theta: torch.Tensor = None,
        device: str = "cpu",
    ):
        """
        Initialize a system.

        args:
            nominal_scenario: a dictionary giving the parameter values for the system
            dt: the timestep to use for simulation
            controller_dt: the timestep for the LQR discretization. Defaults to dt
            use_linearized_controller: if True, linearize the system model to derive a
                                       LQR controller. If false, the system is must
                                       set self.P itself to be a tensor n_dims x n_dims
                                       positive definite matrix.
            scenarios: an optional list of scenarios for robust control
            theta: The true value of the parameter governing this system's dynamics.
        raises:
            ValueError if nominal_scenario are not valid for this system
        """
        # Define parameters
        self.m = 500.0 # kg . Mass of Chaser

        self.mu = 3.986e14
        self.a = 353e3  # 353 km altitude in low Earth orbit
        self.n = np.sqrt(self.mu/np.power(self.a, 3))

        self.T_docking = 100.0

        self.docking_dist = 1.0

        self.device = device

        # Then initialize
        super().__init__(nominal_scenario, Theta, dt, controller_dt, device=device)

    def validate_scenario(self, s: Scenario) -> bool:
        """Check if a given set of parameters is valid

        args:
            params: a dictionary giving the parameter values for the system.
        returns:
            True if parameters are valid, False otherwise
        """
        valid = True

        valid = valid and ("obstacle_center_x" in s)
        valid = valid and ("obstacle_center_y" in s)
        valid = valid and ("obstacle_center_z" in s)
        valid = valid and ("obstacle_width" in s)

        return valid

    @property
    def n_dims(self) -> int:
        return TumblingTarget2.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return [TumblingTarget2.PHI_D]

    @property
    def parameter_angle_dims(self) -> List[int]:
        return []

    @property
    def n_controls(self) -> int:
        return TumblingTarget2.N_CONTROLS

    @property
    def n_params(self) -> int:
        return TumblingTarget2.N_PARAMETERS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # Define Upper and lower values
        upper_limit = torch.ones(self.n_dims).to(self.device)
        upper_limit[TumblingTarget2.P_X] = 25.0
        upper_limit[TumblingTarget2.P_Y] = 25.0
        upper_limit[TumblingTarget2.P_Z] = 25.0
        upper_limit[TumblingTarget2.PHI_D] = 10.0
        upper_limit[TumblingTarget2.V_X] = 25.0
        upper_limit[TumblingTarget2.V_Y] = 25.0
        upper_limit[TumblingTarget2.V_Z] = 25.0

        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)


    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        upper_limit = 10.0 * torch.ones(TumblingTarget2.N_CONTROLS)
        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    def safe_mask(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Return the mask of x indicating safe regions for this system

        args:
            x: a tensor of (batch_size, self.n_dims) points in the state space
        returns:
            a tensor of (batch_size,) booleans indicating whether the corresponding
            point is in this region.
        """
        # Constants
        batch_size = x.shape[0]
        obst_center_x = self.nominal_scenario["obstacle_center_x"]
        obst_center_y = self.nominal_scenario["obstacle_center_y"]
        obst_center_z = self.nominal_scenario["obstacle_center_z"]
        obst_center = torch.Tensor(
            [obst_center_x, obst_center_y, obst_center_z]
        ).to(self.device)
        obst_width = self.nominal_scenario["obstacle_width"]

        # Algorithm
        safe_mask = torch.ones_like(x[:, 0], dtype=torch.bool).to(self.device)

        displacement_to_obst_center = x[:, :3] - obst_center.repeat(batch_size, 1)
        dist_to_obst_center_big_enough = displacement_to_obst_center.norm(dim=-1) >= obst_width

        safe_mask.logical_and_(dist_to_obst_center_big_enough)

        return safe_mask

    def unsafe_mask(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Return the mask of x indicating unsafe regions for this system

        args:
            x: a tensor of (batch_size, self.n_dims) points in the state space
        returns:
            a tensor of (batch_size,) booleans indicating whether the corresponding
            point is in this region.
        """
        # Constants
        batch_size = x.shape[0]
        obst_center_x = self.nominal_scenario["obstacle_center_x"]
        obst_center_y = self.nominal_scenario["obstacle_center_y"]
        obst_center_z = self.nominal_scenario["obstacle_center_z"]
        obst_center = torch.Tensor(
            [obst_center_x, obst_center_y, obst_center_z]
        ).to(self.device)
        obst_width = self.nominal_scenario["obstacle_width"]

        # Algorithm
        safe_mask = torch.zeros_like(x[:, 0], dtype=torch.bool)

        displacement_to_obst_center = x[:, :3] - obst_center.repeat(batch_size, 1)
        dist_to_obst_center_big_enough = displacement_to_obst_center.norm(dim=-1) <= obst_width

        safe_mask.logical_or_(dist_to_obst_center_big_enough)

        return safe_mask

    def goal_mask(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Return the mask of x indicating goal regions for this system

        args:
            x: a tensor of (batch_size, self.n_dims) points in the state space
        returns:
            a tensor of (batch_size,) booleans indicating whether the corresponding
            point is in this region.
        """
        # Include a sensible default
        goal_tolerance = .3
        batch_size = x.shape[0]

        # Algorithm

        r = torch.zeros(batch_size, 3).to(self.device) #get position from state
        r[:, :] = x[:, :3]

        goal_states = self.goal_point(theta)
        goal_r = goal_states[:, :3]

        return (r - goal_r).norm(dim=-1) <= goal_tolerance

    def goal_point(self, theta: torch.Tensor = None) -> torch.Tensor:
        """
        goal_point
        Description
            Return the goal point for each of the thetas in theta Tensor.
            Returns zero if there are none.
        args:
            theta: a tensor of (batch_size, self.n_params) points in the parameter space
        returns:
            a tensor of (batch_size, self.n_dims) points in the state space corresponding to
            goal points.
        """
        # Defaults
        if theta is None:
            theta = torch.zeros(1, self.n_params).to(self.device)

        # Constants
        batch_size = theta.shape[0]
        docking_dist = self.docking_dist
        T_docking = self.T_docking

        # Algorithm
        omega0 = theta

        # Create goal points
        goal = torch.zeros((batch_size, self.n_dims)).to(self.device)
        goal[:, 0] = docking_dist * torch.cos(omega0 * T_docking).squeeze()
        goal[:, 1] = docking_dist * torch.sin(omega0 * T_docking).squeeze()

        return goal

    @property
    def u_eq(self):
        return torch.zeros((1, self.n_controls)).to(self.device)

    def sample_unsafe(self, num_samples: int, max_tries: int= 5000) -> [torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch of unsafe states from the system

        args:
            num_samples: the number of samples to return
            max_tries: the maximum number of tries to sample a point before giving up
        returns:
            a tuple (x, u, theta) of tensors of size (num_samples, n_dims), (num_samples, n_controls),
            and (num_samples, n_params) respectively.
        """
        # Constants
        batch_size = num_samples
        upper_limit, lower_limit = self.state_limits

        # Create polytope for sampling
        obst_center_x = self.nominal_scenario["obstacle_center_x"]
        obst_center_y = self.nominal_scenario["obstacle_center_y"]
        obst_center_z = self.nominal_scenario["obstacle_center_z"]
        obst_center = np.array(
            [obst_center_x, obst_center_y, obst_center_z]
        )
        obst_width = self.nominal_scenario["obstacle_width"]

        state_space_obst_center = np.zeros(self.n_dims)
        state_space_obst_center[:3] = obst_center

        state_space_obst_width = upper_limit.cpu().numpy()
        state_space_obst_width[:3] = obst_width

        P_unsafe = pc.box2poly(
            np.array([state_space_obst_center - state_space_obst_width, state_space_obst_center + state_space_obst_width]).T
        )

        # Sample States
        x_unsafe_np = self.get_N_samples_from_polytope(P_unsafe, num_samples)
        x_unsafe = torch.Tensor(x_unsafe_np.T).to(self.device)

        theta_unsafe = self.sample_Theta_space(num_samples)

        xtheta_unsafe = torch.cat([x_unsafe, theta_unsafe], dim=1)

        return xtheta_unsafe, x_unsafe, theta_unsafe

    # def sample_goal(self, num_samples: int, max_tries: int= 5000) -> [torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """Sample a batch of unsafe states from the system
    #
    #     args:
    #         num_samples: the number of samples to return
    #         max_tries: the maximum number of tries to sample a point before giving up
    #     returns:
    #         a tuple (x, u, theta) of tensors of size (num_samples, n_dims), (num_samples, n_controls),
    #         and (num_samples, n_params) respectively.
    #     """
    #     # Constants
    #     batch_size = num_samples
    #     upper_limit, lower_limit = self.state_limits
    #
    #     # Create polytope for sampling
    #
    #
    #     state_space_obst_center = np.zeros(6)
    #     state_space_obst_center[:3] = obst_center
    #
    #     state_space_obst_width = upper_limit.cpu().numpy()
    #     state_space_obst_width[:3] = obst_width
    #
    #     P_unsafe = pc.box2poly(
    #         np.array([state_space_obst_center - state_space_obst_width, state_space_obst_center + state_space_obst_width]).T
    #     )
    #
    #     # Sample States
    #     x_unsafe_np = self.get_N_samples_from_polytope(P_unsafe, num_samples)
    #     x_unsafe = torch.Tensor(x_unsafe_np.T).to(self.device)
    #
    #     theta_unsafe = self.sample_Theta_space(num_samples)
    #
    #     xtheta_unsafe = torch.cat([x_unsafe, theta_unsafe], dim=1)
    #
    #     return xtheta_unsafe, x_unsafe, theta_unsafe

    def _f(self, x: torch.Tensor, params: Scenario) -> torch.Tensor:
        """
        Return the control-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            f: bs x self.n_dims x 1 tensor
        """
        # Constants
        batch_size = x.shape[0]
        f = torch.zeros((batch_size, self.n_dims, 1)).to(self.device)

        m = self.m
        n = self.n

        g = 10.0 # m/s^2

        # Algorithm
        p_x = x[:, TumblingTarget2.P_X]
        p_y = x[:, TumblingTarget2.P_Y]
        p_z = x[:, TumblingTarget2.P_Z]
        phi_target = x[:, TumblingTarget2.PHI_D]
        v_x = x[:, TumblingTarget2.V_X]
        v_y = x[:, TumblingTarget2.V_Y]
        v_z = x[:, TumblingTarget2.V_Z]

        f[:, TumblingTarget2.P_X, 0] = v_x
        f[:, TumblingTarget2.P_Y, 0] = v_y
        f[:, TumblingTarget2.P_Z, 0] = v_z
        # f[:, TumblingTarget2.PHI_D, 0] = 0.0
        f[:, TumblingTarget2.V_X, 0] = 3 * (n ** 2) * p_x + 2 * n * v_y
        f[:, TumblingTarget2.V_Y, 0] = -2 * n * v_x
        f[:, TumblingTarget2.V_Z, 0] = -(n ** 2) * p_z

        return f

    def _F(self, x: torch.Tensor, params: Scenario) -> torch.Tensor:
        """
        Return the control-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            F: bs x self.n_dims x self.n_params tensor
        """
        # Constants
        batch_size = x.shape[0]
        F = torch.zeros((batch_size, self.n_dims, self.n_params)).to(self.device)

        m = self.m

        # Algorithm
        F[:, TumblingTarget2.PHI_D, TumblingTarget2.OMEGA_0] = 1.0

        return F

    def _g(self, x: torch.Tensor, params: Scenario) -> torch.Tensor:
        """
        Return the control-dependent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            g: bs x self.n_dims x self.n_controls tensor
        """
        # Constants
        batch_size = x.shape[0]
        g = torch.zeros((batch_size, self.n_dims, self.n_controls)).to(self.device)
        g = g.type_as(x)

        m = self.m

        # Algorithm
        g[:, TumblingTarget2.V_X, TumblingTarget2.F_X] = 1.0
        g[:, TumblingTarget2.V_Y, TumblingTarget2.F_Y] = 1.0
        g[:, TumblingTarget2.V_Z, TumblingTarget2.F_Z] = 1.0

        return g

    def _G(self, x: torch.Tensor, params: Scenario) -> torch.Tensor:
        """
        Return the control-dependent and parameter-dependent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            G: bs x self.n_dims x self.n_controls x self.n_params tensor
        """
        # Constants
        batch_size = x.shape[0]
        G = torch.zeros((batch_size, self.n_dims, self.n_controls, self.n_params)).to(self.device)

        return G

    def u_nominal(
        self, x: torch.Tensor, theta_hat: torch.Tensor, params: Optional[Scenario] = None
    ) -> torch.Tensor:
        """
        Compute the nominal control for the nominal parameters, using LQR unless
        overridden

        args:
            x: bs x self.n_dims tensor of state
            theta_hat: bs x self.n_params tensor of state
            params: the model parameters used
        returns:
            u_nominal: bs x self.n_controls tensor of controls
        """
        # Constants
        batch_size = x.shape[0]
        n_dims = self.n_dims
        m = self.m
        # g = 9.8
        average_omega_0 = np.mean(pc.extreme(self.Theta))

        # Compute nominal control from feedback + equilibrium control
        K = self.K.type_as(x)
        estimated_goal = torch.zeros((batch_size, n_dims)).type_as(x)
        # estimated_goal[:, :3] = theta_hat
        # estimated_goal[:, 3:]

        u_nominal = -(K @ (x - estimated_goal).T).T

        # Adjust for the equilibrium setpoint
        u = u_nominal + self.u_eq.type_as(x)
        # u[:, 1] = u[:, 1] + m*g

        # Clamp given the control limits
        upper_u_lim, lower_u_lim = self.control_limits
        for dim_idx in range(self.n_controls):
            u[:, dim_idx] = torch.clamp(
                u[:, dim_idx],
                min=lower_u_lim[dim_idx].item(),
                max=upper_u_lim[dim_idx].item(),
            )

        return u

    def plot_environment(self, ax: Axes) -> None:
        """
        Add a plot of the environment to the given figure. Defaults to do nothing
        unless overidden.

        args:
            ax: the axis on which to plot
        """
        pass

    def basic_mpc1(self, x: torch.Tensor, dt: float, U: pc.Polytope = None, N_mpc: int = 5) -> torch.Tensor:
        """
        basic_mpc1
        Description:
            From the set of batch states x, this function computes the batch of inputs
            that solve the basic linearized MPC.
        """
        # Constants
        n_controls = self.n_controls
        n_dims = self.n_dims

        obstacle_loc = np.array([
            self.nominal_scenario["obstacle_center_x"],
            self.nominal_scenario["obstacle_center_y"],
            self.nominal_scenario["obstacle_center_z"],
        ])
        obstacle_width = self.nominal_scenario["obstacle_width"]

        obstacle = pc.box2poly(
            [[obstacle_loc[0] - 0.5 * obstacle_width, obstacle_loc[0] + obstacle_width*0.5],
            [obstacle_loc[1] - 0.5 * obstacle_width, obstacle_loc[1] + obstacle_width * 0.5],
            [obstacle_loc[2] - 0.5 * obstacle_width, obstacle_loc[2] + obstacle_width * 0.5]],
        )

        M = 1e10

        S_w, S_u, S_x0 = self.get_mpc_matrices(N_mpc, dt=dt)

        U_T_A = np.kron(U.A, np.eye(N_mpc))
        U_T_b = np.kron(U.b, np.ones(N_mpc))
        U_T = pc.Polytope(U_T_A, U_T_b)

        # Solve the MPC problem for each element of the batch
        batch_size = x.shape[0]
        u = torch.zeros(batch_size, n_controls).type_as(x)
        goal_x = np.array([[0.0]])
        for batch_idx in range(batch_size):
            batch_x = x[batch_idx, :n_dims].cpu().detach().numpy()

            # Create input for this batch
            u_T = cp.Variable((n_controls * N_mpc,))

            # Define objective as being the distance from state at t_0 + N_MPC to
            # the goal.
            M_T = np.zeros((n_dims, n_dims * (N_mpc)))
            M_T[:, -n_dims:] = np.eye(n_dims)
            obj = cp.norm(M_T @ (S_u @ u_T + np.dot(S_x0, batch_x)) - goal_x)
            obj += cp.norm(u_T)

            constraints = [U_T.A @ u_T <= U_T.b]

            #Add obstacle avoidance constraints
            bin_vecs = []
            for k in range(1, N_mpc):
                obstacle_binary_vec = cp.Variable((len(obstacle.A),), integer=True)
                bin_vecs.append(obstacle_binary_vec)

                Rx_k = np.zeros((3, n_dims * (N_mpc)))
                Rx_k[:, k*(n_dims):k*(n_dims)+3] = np.eye(3)

                constraints += [0 <= obstacle_binary_vec] + [obstacle_binary_vec <= 1]
                constraints += [sum(obstacle_binary_vec) == 1]
                for A_row_index in range(len(obstacle.A)):
                    A_row_i = obstacle.A[A_row_index]
                    b_i = obstacle.b[A_row_index]

                    constraints += [ A_row_i @ Rx_k @ (S_u @ u_T + np.dot(S_x0, batch_x)) + \
                                     M * (1 - obstacle_binary_vec[A_row_index]) >= b_i ]

            # Solve for the P with largest volume
            prob = cp.Problem(
                cp.Minimize(obj), constraints
            )
            prob.solve()
            # Skip if no solution
            if prob.status != "optimal":
                continue

            # Otherwise, collect optimal input value
            u_T_opt = u_T.value

            u[batch_idx, :] = torch.Tensor(u_T_opt[:n_controls])
            # self.dynamics_model.u_nominal(
            #     x_shifted.reshape(1, -1), track_zero_angle=False  # type: ignore
            # ).squeeze()

        return u

    def get_mpc_matrices(self, T=-1, dt: float=0.01):
        """
        get_mpc_matrices
        Description:
            Get the mpc_matrices for the discrete-time dynamical system described by self.
        Assumes:
            Assumes T is an integer input
        Usage:
            S_w, S_u, S_x0 = ad0.get_mpc_matrices(T)
        """

        # Input Processing
        if T < 0:
            raise DomainError("T should be a positive integer; received " + str(T))

        # Constants
        theta_center = np.mean(pc.extreme(self.Theta))
        A = np.array([[(1 + theta_center)*dt]])
        B = np.array([[np.exp((1+theta_center)*dt)-1]])

        n_x = self.n_dims
        n_u = self.n_controls
        n_w = 1

        # Create the MPC Matrices (S_w)
        E = np.eye(n_x)
        S_w = np.zeros((T * n_x, T * n_w))
        Bw_prefactor = np.zeros((T * n_x, T * n_x))
        for j in range(T):
            for i in range(j, T):
                Bw_prefactor[i * n_x:(i + 1) * n_x, j * n_x:(j + 1) * n_x] = np.linalg.matrix_power(A, i - j)

        S_w = np.dot(Bw_prefactor, np.kron(np.eye(T), E))

        # Create the MPC Matrices (S_u)
        S_u = np.zeros((T * n_x, T * n_u))
        for j in range(T):
            for i in range(j, T):
                S_u[i * n_x:(i + 1) * n_x, j * n_u:(j + 1) * n_u] = np.dot(np.linalg.matrix_power(A, i - j), B)

        # Create the MPC Matrices (S_x0)
        S_x0 = np.zeros((T * n_x, n_x))
        for j in range(T):
            S_x0[j * n_x:(j + 1) * n_x, :] = np.linalg.matrix_power(A, j + 1)

        # # Create the MPC Matrices (S_K)
        # S_K = np.kron(np.ones((T, 1)), K)
        # S_K = np.dot(Bw_prefactor, S_K)

        return S_w, S_u, S_x0

    def compute_linearized_controller(self, scenarios: Optional[ScenarioList] = None):
        """
        Computes the linearized controller K and lyapunov matrix P.
        """
        # We need to compute the LQR closed-loop linear dynamics for each scenario
        Acl_list = []
        # Default to the nominal scenario if none are provided
        if scenarios is None:
            scenarios = [self.nominal_scenario]

        # For each scenario, get the LQR gain and closed-loop linearization
        for s in scenarios:
            # Compute the LQR gain matrix for the nominal parameters
            Act, Bct = self.linearized_ct_dynamics_matrices(s)
            A, B = self.linearized_dt_dynamics_matrices(s)

            # Define cost matrices as identity
            Q = np.eye(self.n_dims)
            R = np.eye(self.n_controls)

            # Get feedback matrix
            # K_np = lqr(A, B, Q, R)
            K_np = np.hstack(
                (-0.01*np.eye(3), np.zeros((3,1)), -0.1*np.eye(3))
            )
            self.K = torch.tensor(K_np)

            Acl_list.append(Act - Bct @ K_np)

        # If more than one scenario is provided...
        # get the Lyapunov matrix by robustly solving Lyapunov inequalities
        if len(scenarios) > 1:
            self.P = torch.tensor(robust_continuous_lyap(Acl_list, Q))
        else:
            # Otherwise, just use the standard Lyapunov equation
            self.P = 0.01 * torch.eye(self.n_dims)

            # Our A matrix is not controllable (one state is autonomously evolving without user input)
            #self.P = torch.tensor(continuous_lyap(Acl_list[0], Q))

    def mpc_about_input_trajectory(
            self,
            x: torch.Tensor,
            theta_hat: torch.Tensor,
            X: torch.Tensor,
            params: Scenario = None,
            U: torch.Tensor = None,
            horizon: int = 10,

    ):
        """
        u = mpc_about_input_trajectory(x, theta_hat, X)
        Description:
            This function computes the mpc control that should steer the system
            closer to the trajectory defined by X.
        """
        raise NotImplementedError("This function is not implemented yet.")
        return False