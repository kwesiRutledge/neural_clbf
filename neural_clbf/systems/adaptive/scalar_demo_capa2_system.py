"""
ScalarCAPA2Demo.py
Description:
    Define an instance of class ControlAffineParameterAffineSystem for the system of
    a scalar CAPA2 system that is useful for illustrating the concepts of adaptive CLFs.
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

class ScalarCAPA2Demo(ControlAffineParameterAffineSystem):
    """
    Represents a simple scalar system with scalar unknown parameter that I used to prove the intuition of the aCLF method.
    Should be easy to train.
    The dynamics are simple:
        dx/dt = (1 + theta) * x + (1 + theta) * u

    The control input will be nicknamed u1.

    representing the force (in Newtons) of the robot end effector in the three axes of motion.
    """

    # Number of states, controls and parameters
    N_DIMS = 1
    N_CONTROLS = 1
    N_PARAMETERS = 1

    # State Indices
    X_DEMO = 0

    # Control indices
    U_DEMO = 0

    # Parameter indices
    P_DEMO = 0

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
        self.device = device

        # Then initialize
        super().__init__(
            nominal_scenario, Theta, dt, controller_dt,
            device=device,
            theta=theta,
            use_linearized_controller=use_linearized_controller,
        )

        self.scenarios = scenarios

    def validate_scenario(self, s: Scenario) -> bool:
        """Check if a given set of parameters is valid

        args:
            params: a dictionary giving the parameter values for the system.
        returns:
            True if parameters are valid, False otherwise
        """
        valid = True

        valid = valid and ("wall_position" in s)

        return valid

    @property
    def n_dims(self) -> int:
        return ScalarCAPA2Demo.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return []

    @property
    def n_controls(self) -> int:
        return ScalarCAPA2Demo.N_CONTROLS

    @property
    def n_params(self) -> int:
        return ScalarCAPA2Demo.N_PARAMETERS

    @property
    def parameter_angle_dims(self) -> List[int]:
        return []

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # Define Upper and lower values
        upper_limit = torch.ones(self.n_dims, device=self.device)
        upper_limit[ScalarCAPA2Demo.X_DEMO] = 10.0

        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)


    @property
    def U(self) -> pc.Polytope:
        """
        U = self.U

        Description:
            This abstract property returns the polytope U, which is the set of all
            allowable control inputs for the system.
        """
        # Define the polytope
        U = pc.box2poly(np.array([[-10.0, 10.0]]))

        return U

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
        wall_pos = self.nominal_scenario["wall_position"]

        # Algorithm
        safe_mask = torch.ones_like(x[:, 0], dtype=torch.bool).to(self.device)

        safe_mask.logical_and_(wall_pos <= x[:, 0])

        return safe_mask

    def unsafe_mask(self, x: torch.Tensor, theta:torch.Tensor) -> torch.Tensor:
        """Return the mask of x indicating unsafe regions for this system

        args:
            x: a tensor of (batch_size, self.n_dims) points in the state space
        returns:
            a tensor of (batch_size,) booleans indicating whether the corresponding
            point is in this region.
        """
        # Constants
        batch_size = x.shape[0]
        wall_pos = self.nominal_scenario["wall_position"]

        # Algorithm
        unsafe_mask = torch.zeros_like(x[:, 0], dtype=torch.bool).to(self.device)

        unsafe_mask.logical_and_(wall_pos > x[:, 0])

        return unsafe_mask

    def goal_mask(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Return the mask of x indicating goal regions for this system

        args:
            x: a tensor of (batch_size, self.n_dims) points in the state space
        returns:
            a tensor of (batch_size,) booleans indicating whether the corresponding
            point is in this region.
        """
        # Include a sensible default
        goal_tolerance = 0.1

        return (x - self.goal_point(theta)).norm(dim=-1) <= goal_tolerance

    def goal_point(self, theta: torch.Tensor):
        """
        goal_point
        Description:
            Return the goal point for this system
        """
        # Constants
        bs = theta.shape[0]

        # Algorithm
        goal = torch.zeros(bs, self.n_dims, device=self.device)
        goal[0, 0] = 1.0
        return goal

    @property
    def u_eq(self):
        return torch.zeros((1, self.n_controls)).to(self.device)

    def _f(self, x: torch.Tensor, params: Scenario) -> torch.Tensor:
        """
        Return the control-independent part of the control-affine dynamics.

        Notes:
            This should be the function f(x) = x for the scalar input x.

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            f: bs x self.n_dims x 1 tensor
        """
        # Constants
        batch_size = x.shape[0]
        f = torch.zeros(
            (batch_size, self.n_dims, 1),
            device=self.device,
        )
        p_demo = x[:, ScalarCAPA2Demo.P_DEMO]

        # Algorithm
        f[:, ScalarCAPA2Demo.X_DEMO, 0] = p_demo

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
        F = torch.zeros(
            (batch_size, self.n_dims, self.n_params),
            device=self.device,
        )
        p_demo = x[:, ScalarCAPA2Demo.P_DEMO]

        # Algorithm
        F[:, ScalarCAPA2Demo.X_DEMO, ScalarCAPA2Demo.P_DEMO] = p_demo

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
        g = torch.zeros(
            (batch_size, self.n_dims, self.n_controls),
            device=self.device,
        )
        g = g.type_as(x)

        # Algorithm
        g[:, ScalarCAPA2Demo.X_DEMO, ScalarCAPA2Demo.U_DEMO] = 1.0

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
        G = torch.zeros(
            (batch_size, self.n_dims, self.n_controls, self.n_params),
            device=self.device,
        )

        # Algorithm
        G[:, ScalarCAPA2Demo.X_DEMO, ScalarCAPA2Demo.U_DEMO, ScalarCAPA2Demo.P_DEMO] = 1.0

        return G

    def u_nominal(
        self, x: torch.Tensor, theta_hat: torch.Tensor, params: Optional[Scenario] = None
    ) -> torch.Tensor:
        """
        Compute the nominal control for the nominal parameters, using LQR unless
        overridden

        args:
            x: bs x self.n_dims tensor of state
            params: the model parameters used
        returns:
            u_nominal: bs x self.n_controls tensor of controls
        """
        # Compute nominal control from feedback + equilibrium control
        K = self.K.type_as(x)
        goal = self.goal_point(theta_hat).type_as(x)
        u_nominal = -(K @ (x - goal).T).T

        # Adjust for the equilibrium setpoint
        u = u_nominal + self.u_eq.type_as(x)

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
        unless overridden.

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
        wall_pos = self.nominal_scenario["wall_position"]
        S_w, S_u, S_x0 = self.get_mpc_matrices(N_mpc, dt=dt)

        U_T_A = np.kron(U.A, np.eye(N_mpc))
        U_T_b = np.kron(U.b, np.ones(N_mpc))
        U_T = pc.Polytope(U_T_A, U_T_b)

        # Solve the MPC problem for each element of the batch
        batch_size = x.shape[0]
        u = torch.zeros(
            (batch_size, n_controls),
            device=self.device,
        ).type_as(x)
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
            constraints += [S_u @ u_T + np.dot(S_x0, batch_x) >= wall_pos]
            constraints += [S_u @ u_T + np.dot(S_x0, batch_x) <= wall_pos + 2]

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

            u[batch_idx, :] = torch.tensor(u_T_opt[:n_controls], self.device)
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
