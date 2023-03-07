"""
pusher_slider_sticking_force_input.py
Description:
    Define an instance of class ControlAffineParameterAffineSystem for the system of a
    point finger attempting to push a sliding square with a force input.
    It is assumed that sticking contact is always made.
"""

from typing import Callable, Tuple, Optional, List
from matplotlib.axes import Axes
import torch

from neural_clbf.systems.adaptive.control_affine_parameter_affine_system import (
    ControlAffineParameterAffineSystem,
)

from neural_clbf.systems.utils import (
    Scenario,
    ScenarioList,
    lqr,
)

import polytope as pc
import numpy as np

class PusherSliderStickingForceInput(ControlAffineParameterAffineSystem):
    """
    Represents a point finger attempting to push a sliding square with a force input.
    Sticking contact only is assumed.

    The system has state defined in the plane (x, y)

        x = [s_x, s_y, s_theta]

    where s_x, s_y are the position of the center of the sliding square, and s_theta
    is the orientation (angle) of the sliding square.

    The system has two inputs:

        u = [f_x, f_y]

    representing the force (in Newtons) of the robot finger at the point of contact.
    The finger does not slide at all and remains at the center of te square block.

    The system has unknown parameters which are the position of the center of mass
    which determines the center of rotation of the sliding square:

        theta = [c_x, c_y]

    where c_x, c_y are the position of the center of mass of the sliding square
    relative to the geometric center of the sliding square.
    """

    # Number of states, controls and paramters
    N_DIMS = 3
    N_CONTROLS = 2
    N_PARAMETERS = 2

    # State Indices
    S_X = 0
    S_Y = 1
    S_THETA = 2

    # Control indices
    F_X = 0
    F_Y = 1

    # Parameter indices
    C_X = 0
    C_Y = 1

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
        # =================
        self.s_mass = 1.05  # kg

        self.K_x = float(-1.0)
        self.K_y = float(-1.0)
        self.K_z = float(-1.0)

        # Geometric Parameters (helpful for plotting and some calculations)
        self.s_length = 0.09
        self.s_width = self.s_length
        self.ps_cof = 0.3
        self.st_cof = 0.35
        self.p_radius = 0.01

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
        valid = valid and ("obstacle_radius" in s)

        return valid

    @property
    def n_dims(self) -> int:
        return PusherSliderStickingForceInput.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return [PusherSliderStickingForceInput.S_THETA]

    @property
    def parameter_angle_dims(self) -> List[int]:
        return []

    @property
    def n_controls(self) -> int:
        return PusherSliderStickingForceInput.N_CONTROLS

    @property
    def n_params(self) -> int:
        return PusherSliderStickingForceInput.N_PARAMETERS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # Define Upper and lower values
        upper_limit = torch.ones(self.n_dims).to(self.device)
        upper_limit[PusherSliderStickingForceInput.S_X] = 1.0
        upper_limit[PusherSliderStickingForceInput.S_Y] = 1.0
        upper_limit[PusherSliderStickingForceInput.S_THETA] = np.pi

        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        upper_limit = 10.0 * torch.ones(PusherSliderStickingForceInput.N_CONTROLS)
        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    def safe_mask(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Return the mask of x indicating safe regions for this system

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

        obst_center = torch.Tensor(
            [obst_center_x, obst_center_y]
        ).to(self.device)
        obst_radius = self.nominal_scenario["obstacle_radius"]

        # Algorithm
        safe_mask = torch.ones_like(x[:, 0], dtype=torch.bool).to(self.device)

        displacement_to_obst_center = x[:, :2] - obst_center.repeat(batch_size, 1)

        # distance to obstacle center is greater than radius + half of the length of the square
        dist_to_obst_center_big_enough = displacement_to_obst_center.norm(dim=-1) >= (obst_radius + self.s_length/2)

        safe_mask.logical_and_(dist_to_obst_center_big_enough)

        return safe_mask

    def unsafe_mask(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Return the mask of x indicating safe regions for this system

        args:
            x: a tensor of (batch_size, self.n_dims) points in the state space
        returns:
            a tensor of (batch_size,) booleans indicating whether the corresponding
            point is in the unsafe region.
        """
        # Constants
        batch_size = x.shape[0]

        obst_center_x = self.nominal_scenario["obstacle_center_x"]
        obst_center_y = self.nominal_scenario["obstacle_center_y"]

        obst_center = torch.Tensor(
            [obst_center_x, obst_center_y]
        ).to(self.device)
        obst_radius = self.nominal_scenario["obstacle_radius"]

        # Algorithm
        unsafe_mask = torch.ones_like(x[:, 0], dtype=torch.bool).to(self.device)

        displacement_to_obst_center = x[:, :2] - obst_center.repeat(batch_size, 1)

        # distance to obstacle center is greater than radius + half of the length of the square
        dist_to_obst_center_big_enough = displacement_to_obst_center.norm(dim=-1) <= (obst_radius + self.s_length / 2)

        unsafe_mask.logical_and_(dist_to_obst_center_big_enough)

        return unsafe_mask

    def goal_mask(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Return the mask of x indicating goal regions for this system

        args:
            x: a tensor of (batch_size, self.n_dims) points in the state space
        returns:
            a tensor of (batch_size,) booleans indicating whether the corresponding
            point is in this region.
        """
        # Include a sensible default
        goal_tolerance = 0.1
        batch_size = x.shape[0]

        # Create goal position
        goal = torch.ones(batch_size, self.n_dims-1).to(self.device)
        goal *= 0.5  # goal is in the upper right corner of the workspace

        # Algorithm
        r = torch.zeros(batch_size, 2).to(self.device) #get position from state
        r[:, :] = x[:, :2]

        return (r - goal).norm(dim=-1) <= goal_tolerance

    def goal_point(self, theta: torch.Tensor) -> torch.Tensor:
        """
        goal_point
        Description:
            In this case, we force the goal state.
        """
        # Defaults
        if theta is None:
            theta = torch.zeros(1, self.n_params).to(self.device)

        # Constants
        batch_size = theta.shape[0]

        # Algorithm
        goal = torch.ones(batch_size, self.n_dims).to(self.device)
        goal *= 0.5  # goal is in the upper right corner of the workspace
        goal[:, PusherSliderStickingForceInput.S_THETA] = 0.0

        return goal

    @property
    def u_eq(self):
        return torch.zeros((1, self.n_controls)).to(self.device)

    def sample_unsafe(self, num_samples: int, max_tries: int = 5000) -> [torch.Tensor, torch.Tensor, torch.Tensor]:
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
        obst_center = np.array(
            [obst_center_x, obst_center_y]
        )
        obst_radius = self.nominal_scenario["obstacle_radius"]

        state_space_obst_center = np.zeros(self.n_dims)
        state_space_obst_center[:2] = obst_center

        state_space_obst_width = upper_limit.cpu().numpy()
        state_space_obst_width[:3] = obst_radius

        P_unsafe = pc.box2poly(
            np.array(
                [state_space_obst_center - state_space_obst_width, state_space_obst_center + state_space_obst_width]).T
        )

        # Sample States
        x_unsafe_np = self.get_N_samples_from_polytope(P_unsafe, num_samples)
        x_unsafe = torch.Tensor(x_unsafe_np.T).to(self.device)

        theta_unsafe = self.sample_Theta_space(num_samples)

        xtheta_unsafe = torch.cat([x_unsafe, theta_unsafe], dim=1)

        return xtheta_unsafe, x_unsafe, theta_unsafe

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

        f_max, tau_max = self.limit_surface_bounds()
        a = (1/10.0)*(1/(f_max ** 2))
        b = (1/10.0)*(1/(tau_max ** 2))

        # States
        s_x = x[:, PusherSliderStickingForceInput.S_X]
        s_y = x[:, PusherSliderStickingForceInput.S_Y]
        s_theta = x[:, PusherSliderStickingForceInput.S_THETA]

        # Algorithm
        g[:, PusherSliderStickingForceInput.S_X, PusherSliderStickingForceInput.F_X] = torch.cos(s_theta) * a
        g[:, PusherSliderStickingForceInput.S_X, PusherSliderStickingForceInput.F_Y] = -torch.sin(s_theta) * a

        g[:, PusherSliderStickingForceInput.S_Y, PusherSliderStickingForceInput.F_X] = torch.sin(s_theta) * a
        g[:, PusherSliderStickingForceInput.S_Y, PusherSliderStickingForceInput.F_Y] = torch.cos(s_theta) * a

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
        g = torch.zeros((batch_size, self.n_dims, self.n_controls)).to(self.device)
        g = g.type_as(x)

        f_max, tau_max = self.limit_surface_bounds()
        a = (1 / 10.0) * (1 / (f_max ** 2))
        b = (1 / 10.0) * (1 / (tau_max ** 2))

        # States
        s_x = x[:, PusherSliderStickingForceInput.S_X]
        s_y = x[:, PusherSliderStickingForceInput.S_Y]
        s_theta = x[:, PusherSliderStickingForceInput.S_THETA]

        # Create output
        G = torch.zeros((batch_size, self.n_dims, self.n_controls, self.n_params)).to(self.device)

        G[:, PusherSliderStickingForceInput.S_THETA, PusherSliderStickingForceInput.F_X, PusherSliderStickingForceInput.C_Y] = -b
        G[:, PusherSliderStickingForceInput.S_THETA, PusherSliderStickingForceInput.F_Y, PusherSliderStickingForceInput.C_X] = b

        return G

    def limit_surface_bounds(self):
        # Constants
        g = 9.8

        # Create output
        f_max = self.st_cof * self.s_mass * g

        slider_area = self.s_width * self.s_length
        # circular_density_integral = 2*pi*((ps.s_length/2)^2)*(1/2)
        circular_density_integral = (1 / 12) * ((self.s_length / 2) ** 2 + (self.s_width / 2) ** 2) * np.exp(1)

        tau_max = self.st_cof * self.s_mass * g * (1 / slider_area) * circular_density_integral
        return f_max, tau_max

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
        m = self.s_mass
        g = 9.8

        # Compute nominal control from feedback + equilibrium control
        K = self.K.type_as(x)
        estimated_goal = self.goal_point(theta_hat)
        # estimated_goal[:, 3:] = theta_hat

        u_nominal = -(K @ (x - estimated_goal).T).T

        # Adjust for the equilibrium setpoint
        u = u_nominal + self.u_eq.type_as(x)
        #u[:, 1] = u[:, 1] + m*g

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
            K_np = lqr(A, B, Q, R)
            self.K = torch.tensor(K_np)

            Acl_list.append(Act - Bct @ K_np)

        print(len(scenarios))
        print(scenarios)

        # If more than one scenario is provided...
        # get the Lyapunov matrix by robustly solving Lyapunov inequalities
        if len(scenarios) > 1:
            #self.P = torch.tensor(robust_continuous_lyap(Acl_list, Q))
            self.P = torch.eye(self.n_dims)
        else:
            # Otherwise, just use the standard Lyapunov equation
            #self.P = torch.tensor(continuous_lyap(Acl_list[0], Q))
            self.P = torch.eye(self.n_dims)