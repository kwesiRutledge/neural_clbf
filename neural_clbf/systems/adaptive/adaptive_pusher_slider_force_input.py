"""
adaptive_pusher_slider_sticking_force_input.py
Description:
    Define an instance of class ControlAffineParameterAffineSystem for the system of a
    point finger attempting to push a sliding square with a force input.
    It is assumed that sticking contact is always made.
    The center of mass of the sliding square is unknown and is a parameter of the system.
"""

from typing import Callable, Tuple, Optional, List
from matplotlib.axes import Axes
import matplotlib.animation as manimation
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

import matplotlib.pyplot as plt

class AdaptivePusherSliderStickingForceInput(ControlAffineParameterAffineSystem):
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
        max_force: float = 10.0,
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

        # self.K_x = float(-1.0)
        # self.K_y = float(-1.0)
        # self.K_z = float(-1.0)

        # Geometric Parameters (helpful for plotting and some calculations)
        self.s_length = 0.09
        self.s_width = self.s_length
        self.ps_cof = 0.3
        self.st_cof = 0.35
        self.p_radius = 0.01

        self.max_force = max_force

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
        return AdaptivePusherSliderStickingForceInput.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return [AdaptivePusherSliderStickingForceInput.S_THETA]

    @property
    def parameter_angle_dims(self) -> List[int]:
        return []

    @property
    def n_controls(self) -> int:
        return AdaptivePusherSliderStickingForceInput.N_CONTROLS

    @property
    def n_params(self) -> int:
        return AdaptivePusherSliderStickingForceInput.N_PARAMETERS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # Define Upper and lower values
        upper_limit = torch.ones(self.n_dims).to(self.device)
        upper_limit[AdaptivePusherSliderStickingForceInput.S_X] = 1.0
        upper_limit[AdaptivePusherSliderStickingForceInput.S_Y] = 1.0
        upper_limit[AdaptivePusherSliderStickingForceInput.S_THETA] = torch.pi

        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    @property
    def U(self) -> pc.Polytope:
        """
        P_U = self.U

        Description:
            Return the polytope representing the control constraints for the given system.
        """
        # Define the matrices which define the polytope
        H = np.array([
            [1.0, -self.ps_cof],
            [-1.0, -self.ps_cof],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ])

        h = np.zeros((5, 1))
        h[2, 0] = self.max_force
        h[3, 0] = self.max_force

        return pc.Polytope(H, h)

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

        obst_center = torch.tensor(
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

        obst_center = torch.tensor(
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
        goal_tolerance = self.goal_radius
        batch_size = x.shape[0]

        # Create goal position
        goal_pose = self.goal_point(theta)
        goal_xy = goal_pose[:, :2].to(goal_pose.device)

        # Algorithm
        r = torch.zeros(batch_size, 2).to(self.device) #get position from state
        r[:, :] = x[:, :2]

        return (r - goal_xy).norm(dim=-1) <= goal_tolerance

    def goal_point(self, theta: torch.Tensor) -> torch.Tensor:
        """
        goal_point
        Description:
            In this case, we force the goal state to be the same point [0.5,0.5,0]
            for any theta input.
        """
        # Defaults
        if theta is None:
            theta = torch.zeros(1, self.n_params).to(self.device)

        # Constants
        batch_size = theta.shape[0]

        # Algorithm
        goal = torch.ones(batch_size, self.n_dims).to(self.device)
        goal *= 0.5  # goal is in the upper right corner of the workspace
        goal[:, AdaptivePusherSliderStickingForceInput.S_THETA] = 0.0

        return goal

    @property
    def goal_radius(self):
        """Return the radius of the goal region"""
        return 0.1

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
        if torch.get_default_dtype() == torch.float32:
            x_unsafe_np = np.float32(x_unsafe_np)
        x_unsafe = torch.tensor(x_unsafe_np, device=self.device)

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
        g = torch.zeros((batch_size, self.n_dims, self.n_controls), device=self.device)
        g = g.type_as(x)

        f_max, tau_max = self.limit_surface_bounds()
        # a = (1/10.0)*(1/(f_max ** 2))
        # b = (1/10.0)*(1/(tau_max ** 2))
        a = 1.0537  # Copied from paper
        b = 1.5087  # Copied from paper

        # States
        s_x = x[:, AdaptivePusherSliderStickingForceInput.S_X]
        s_y = x[:, AdaptivePusherSliderStickingForceInput.S_Y]
        s_theta = x[:, AdaptivePusherSliderStickingForceInput.S_THETA]

        # Algorithm
        g[:, AdaptivePusherSliderStickingForceInput.S_X, AdaptivePusherSliderStickingForceInput.F_X] = torch.cos(s_theta) * a
        g[:, AdaptivePusherSliderStickingForceInput.S_X, AdaptivePusherSliderStickingForceInput.F_Y] = -torch.sin(s_theta) * a

        g[:, AdaptivePusherSliderStickingForceInput.S_Y, AdaptivePusherSliderStickingForceInput.F_X] = torch.sin(s_theta) * a
        g[:, AdaptivePusherSliderStickingForceInput.S_Y, AdaptivePusherSliderStickingForceInput.F_Y] = torch.cos(s_theta) * a

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

        f_max, tau_max = self.limit_surface_bounds()
        # a = (1 / 10.0) * (1 / (f_max ** 2))
        # b = (1 / 10.0) * (1 / (tau_max ** 2))
        a = 1.0537  # Copied from paper
        b = 1.5087  # Copied from paper

        # States
        s_x = x[:, AdaptivePusherSliderStickingForceInput.S_X]
        s_y = x[:, AdaptivePusherSliderStickingForceInput.S_Y]
        s_theta = x[:, AdaptivePusherSliderStickingForceInput.S_THETA]

        # Create output
        G = torch.zeros((batch_size, self.n_dims, self.n_controls, self.n_params)).to(self.device)

        G[:, AdaptivePusherSliderStickingForceInput.S_THETA, AdaptivePusherSliderStickingForceInput.F_X, AdaptivePusherSliderStickingForceInput.C_Y] = -b
        G[:, AdaptivePusherSliderStickingForceInput.S_THETA, AdaptivePusherSliderStickingForceInput.F_Y, AdaptivePusherSliderStickingForceInput.C_X] = b

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

        # State
        s_x = x[:, AdaptivePusherSliderStickingForceInput.S_X]
        s_y = x[:, AdaptivePusherSliderStickingForceInput.S_Y]
        s_th = x[:, AdaptivePusherSliderStickingForceInput.S_THETA]

        # Create simple, case-based controller:
        #   - if current difference vector between current position and goal position
        #       has an angle within friction cone vector
        #   - otherwise

        f_l, f_u = self.friction_cone_extremes()
        angle_upper, angle_lower = torch.atan2(f_u[1], f_u[0]), torch.atan2(f_l[1], f_l[0])

        assert angle_upper > angle_lower, f"Expected angle_upper ({angle_upper}) to be larger than ({angle_lower}), but it wasn't!"

        goal = self.goal_point(theta_hat)

        des_direction = goal - x
        des_angle = torch.atan2(des_direction[:, 1], des_direction[:, 0]) + (np.pi/2 - s_th)

        # Handle each case depending on where the desired motion vector is pointing.
        # 1. Vector points above upper friction cone vector
        # 2. Vector points below lower friction cone vector
        # 3. Vector points in the friction cone
        u_nominal = torch.zeros((batch_size, AdaptivePusherSliderStickingForceInput.N_CONTROLS)).to(self.device)

        # 1. Vector points above upper friction cone vector
        # Assign inputs that are above the friction cone.
        des_angle_greater_mask = torch.ones_like(x[:, 0], dtype=torch.bool).to(self.device)
        des_angle_greater_mask.logical_and_(
            des_angle > angle_upper,
        )
        u_nominal[des_angle_greater_mask, :] = f_l / torch.norm(f_l)

        # 2. Vector points below lower friction cone vector
        des_angle_less_mask = torch.ones_like(x[:, 0], dtype=torch.bool).to(self.device)
        des_angle_less_mask.logical_and_(
            des_angle < angle_lower,
        )
        u_nominal[des_angle_less_mask, :] = f_u / torch.norm(f_u)

        # 3. Vector points in the friction cone
        des_angle_in_fc_mask = torch.ones_like(x[:, 0], dtype=torch.bool).to(self.device)
        des_angle_in_fc_mask.logical_and_(
            torch.logical_not(des_angle_greater_mask),
        )
        des_angle_in_fc_mask.logical_and_(
            torch.logical_not(des_angle_less_mask),
        )
        u_nominal[des_angle_in_fc_mask, AdaptivePusherSliderStickingForceInput.F_X] = \
            torch.cos(des_angle[des_angle_in_fc_mask])
        u_nominal[des_angle_in_fc_mask, AdaptivePusherSliderStickingForceInput.F_Y] = \
            torch.sin(des_angle[des_angle_in_fc_mask])



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
        unless overidden.

        args:
            ax: the axis on which to plot
        """
        pass

    def friction_cone_extremes(self)->(torch.tensor, torch.tensor):
        """
        [f_l, f_u] = self.friction_cone_extremes()

        Description:
            This function returns two unit vectors defining the boundary of the friction cone.
            The friction cone vectors are written in the frame of reference at the contact point
            with:
            - positive x being along the edge of the slider and
            - positive y being perpendicular and into the slider.

        Args:

        Outputs:
            f_u: A vector in the direction of the "upper" edge of the friction cone

            f_l: A vector in the direction of the "lower" edge of the friction cone
        """

        # Constants
        mu = self.ps_cof

        # Create output
        return torch.tensor([mu, 1.0], device=self.device), torch.tensor([-mu, 1.0], device=self.device)


    def compute_linearized_controller(self, scenarios: Optional[ScenarioList] = None):
        """
        Computes the linearized controller K and lyapunov matrix P.
        """
        # We need to compute the LQR closed-loop linear dynamics for each scenario
        Acl_list = []
        # Default to the nominal scenario if none are provided
        if scenarios is None:
            scenarios = [self.nominal_scenario]

        # # For each scenario, get the LQR gain and closed-loop linearization
        # for s in scenarios:
        #     # Compute the LQR gain matrix for the nominal parameters
        #     Act, Bct = self.linearized_ct_dynamics_matrices(s)
        #     A, B = self.linearized_dt_dynamics_matrices(s)
        #
        #     print("A: ", A)
        #     print("B: ", B)
        #
        #     # Define cost matrices as identity
        #     Q = np.eye(self.n_dims)
        #     R = np.eye(self.n_controls)
        #
        #     # Get feedback matrix
        #     K_np = lqr(A, B, Q, R)
        #     self.K = torch.tensor(K_np)
        #
        #     Acl_list.append(Act - Bct @ K_np)
        #
        # print(len(scenarios))
        # print(scenarios)
        # self.K = torch.tensor(
        #     [[0.1]]
        # )

        self.K = torch.tensor([
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
        ])

        # If more than one scenario is provided...
        # get the Lyapunov matrix by robustly solving Lyapunov inequalities
        if len(scenarios) > 1:
            #self.P = torch.tensor(robust_continuous_lyap(Acl_list, Q))
            self.P = torch.eye(self.n_dims)
        else:
            # Otherwise, just use the standard Lyapunov equation
            #self.P = torch.tensor(continuous_lyap(Acl_list[0], Q))
            self.P = torch.eye(self.n_dims)

    """
    contact_point
    Description:
        This function returns the contact point of the slider with the pusher.
    
    Returns
        contact_point: 1x2 tensor of the contact point in the world frame
    """
    def contact_point(self, x: torch.Tensor) -> torch.Tensor:
        # Input Processing
        assert x.shape == (3,), f"Expected x to be of shape (3,); received shape {x.shape}"

        # Constants

        # Get State
        s_x = x[0]
        s_y = x[1]
        s_th = x[2]

        # Compute contact point
        rot2 = torch.tensor([[np.cos(s_th), -np.sin(s_th)], [np.sin(s_th), np.cos(s_th)]])
        contact_point = torch.tensor([[s_x], [s_y]]) + rot2 @ torch.tensor([[-self.s_length/2], [0]])

        return contact_point.T


    def plot_single(self,
        x: torch.Tensor, theta: torch.Tensor,
        ax: plt.Axes,
        limits: Optional[List[List[float]]] = [[0.0, 0.3], [0.0, 0.3]],
        hide_axes: bool = True,
        equal_aspect: bool = True,
        show_geometric_center: bool = False,
        show_CoM: bool = True,
        show_friction_cone_vectors: bool = True,
        show_obstacle: bool = True,
        show_goal: bool = True,
        current_force: torch.Tensor = None) -> plt.Figure:
        """
        plot_single
        Description:
            Plots a single pusher-slider system in a 2d plot and returns the figure.
        """

        # Input Checking
        assert (x.shape == (3,)), f"x must have shape ({self.n_dims},); received {x.shape}."
        assert (theta.shape == (2,)), f"theta must have shape ({self.n_params},); received {theta.shape}."

        # We can only visualize one state at a time.

        # Constants
        s_length = self.s_length
        s_width = self.s_width
        p_radius = self.p_radius

        # Get state
        s_x = x[0]
        s_y = x[1]
        s_th = x[2]

        # Get parameters
        CoM_x = theta[0]
        CoM_y = theta[1]

        # Setup Figure
        # =========
        fig = plt.gcf()
        # ax = fig.add_subplot(111)
        if hide_axes:
            ax.axis('off')

        # Set Axes Limits
        plt.xlim(limits[0])
        plt.ylim(limits[1])

        if equal_aspect:
            plt.gca().set_aspect('equal', adjustable='box')

        plot_objects = {}

        # Plot Objects
        # ============

        # Plot Obstacle first (just in case we collide with it, we want to make it clear that collision happens)
        if show_obstacle:
            s = self.nominal_scenario
            obstacle_x, obstacle_y, obstacle_radius = s["obstacle_center_x"], s["obstacle_center_y"], s["obstacle_radius"]
            obstacle = plt.Circle(
                (obstacle_x, obstacle_y),
                obstacle_radius,
                color="#E31A1C")
            ax.add_patch(obstacle)
            plot_objects["obstacle"] = obstacle

        # Plot Goal
        if show_goal:
            goal_pose = self.goal_point(theta)
            goal_xy = goal_pose[:, :2].to(goal_pose.device)
            goal = plt.Circle(
                (goal_xy[0, 0], goal_xy[0, 1]),
                self.goal_radius,
                color="#009392",
            )
            ax.add_patch(goal)
            plot_objects["goal"] = goal

        # Plot Slider
        s_th_in_degrees = np.degrees(s_th)
        alpha0 = np.arctan(s_length/s_width)  # Angle made by diagonal line going from rect lower left to upper right
        half_diagonal_length = np.sqrt((s_length/2)**2 + (s_width/2)**2)

        slider = plt.Rectangle(
            (s_x-half_diagonal_length*np.cos(alpha0+s_th), s_y-half_diagonal_length*np.sin(alpha0+s_th)),
            s_width, s_length,
            angle=s_th_in_degrees,
            color='cyan')
        plot_objects["slider"] = slider # Save slider for later updates in animation.

        ax.add_patch(slider)

        cp = self.contact_point(x)

        # Plot Slider's Geometric Center
        if show_geometric_center:
            geom_center = plt.scatter(s_x, s_y, color='blue', s=10)
            plot_objects["geom_center"] = geom_center

        # Plot Slider's Center of Mass
        if show_CoM:
            th_in_contact_point_frame = s_th - np.pi/2
            rotation_matrix = torch.tensor([
                [np.cos(th_in_contact_point_frame), -np.sin(th_in_contact_point_frame)],
                [np.sin(th_in_contact_point_frame), np.cos(th_in_contact_point_frame)]
            ])
            CoM = cp.T + rotation_matrix @ torch.tensor([[CoM_x], [CoM_y]])

            CoM_plot = plt.scatter(CoM[0, 0], CoM[1, 0], color='red', s=10)
            plot_objects["CoM"] = CoM_plot


        # Plot Pusher
        pusher = plt.Circle(
            (cp[0, 0] - p_radius * np.cos(s_th), cp[0, 1] - p_radius * np.sin(s_th)),
            p_radius,
            color="#03DAC6")
        ax.add_patch(pusher)
        plot_objects["pusher"] = pusher

        # Plot Friction Cone Vectors
        if show_friction_cone_vectors:
            # Get Friction Cone Vectors
            friction_cone_vectors = self.friction_cone_extremes()

            # Plot Friction Cone Vectors
            plot_objects["friction_cone_vectors"] = []
            for i, vec in enumerate(friction_cone_vectors):
                vec_clone = vec.clone().detach()
                norm_vec = vec_clone / torch.norm(vec_clone)
                scaled_vec = (s_length/2.0) * norm_vec

                th_in_contact_point_frame = s_th - np.pi / 2
                rotation_matrix = torch.tensor([
                    [np.cos(th_in_contact_point_frame), -np.sin(th_in_contact_point_frame)],
                    [np.sin(th_in_contact_point_frame), np.cos(th_in_contact_point_frame)]
                ])
                rotated_vec = rotation_matrix @ scaled_vec.T

                fcv_plot_i = plt.arrow(cp[0, 0], cp[0, 1],
                          rotated_vec[0], rotated_vec[1],
                          color="orange", width=0.001)
                plot_objects["friction_cone_vectors"].append(fcv_plot_i)

        # Plot Current Force
        if current_force is not None:
            # Normalize and plot vector of force
            current_force_clone = current_force.clone().detach()
            norm_vec = current_force_clone / torch.norm(current_force_clone)
            scaled_vec = (s_length/2.0) * norm_vec

            th_in_contact_point_frame = s_th - np.pi / 2
            rotation_matrix = torch.tensor([
                [np.cos(th_in_contact_point_frame), -np.sin(th_in_contact_point_frame)],
                [np.sin(th_in_contact_point_frame), np.cos(th_in_contact_point_frame)]
            ])
            rotated_vec = rotation_matrix @ scaled_vec.T

            cf_plot = plt.arrow(cp[0, 0], cp[0, 1],
                      rotated_vec[0], rotated_vec[1],
                      color="green", width=0.001)
            plot_objects["current_force"] = cf_plot

        return plot_objects

    def update_plot_objects(
            self,
            plot_objects: dict,
            x: torch.Tensor, theta: torch.Tensor,
            show_geometric_center: bool = False,
            show_CoM: bool = True,
            show_friction_cone_vectors: bool = True,
            current_force: torch.Tensor = None
    ) -> None:
        # Input Processing
        assert x.shape == (3,), "x must be a tensor of shape (3,); got {}".format(x.shape)
        assert theta.shape == (2,), "theta must be a tensor of shape (2,); got {}".format(theta.shape)

        # Constants
        s_length = self.s_length
        s_width = self.s_width
        p_radius = self.p_radius

        # State
        s_x = x[0]
        s_y = x[1]
        s_th = x[2]

        # Unknown Parameters
        CoM_x = theta[0]
        CoM_y = theta[1]

        # Update Slider
        alpha0 = np.arctan(s_length / s_width)  # Angle made by diagonal line going from rect lower left to upper right
        half_diagonal_length = np.sqrt((s_length / 2) ** 2 + (s_width / 2) ** 2)

        plot_objects["slider"].set_x(s_x - half_diagonal_length*np.cos(alpha0+s_th))
        plot_objects["slider"].set_y(s_y - half_diagonal_length*np.sin(alpha0+s_th))
        plot_objects["slider"].set_angle(np.degrees(s_th))

        # Update Geometric Center
        if show_geometric_center:
            # print("showing geometric center = ", show_geometric_center)
            plot_objects["geom_center"].set_offsets((s_x, s_y))

        # Update Center of Mass
        cp = self.contact_point(x)
        if show_CoM:
            th_in_contact_point_frame = s_th - np.pi / 2
            rotation_matrix = torch.tensor([
                [np.cos(th_in_contact_point_frame), -np.sin(th_in_contact_point_frame)],
                [np.sin(th_in_contact_point_frame), np.cos(th_in_contact_point_frame)]
            ])
            CoM = cp.T + rotation_matrix @ torch.tensor([[CoM_x], [CoM_y]])
            plot_objects["CoM"].set_offsets((CoM[0, 0], CoM[1, 0]))

        # Update Pusher
        plot_objects["pusher"].center = (cp[0, 0] - p_radius * np.cos(s_th), cp[0, 1] - p_radius * np.sin(s_th))

        # Update Friction Cone Vectors
        if show_friction_cone_vectors:
            # Get Friction Cone Vectors
            friction_cone_vectors = self.friction_cone_extremes()

            # Plot Friction Cone Vectors
            for i, vec in enumerate(friction_cone_vectors):
                vec_clone = vec.clone().detach()
                norm_vec = vec_clone / torch.norm(vec_clone)
                scaled_vec = (s_length/2.0) * norm_vec

                th_in_contact_point_frame = s_th - np.pi / 2
                rotation_matrix = torch.tensor([
                    [np.cos(th_in_contact_point_frame), -np.sin(th_in_contact_point_frame)],
                    [np.sin(th_in_contact_point_frame), np.cos(th_in_contact_point_frame)]
                ])
                rotated_vec = rotation_matrix @ scaled_vec.T

                plot_objects["friction_cone_vectors"][i].set_data(
                    x = cp[0, 0], y = cp[0, 1],
                    dx = rotated_vec[0],
                    dy = rotated_vec[1])

        # Update Current Force
        if current_force is not None:
            # Normalize and plot vector of force
            current_force_clone = current_force.clone().detach()
            norm_vec = current_force_clone / torch.norm(current_force_clone)
            scaled_vec = (s_length/2.0) * norm_vec

            th_in_contact_point_frame = s_th - np.pi / 2
            rotation_matrix = torch.tensor([
                [np.cos(th_in_contact_point_frame), -np.sin(th_in_contact_point_frame)],
                [np.sin(th_in_contact_point_frame), np.cos(th_in_contact_point_frame)]
            ])
            rotated_vec = rotation_matrix @ scaled_vec.T

            plot_objects["current_force"].set_data(
                x = cp[0, 0], y = cp[0, 1],
                dx = rotated_vec[0],
                dy = rotated_vec[1])



    def save_animated_trajectory(
            self,
            x_trajectory: torch.Tensor,
            th: torch.Tensor,
            f_trajectory: torch.Tensor,
            limits: Optional[List[List[float]]] = None,
            filename: str="pusherslider-animation1.mp4",
            hide_axes: bool = True,
            show_obstacle: bool = True,
            show_goal: bool = True,
        ):
        """
        save_animated_trajectory
        Description:
            Animates a trajectory of the pusher-slider system.
        Inputs:
            x_trajectory: A bs x N_traj x 3 tensor containing the trajectory of the system.
            th: A bs x 2 tensor containing the parameters of the system.
            f_trajectory: A bs x (N_traj-1) x 2 tensor containing the trajectory of the forces applied to the system.
            filename: The name of the file to save the animation to.
        """
        # Input Processing
        assert len(x_trajectory.shape) == 3, f"x is of the wrong dimension. Received tensor of {len(x_trajectory.shape)} dimensions; expected 3."
        assert x_trajectory.shape[1] == 3, f"Expected state tensor to have second dimension size 3; received {x_trajectory.shape[1]}."

        assert x_trajectory.shape[0] == th.shape[0], f"The batch size of x ({x_trajectory.shape[0]}) is different from batch size of theta ({th.shape[0]})."
        assert len(th.shape) == 2, f"theta is of the wrong dimension. Received tensor of {len(th.shape)} dimensions; expected 2."
        assert th.shape[1] == 2, f"Expected parameter tensor to have second dimension of size 2; received {th.shape[1]}"

        # Constants
        batch_size = x_trajectory.shape[0]
        N_traj = x_trajectory.shape[2]
        num_frames = N_traj
        dt = self.dt
        max_t = num_frames * dt
        min_t = 0.0
        s = self.nominal_scenario
        obstacle_x, obstacle_y, obstacle_radius = s["obstacle_center_x"], s["obstacle_center_y"], s["obstacle_radius"]
        goal_pose = self.goal_point(th)
        goal_xy = goal_pose[:, :2].to(goal_pose.device)

        # Create axis limits
        if limits is None:
            limits = []
            x_buffer = self.s_width
            x_limits = [torch.min(x_trajectory[:, self.S_X, :]) - x_buffer, torch.max(x_trajectory[:, self.S_X, :]) + x_buffer]
            limits.append(
                x_limits
            )

            y_buffer = self.s_length
            y_limits = [torch.min(x_trajectory[:, self.S_Y, :] - y_buffer), torch.max(x_trajectory[:, self.S_Y, :]) + y_buffer]
            limits.append(
                y_limits
            )

            if show_obstacle:
                # Incorporate obstacle in limits
                limits[0][0] = min(limits[0][0], obstacle_x - 2*obstacle_radius)
                limits[0][1] = max(limits[0][1], obstacle_x + 2*obstacle_radius)
                limits[1][0] = min(limits[1][0], obstacle_y - 2*obstacle_radius)
                limits[1][1] = max(limits[1][1], obstacle_y + 2*obstacle_radius)

            if show_goal:
                # incorporate goal in limits
                goal_radius = 0.1
                limits[0][0] = min(limits[0][0], goal_xy[0, 0] - 2*self.goal_radius)
                limits[0][1] = max(limits[0][1], goal_xy[0, 0] + 2*self.goal_radius)
                limits[1][0] = min(limits[1][0], goal_xy[0, 1] - 2*self.goal_radius)
                limits[1][1] = max(limits[1][1], goal_xy[0, 1] + 2*self.goal_radius)

            # print(limits)

            # End result should be a list of lists (e.g., [[0.0, 0.3], [0.0, 0.3]])

        # Create a figure and an axis.
        fig = plt.figure()
        # ax = fig.add_subplot(111)

        # Plot the initial state.
        x0 = x_trajectory[:, :, 0]
        f0 = f_trajectory[:, :, 0]
        plot_collection = self.plot(
            x0, th,
            limits=limits, hide_axes=hide_axes, current_force=f0,
            show_obstacle=show_obstacle, show_goal=show_goal,
        )

        # This function will modify each of the values of the functions above.
        def update(frame_index):

            for batch_index in range(batch_size):
                x_t_bi = x_trajectory[batch_index, :, frame_index]
                f_t_bi = f_trajectory[batch_index, :, frame_index]

                self.update_plot_objects(
                    plot_collection[batch_index],
                    x_t_bi.flatten(), th[batch_index, :].flatten(),
                    current_force=f_t_bi.flatten())

        # Construct the animation, using the update function as the animation
        # director.
        animation = manimation.FuncAnimation(
            fig, update,
            np.arange(0, num_frames-1), interval=5)

        animation.save(filename=filename, fps=15)


    def plot(self, x: torch.Tensor, theta: torch.Tensor,
             limits: Optional[List[List[float]]] = None,
             hide_axes: bool = True,
             ax: plt.Axes = None,
             equal_aspect: bool = True,
             show_geometric_center: bool = False,
             show_CoM: bool = True,
             show_friction_cone_vectors: bool = True,
             show_obstacle: bool = True,
             show_goal: bool = True,
             current_force: torch.Tensor = None) -> List[plt.Figure]:
        """
        plot
        Description
            Plots all pusher sliders in the batch on screen (be careful with this!)
            x:              A bs x 3 tensor containing the states of bs systems.
            theta:          A bs x 2 tensor containing the parameters of bs systems.
            current_force:  A bs x ps.num_controls tensor containing the inputs of bs systems.
        """

        # Constants
        batch_size = x.shape[0]
        s = self.nominal_scenario
        obstacle_x, obstacle_y, obstacle_radius = s["obstacle_center_x"], s["obstacle_center_y"], s["obstacle_radius"]
        goal_pose = self.goal_point(theta)
        goal_xy = goal_pose[:, :2].to(goal_pose.device)

        # Input Processing
        assert len(x.shape) == 2, f"x is of the wrong dimension. Received tensor of {len(x.shape)} dimensions; expected 2."
        assert x.shape[1] == 3, f"Expected state tensor to have second dimension size 3; received {x.shape[1]}."

        assert x.shape[0] == theta.shape[0], f"The batch size of x ({x.shape[0]}) is different from batch size of theta ({theta.shape[0]})."

        assert len(theta.shape) == 2, f"theta is of the wrong dimension. Received tensor of {len(theta.shape)} dimensions; expected 2."
        assert theta.shape[1] == 2, f"Expected parameter tensor to have second dimension of size 2; received {theta.shape[1]}"

        # Compute Axis Limits
        # Compute Limits
        if limits is None:
            limits = []
            x_buffer = self.s_width
            x_limits = [torch.min(x[:, self.S_X]) - x_buffer,
                        torch.max(x[:, self.S_X]) + x_buffer]
            limits.append(
                x_limits
            )

            y_buffer = self.s_length
            y_limits = [torch.min(x[:, self.S_Y] - y_buffer),
                        torch.max(x[:, self.S_Y]) + y_buffer]
            limits.append(
                y_limits
            )

            if show_obstacle:
                # Incorporate obstacle in limits
                limits[0][0] = min(limits[0][0], obstacle_x - 2*obstacle_radius)
                limits[0][1] = max(limits[0][1], obstacle_x + 2*obstacle_radius)
                limits[1][0] = min(limits[1][0], obstacle_y - 2*obstacle_radius)
                limits[1][1] = max(limits[1][1], obstacle_y + 2*obstacle_radius)

            if show_goal:
                # incorporate goal in limits
                goal_radius = 0.1
                limits[0][0] = min(limits[0][0], goal_xy[0, 0] - 2*self.goal_radius)
                limits[0][1] = max(limits[0][1], goal_xy[0, 0] + 2*self.goal_radius)
                limits[1][0] = min(limits[1][0], goal_xy[0, 1] - 2*self.goal_radius)
                limits[1][1] = max(limits[1][1], goal_xy[0, 1] + 2*self.goal_radius)

            # End result should be a list of lists (e.g., [[0.0, 0.3], [0.0, 0.3]])

        # Algorithm
        fig = plt.gcf()
        if ax is None:
            ax = fig.add_subplot(111)

        plot_objects_collection = []
        for batch_index in range(batch_size):
            x_bi = x[batch_index, :].flatten()
            theta_bi = theta[batch_index, :].flatten()
            f_bi = current_force[batch_index, :].flatten()

            plot_objects_bi = self.plot_single(x_bi, theta_bi,
                ax,
                limits=limits, equal_aspect=equal_aspect,
                hide_axes=hide_axes,
                show_geometric_center=show_geometric_center,
                show_CoM=show_CoM,
                show_friction_cone_vectors=show_friction_cone_vectors,
                show_goal=show_goal and (batch_index == 0),
                show_obstacle=show_obstacle and (batch_index == 0),
                current_force=f_bi,
            )

            plot_objects_collection.append(plot_objects_bi)

        return plot_objects_collection