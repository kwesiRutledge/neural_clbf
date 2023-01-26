"""
load_sharing_manipulator.py
Description:
    Define an instance of class ControlAffineParameterAffineSystem for the system of
    a manipulator attempting to move a specified load
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

class LoadSharingManipulator(ControlAffineParameterAffineSystem):
    """
    Represents a manipulator imparting forces along with a human (linear controller)
    with the goal of reaching a point in space desired by the human.

    The system has state defined in the 3D cartesian frame only in terms of the translational
    coordinates.

        x = [ p_x , p_y , p_z, v_x, v_y, v_z]

    where p_x, p_y, and p_z are the x-, y-, and z- position of the manipulator's end effector in
    free space, v_x, v_y, and v_z are the x-, y- and z- velocities of the manipulator's end effector
    in free space.

    The control inputs are:

        u = [ F_x, F_y, F_z ]

    representing the force (in Newtons) of the robot end effector in the three axes of motion.
    """

    # Number of states, controls and parameters
    N_DIMS = 6
    N_CONTROLS = 3
    N_PARAMETERS = 3

    # State Indices
    P_X = 0
    P_Y = 1
    P_Z = 2
    V_X = 3
    V_Y = 4
    V_Z = 5

    # Control indices
    F_X = 0
    F_Y = 1
    F_Z = 2

    # Parameter indices
    P_X_DES = 0
    P_Y_DES = 1
    P_Z_DES = 2

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
        self.m = 10.0 # kg

        self.K_x = float(-1.0)
        self.K_y = float(-1.0)
        self.K_z = float(-1.0)

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

        valid = valid and ("obstacle_center" in s)
        valid = valid and ("obstacle_width" in s)

        return valid

    @property
    def n_dims(self) -> int:
        return LoadSharingManipulator.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return []

    @property
    def parameter_angle_dims(self) -> List[int]:
        return []

    @property
    def n_controls(self) -> int:
        return LoadSharingManipulator.N_CONTROLS

    @property
    def n_params(self) -> int:
        return LoadSharingManipulator.N_PARAMETERS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # Define Upper and lower values
        upper_limit = torch.ones(self.n_dims).to(self.device)
        upper_limit[LoadSharingManipulator.P_X] = 1.0
        upper_limit[LoadSharingManipulator.P_Y] = 1.0
        upper_limit[LoadSharingManipulator.P_Z] = 1.0
        upper_limit[LoadSharingManipulator.V_X] = 1.0
        upper_limit[LoadSharingManipulator.V_Y] = 1.0
        upper_limit[LoadSharingManipulator.V_Z] = 1.0

        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)


    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        upper_limit = 10 * torch.ones(LoadSharingManipulator.N_CONTROLS)
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
        obst_center = torch.Tensor(self.nominal_scenario["obstacle_center"]).to(self.device)
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
        obst_center = torch.Tensor(self.nominal_scenario["obstacle_center"]).to(self.device)
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
        goal_tolerance = 0.1
        batch_size = x.shape[0]

        # Algorithm
        r = torch.zeros(batch_size, 3).to(self.device) #get position from state
        r[:, :] = x[:, :3]

        return (r - theta).norm(dim=-1) <= goal_tolerance

    @property
    def goal_point(self):
        goal = torch.zeros((1, self.n_dims)).to(self.device)
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
        obst_center = np.array(self.nominal_scenario["obstacle_center"])
        obst_width = self.nominal_scenario["obstacle_width"]

        state_space_obst_center = np.zeros(6)
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

        K_x = self.K_x
        K_y = self.K_y
        K_z = self.K_z
        m = self.m

        g = 10.0 # m/s^2

        # Algorithm
        p_x = x[:, LoadSharingManipulator.P_X]
        p_y = x[:, LoadSharingManipulator.P_Y]
        p_z = x[:, LoadSharingManipulator.P_Z]
        v_x = x[:, LoadSharingManipulator.V_X]
        v_y = x[:, LoadSharingManipulator.V_Y]
        v_z = x[:, LoadSharingManipulator.V_Z]

        f[:, LoadSharingManipulator.P_X, 0] = v_x
        f[:, LoadSharingManipulator.P_Y, 0] = v_y
        f[:, LoadSharingManipulator.P_Z, 0] = v_z
        f[:, LoadSharingManipulator.V_X, 0] = (1/m) * K_x * p_x
        f[:, LoadSharingManipulator.V_Y, 0] = (1 / m) * K_y * p_y - g
        f[:, LoadSharingManipulator.V_Z, 0] = (1 / m) * K_z * p_z

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

        K_x = self.K_x
        K_y = self.K_y
        K_z = self.K_z
        m = self.m

        # Algorithm
        F[:, LoadSharingManipulator.V_X, LoadSharingManipulator.P_X_DES] = - K_x / m
        F[:, LoadSharingManipulator.V_Y, LoadSharingManipulator.P_Y_DES] = - K_y / m
        F[:, LoadSharingManipulator.V_Z, LoadSharingManipulator.P_Z_DES] = - K_z / m

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
        g[:, LoadSharingManipulator.V_X, LoadSharingManipulator.F_X] = (1 / m)
        g[:, LoadSharingManipulator.V_Y, LoadSharingManipulator.F_Y] = (1 / m)
        g[:, LoadSharingManipulator.V_Z, LoadSharingManipulator.F_Z] = (1 / m)

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
        self, x: torch.Tensor, params: Optional[Scenario] = None
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
        goal = self.goal_point.squeeze().type_as(x)
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
        unless overidden.

        args:
            ax: the axis on which to plot
        """
        pass
