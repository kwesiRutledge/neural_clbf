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

        # Then initialize
        super().__init__(nominal_scenario, Theta, dt, controller_dt)

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
        upper_limit = torch.ones(self.n_dims)
        upper_limit[ScalarCAPA2Demo.X_DEMO] = 10.0

        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)


    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        upper_limit = 10 * torch.ones(ScalarCAPA2Demo.N_CONTROLS)
        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    def safe_mask(self, x: torch.Tensor) -> torch.Tensor:
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
        safe_mask = torch.ones_like(x[:, 0], dtype=torch.bool)

        safe_mask.logical_and_(wall_pos <= x[:, 0])

        return safe_mask

    def unsafe_mask(self, x: torch.Tensor) -> torch.Tensor:
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
        unsafe_mask = torch.zeros_like(x[:, 0], dtype=torch.bool)

        unsafe_mask.logical_and_(wall_pos > x[:, 0])

        return unsafe_mask

    def goal_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Return the mask of x indicating goal regions for this system

        args:
            x: a tensor of (batch_size, self.n_dims) points in the state space
        returns:
            a tensor of (batch_size,) booleans indicating whether the corresponding
            point is in this region.
        """
        # Include a sensible default
        goal_tolerance = 0.1

        return (x - self.goal_point).norm(dim=-1) <= goal_tolerance

    @property
    def goal_point(self):
        goal = torch.zeros((1, self.n_dims))
        goal[0, 0] = 0.0
        return goal

    @property
    def u_eq(self):
        return torch.zeros((1, self.n_controls))

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
        f = torch.zeros((batch_size, self.n_dims, 1))
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
        F = torch.zeros((batch_size, self.n_dims, self.n_params))
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
        g = torch.zeros((batch_size, self.n_dims, self.n_controls))
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
        G = torch.zeros((batch_size, self.n_dims, self.n_controls, self.n_params))

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
