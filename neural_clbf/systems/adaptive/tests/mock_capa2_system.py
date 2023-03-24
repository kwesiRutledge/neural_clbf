"""
Define a mock ControlAffineParameterAffineSystem for testing use
"""

from typing import Tuple, List

import torch

from neural_clbf.systems.adaptive import ControlAffineParameterAffineSystem
from neural_clbf.systems.utils import Scenario

import polytope as pc


class MockCAPA2System(ControlAffineParameterAffineSystem):
    """
    Represents a mock capa2 system.
    """

    # Number of states and controls
    N_DIMS = 2
    N_CONTROLS = 2
    N_PARAMETERS = 2

    def __init__(
        self,
        nominal_scenario: Scenario,
        Theta: pc.Polytope,
        dt: float = 0.01,
        controller_dt: float = None,
        device: str = "cpu",
    ):
        """
        Initialize the mock system.

        args:
            nominal_params: a dictionary giving the parameter values for the system.
                    Requires no keys
        """

        self.device = device

        super().__init__(
            nominal_scenario, Theta, dt, controller_dt,
            device=device,
        )

    def validate_scenario(self, params: Scenario) -> bool:
        """Check if a given set of parameters is valid

        args:
            params: a dictionary giving the parameter values for the system.
                    Requires keys ["m", "I", "r"]
        returns:
            True if parameters are valid, False otherwise
        """
        # Nothing to validate for the mock system
        return True

    @property
    def n_dims(self) -> int:
        return MockCAPA2System.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return [1]

    @property
    def parameter_angle_dims(self) -> List[int]:
        return [0]

    @property
    def n_controls(self) -> int:
        return MockCAPA2System.N_CONTROLS

    @property
    def n_params(self) -> int:
        return MockCAPA2System.N_PARAMETERS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # define upper and lower limits based around the nominal equilibrium input
        lower_limit = -1.0 * torch.ones(self.n_dims)
        upper_limit = 10.0 * torch.ones(self.n_dims)

        return (upper_limit, lower_limit)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.tensor([1.0, 1.0])
        lower_limit = torch.tensor([-1.0, -1.0])

        return (upper_limit, lower_limit)

    def u_nominal(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the nominal (e.g. LQR or proportional) control for the nominal
        parameters. MockSystem just returns a zero input

        args:
            x: bs x self.n_dims tensor of state
        returns:
            u_nominal: bs x self.n_controls tensor of controls
        """
        batch_size = x.shape[0]
        u_nominal = torch.zeros((batch_size, self.n_controls)).type_as(x)

        return u_nominal

    def _f(self, x: torch.Tensor, params: Scenario)->torch.Tensor:
        """
        Return the control-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            f: bs x self.n_dims x 1 tensor
        """
        # Extract batch size and set up a tensor for holding the result
        batch_size = x.shape[0]
        f = torch.zeros((batch_size, self.n_dims, 1))

        # Mock dynamics
        f[:, 0, 0] = 1.0
        f[:, 1, 0] = 2.0

        return f

    def _F(self, x: torch.Tensor, params: Scenario)->torch.Tensor:
        """
        _F
        Description:
            Return the control-independent, but parameter dependent part of the differential equation.
        """
        # Constants
        batch_size = x.shape[0]
        F = torch.zeros((batch_size, self.n_dims, self.n_params))

        # Mock dynamics
        F[:, 0, 0] = 3.0
        F[:, 1, 1] = 5.0

        return F

    def _g(self, x: torch.Tensor, params: Scenario)->torch.Tensor:
        """
        Return the control-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            g: bs x self.n_dims x self.n_controls tensor
        """
        # Extract batch size and set up a tensor for holding the result
        batch_size = x.shape[0]
        g = torch.zeros((batch_size, self.n_dims, self.n_controls))

        # Mock dynamics
        g[:, 0, 0] = 1.0
        g[:, 0, 1] = 2.0
        g[:, 1, 0] = 3.0
        g[:, 1, 1] = 4.0

        return g

    def _G(self, x:torch.Tensor, params:Scenario)->torch.Tensor:
        """
        _G
        Description
            Return the control-dependent, but parameter dependent part of the control-affine dynamics.
        """
        # Constants
        batch_size = x.shape[0]
        G = torch.zeros((batch_size, self.n_dims, self.n_params, self.n_controls))

        # Mock dynamics
        G[:, 0, 0, 0] = 7.0
        G[:, 0, 0, 1] = 11.0
        G[:, 1, 1, 0] = 13.0
        G[:, 1, 1, 1] = 17.0

        return G

    def safe_mask(self, x):
        """Return the mask of x indicating safe regions for the GCAS

        args:
            x: a tensor of points in the state space
        """
        safe_mask = x[:, 0] >= 0

        return safe_mask

    def unsafe_mask(self, x):
        """Return the mask of x indicating unsafe regions for the obstacle task

        args:
            x: a tensor of points in the state space
        """
        unsafe_mask = x[:, 0] <= 0

        return unsafe_mask

    def goal_mask(self, x):
        """Return the mask of x indicating points in the goal set (within 0.2 m of the
        goal).

        args:
            x: a tensor of points in the state space
        """
        goal_mask = x[:, 0].abs() <= 0.1

        return goal_mask
