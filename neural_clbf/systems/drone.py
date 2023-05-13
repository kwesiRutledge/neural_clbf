"""Define a dymamical system for two drones"""
from typing import Tuple, Optional, List

import torch

from .control_affine_system import ControlAffineSystem
from neural_clbf.systems.utils import grav, Scenario, ScenarioList


class Drone(ControlAffineSystem):
    """
    Represents a system of two drones.

    The system has state

        x = [p1x, p1y, v1x, v1y,
            p2x, p2y, v2x, v2y]

    representing the position and velocity of the two drones, and it
    has control inputs

        u = [fx1 fy1 fx2 fy2]

    representing the torque applied.

    The system is parameterized by
        m1: drone 1 mass
        m2: drone 2 mass
    """

    # Number of states and controls
    N_DIMS = 8
    N_CONTROLS = 4

    # State indices
    p1x = 0 
    p1y = 1 
    v1x = 2 
    v1y = 3
    p2x = 4
    p2y = 5
    v2x = 6 
    v2y = 7
    # Control indices
    f1x = 0
    f1y = 1
    f2x = 2
    f2y = 3

    def __init__(
        self,
        nominal_params: Scenario,
        dt: float = 0.01,
        controller_dt: Optional[float] = None,
        scenarios: Optional[ScenarioList] = None,
    ):
        """
        Initialize the drone system.

        args:
            nominal_params: a dictionary giving the parameter values for the system.
                            Requires keys ["m1", "m2"]
            dt: the timestep to use for the simulation
            controller_dt: the timestep for the LQR discretization. Defaults to dt
            secarios: scenarios (If I make them)
        raises:
            ValueError if nominal_params are not valid for this system
        """
        super().__init__(
            nominal_params, dt=dt, controller_dt=controller_dt, scenarios=scenarios
        )

    def validate_params(self, params: Scenario) -> bool:
        """Check if a given set of parameters is valid

        args:
            params: a dictionary giving the parameter values for the system.
                    Requires keys ["m1", "m2"]
        returns:
            True if parameters are valid, False otherwise
        """
        valid = True
        # Make sure all needed parameters were provided
        valid = valid and "m1" in params
        valid = valid and "m2" in params

        # Make sure all parameters are physically valid
        valid = valid and params["m1"] > 0
        valid = valid and params["m2"] > 0

        return valid

    @property
    def n_dims(self) -> int:
        return Drone.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return []

    @property
    def n_controls(self) -> int:
        return Drone.N_CONTROLS


    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.ones(self.n_dims)
        upper_limit[Drone.p1x] = 10.0
        upper_limit[Drone.p1y] = 10.0
        upper_limit[Drone.v1x] = 2.0
        upper_limit[Drone.v1y] = 2.0
        upper_limit[Drone.p2x] = 10.0
        upper_limit[Drone.p2y] = 10.0
        upper_limit[Drone.v2x] = 2.0
        upper_limit[Drone.v2y] = 2.0

        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.ones(self.n_controls)
        upper_limit[Drone.f1x] = 1.0
        upper_limit[Drone.f1y] = 1.0
        upper_limit[Drone.f2x] = 1.0
        upper_limit[Drone.f2y] = 1.0

        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    def goal_point(self):
        return torch.tensor([[9.0, 9.0, 1.0, 1.0, -9.0, -9.0, 0, 0]])

    def safe_mask(self, x):
        """Return the mask of x indicating safe regions for the obstacle task

        args:
            x: a tensor of (batch_size, self.n_dims) points in the state space
        returns:
            a tensor of (batch_size,) booleans indicating whether the corresponding
            point is in this region.
        """
        safe_mask = torch.ones_like(x[:, 0], dtype=torch.bool)
        # Avoid walls
        wall = 10
        wall_mask_x = torch.logical_or(x[:,0] <= wall, x[:,0] >= -wall)
        wall_mask_y = torch.logical_or(x[:,1] <= wall, x[:,1] >= -wall)
        wall_mask_1 = torch.logical_or(wall_mask_x, wall_mask_y)
        safe_mask.logical_and_(wall_mask_1)

        p1x = x[:, Drone.p1x]
        p1y = x[:, Drone.p1y]
        p2x = x[:, Drone.p2x]
        p2y = x[:, Drone.p2y]

        collision = 1.0

        separation = (p1x - p2x) ** 2 + (p1y - p2y) ** 2
        separation = torch.sqrt(separation)
        safe_mask = torch.logical_and(unsafe_mask, separation >= collision)

        return safe_mask

    def unsafe_mask(self, x):
        """Return the mask of x indicating unsafe regions for the obstacle task

        args:
            x: a tensor of (batch_size, self.n_dims) points in the state space
        returns:
            a tensor of (batch_size,) booleans indicating whether the corresponding
            point is in this region.
        """
        unsafe_mask = torch.ones_like(x[:, 0], dtype=torch.bool)
        # Avoid walls
        wall = 10
        wall_mask = abs(x[:,0]) >= wall
        unsafe_mask.logical_and_(wall_mask)
        
        p1x = x[:, Drone.p1x]
        p1y = x[:, Drone.p1y]
        p2x = x[:, Drone.p2x]
        p2y = x[:, Drone.p2y]

        collision = 1.0

        separation = (p1x - p2x) ** 2 + (p1y - p2y) ** 2
        separation = torch.sqrt(separation)
        unsafe_mask = torch.logical_or(unsafe_mask, separation <= collision)

        return unsafe_mask

    def goal_mask(self, x):
        """Return the mask of x indicating points in the goal set

        args:
            x: a tensor of (batch_size, self.n_dims) points in the state space
        returns:
            a tensor of (batch_size,) booleans indicating whether the corresponding
            point is in this region.
        """
        goal_mask = (x - self.goal_point).norm(dim=-1) <= 0.5

        return goal_mask

    def _f(self, x: torch.Tensor, params: Scenario):
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
        f = f.type_as(x)

        # Extract the needed parameters
        m1, m2 = params["m1"], params["m2"]
        # and state variables
        p1x = x[:, Drone.p1x]
        p1y = x[:, Drone.p1y]
        v1x = x[:, Drone.v1x]
        v1y = x[:, Drone.v1y]
        p2x = x[:, Drone.p2x]
        p2y = x[:, Drone.p2y]
        v2x = x[:, Drone.v2x]
        v2y = x[:, Drone.v2y]

        f[:, Drone.p1x, 0] = v1x
        f[:, Drone.p1y, 0] = v1y
        f[:, Drone.v1x, 0] = 0
        f[:, Drone.v1y, 0] = 0
        f[:, Drone.p2x, 0] = v2x
        f[:, Drone.p2y, 0] = v2y
        f[:, Drone.v2x, 0] = 0
        f[:, Drone.v2y, 0] = 0

        return f

    def _g(self, x: torch.Tensor, params: Scenario):
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
        g = g.type_as(x)

        # Extract the needed parameters
        m1, m2 = params["m1"], params["m2"]
        # and state variables
        p1x = x[:, Drone.p1x]
        p1y = x[:, Drone.p1y]
        p2x = x[:, Drone.p2x]
        p2y = x[:, Drone.p2y]

        # Effect on drone 1 acceleration
        g[:, Drone.v1x, Drone.f1x] = 1 / m1
        g[:, Drone.v1y, Drone.f1y] = 1 / m1

        # Effect on drone 2 acceleration
        g[:, Drone.v2x, Drone.f2x] = 1 / m2
        g[:, Drone.v2y, Drone.f2y] = 1 / m2

        return g
