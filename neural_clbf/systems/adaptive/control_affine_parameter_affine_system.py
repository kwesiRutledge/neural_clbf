"""Define an abstract base class for dymamical systems"""
from abc import (
    ABC,
    abstractmethod,
    abstractproperty,
)
from typing import Callable, Tuple, Optional, List

from matplotlib.axes import Axes
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd.functional import jacobian

from neural_clbf.systems.utils import (
    Scenario,
    ScenarioList,
    lqr,
    robust_continuous_lyap,
    continuous_lyap,
)

import polytope as pc

class ControlAffineParameterAffineSystem(ABC):
    """
    Represents an abstract control-affine, parameter-affine dynamical system.

    A control-affine, parameter-affine dynamical system is one where the state derivatives are affine in
    the control input, e.g.:

        dx/dt = f(x) + F(x) * theta + ( g(x) + \sum_i G_i(x) \theta) u

    These can be used to represent a wide range of dynamical systems, and they have some
    useful properties when it comes to designing controllers.
    """

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
        super().__init__()

        # Validate parameters, raise error if they're not valid
        if not self.validate_scenario(nominal_scenario):
            raise ValueError(f"Scenario not valid: {nominal_scenario}")

        self.nominal_scenario = nominal_scenario

        # Make sure the timestep is valid
        assert dt > 0.0
        self.dt = dt

        if controller_dt is None:
            controller_dt = self.dt
        self.controller_dt = controller_dt

        self.Theta = Theta

        if theta is None:
            theta = self.get_N_samples_from_polytope(self.Theta, 1) # Return single vector representing unknown parameters.
        self.theta = theta

        # Compute the linearized controller
        if use_linearized_controller:
            self.compute_linearized_controller(scenarios)

        self.device = device

    @torch.enable_grad()
    def compute_A_matrix(self, theta: torch.Tensor, scenario: Optional[Scenario]) -> np.ndarray:
        """
        Description:
            Compute the linearized continuous-time state-state derivative transfer matrix
            about the goal point

        args:
            theta: 1 x n_params tensor describing the current parameters

        """
        # Constants
        if theta is None:
            theta = self.theta

        # Linearize the system about the x = 0, u = 0
        x0 = self.goal_point(theta)
        u0 = self.u_eq
        dynamics = lambda x: self.closed_loop_dynamics(x, u0, theta, scenario).squeeze()
        A = jacobian(dynamics, x0).squeeze().cpu().numpy()
        A = np.reshape(A, (self.n_dims, self.n_dims))

        return A

    def compute_B_matrix(self, theta: torch.Tensor, scenario: Optional[Scenario]) -> np.ndarray:
        """
        Compute the linearized continuous-time state-state derivative transfer matrix
        about the goal point

        args:
            theta: 1 x n_params tensor describing the current parameters
        """
        # Defaults
        if scenario is None:
            scenario = self.nominal_scenario

        # Constants

        # Linearize the system about the x = 0, u = 0
        x0 = self.goal_point(theta)
        g_like = self.input_gain_matrix(x0, theta, scenario)

        B = g_like.squeeze().cpu().numpy()
        B = np.reshape(B, (self.n_dims, self.n_controls))

        return B

    def linearized_ct_dynamics_matrices(
        self, scenario: Optional[Scenario] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the continuous time linear dynamics matrices, dx/dt = Ax + Bu"""
        A = self.compute_A_matrix(
            torch.Tensor(self.theta).to(self.device).reshape((1, self.n_params)),
            scenario,
        )
        B = self.compute_B_matrix(
            torch.Tensor(self.theta).to(self.device).reshape((1, self.n_params)),
            scenario,
        )

        return A, B

    def linearized_dt_dynamics_matrices(
        self, scenario: Optional[Scenario] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the continuous time linear dynamics matrices, x_{t+1} = Ax_{t} + Bu
        """
        Act, Bct = self.linearized_ct_dynamics_matrices(scenario)
        A = np.eye(self.n_dims) + self.controller_dt * Act
        B = self.controller_dt * Bct

        return A, B

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
            self.K = torch.from_numpy(np.float32(K_np)).to(device=self.device)

            Acl_list.append(Act - Bct @ K_np)

        # If more than one scenario is provided...
        # get the Lyapunov matrix by robustly solving Lyapunov inequalities
        if len(scenarios) > 1:
            if self.device == "mps":
                self.P = torch.tensor(
                    np.float32(robust_continuous_lyap(Acl_list, Q)),
                    device=self.device,
                )
            else:
                self.P = torch.tensor(
                    robust_continuous_lyap(Acl_list, Q),
                    device=self.device,
                )
        else:
            # Otherwise, just use the standard Lyapunov equation
            if self.device == "mps":
                self.P = torch.tensor(
                    np.float32(continuous_lyap(Acl_list[0], Q)),
                    device=self.device,
                )
            else:
                self.P = torch.tensor(
                    continuous_lyap(Acl_list[0], Q),
                    device=self.device,
                )

    @abstractmethod
    def validate_scenario(self, s: Scenario) -> bool:
        """Check if a given set of parameters is valid

        args:
            params: a dictionary giving the parameter values for the system.
        returns:
            True if parameters are valid, False otherwise
        """
        pass

    @abstractproperty
    def n_dims(self) -> int:
        pass

    @abstractproperty
    def angle_dims(self) -> List[int]:
        pass

    @abstractproperty
    def n_controls(self) -> int:
        pass

    @abstractproperty
    def n_params(self) -> int:
        pass

    @abstractproperty
    def parameter_angle_dims(self) -> List[int]:
        pass

    @abstractproperty
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        pass

    @abstractproperty
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        pass

    @property
    def intervention_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable changes to
        control for this system
        """
        upper_limit, lower_limit = self.control_limits

        return (upper_limit, lower_limit)

    def out_of_bounds_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Return the mask of x indicating whether rows are outside the state limits
        for this system

        args:
            x: a tensor of (batch_size, self.n_dims) points in the state space
        returns:
            a tensor of (batch_size,) booleans indicating whether the corresponding
            point is in this region.
        """
        upper_lim, lower_lim = self.state_limits
        out_of_bounds_mask = torch.zeros_like(x[:, 0], dtype=torch.bool, device=self.device)
        for i_dim in range(x.shape[-1]):
            out_of_bounds_mask.logical_or_(x[:, i_dim] >= upper_lim[i_dim])
            out_of_bounds_mask.logical_or_(x[:, i_dim] <= lower_lim[i_dim])

        return out_of_bounds_mask

    @abstractmethod
    def safe_mask(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Return the mask of x indicating safe regions for this system

        args:
            x: a tensor of (batch_size, self.n_dims) points in the state space
            theta: a tensor of (batch_size, self.n_params) points in the state space which exactly map to the states of the system
                    (i.e., theta[1, :] is the parameters of the system at state x[1, :])
        returns:
            a tensor of (batch_size,) booleans indicating whether the corresponding
            point is in this region.
        """
        pass

    @abstractmethod
    def unsafe_mask(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Return the mask of x indicating unsafe regions for this system

        args:
            x: a tensor of (batch_size, self.n_dims) points in the state space
            theta: a tensor of (batch_size, self.n_params) points in the state space which exactly map to the states of the system
                    (i.e., theta[1, :] is the parameters of the system at state x[1, :])
        returns:
            a tensor of (batch_size,) booleans indicating whether the corresponding
            point is in this region.
        """
        pass

    def failure(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Return the mask of x indicating failure. This usually matches with the
        unsafe region

        args:
            x: a tensor of (batch_size, self.n_dims) points in the state space
        returns:
            a tensor of (batch_size,) booleans indicating whether the corresponding
            point is in this region.
        """
        return self.unsafe_mask(x, theta)

    def boundary_mask(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Return the mask of x indicating regions that are neither safe nor unsafe

        args:
            x: a tensor of (batch_size, self.n_dims) points in the state space
        returns:
            a tensor of (batch_size,) booleans indicating whether the corresponding
            point is in this region.
        """
        return torch.logical_not(
            torch.logical_or(
                self.safe_mask(x, theta),
                self.unsafe_mask(x, theta),
            )
        )

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
        # constants
        if theta is None:
            batch_size = 1
        else:
            batch_size = theta.shape[0]

        return torch.zeros((batch_size, self.n_dims), device=self.device)

    @property
    def u_eq(self):
        return torch.zeros((1, self.n_controls), device=self.device)

    def sample_state_space(self, num_samples: int) -> torch.Tensor:
        """
        Description:
            Sample uniformly from the state space

        Outputs:
            x: num_samples x self.n_dims
        """
        x_max, x_min = self.state_limits

        # Sample uniformly from 0 to 1 and then shift and scale to match state limits
        x = torch.zeros((num_samples, self.n_dims), device=self.device).uniform_(0.0, 1.0)
        for i in range(self.n_dims):
            x[:, i] = x[:, i] * (x_max[i] - x_min[i]) + x_min[i]

        return x

    def sample_Theta_space(self, num_samples: int) -> torch.Tensor:
        """
        Description:
            Sample uniformly from the Theta space
        Outputs:
            theta_samples: N_samples x self.n_params
        """

        theta_samples_np = self.get_N_samples_from_polytope(self.Theta, num_samples)
        if torch.get_default_dtype() == torch.float32:
            theta_samples_np = np.float32(theta_samples_np)

        return torch.tensor(theta_samples_np, device=self.device)

    def sample_with_mask(
        self,
        num_samples: int,
        mask_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        max_tries: int = 5000,
    ) -> [torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample num_samples so that mask_fn is True for all samples. Makes a
        best-effort attempt, but gives up after max_tries, so may return some points
        for which the mask is False, so watch out!
        """
        # Get a uniform sampling
        x_samples = self.sample_state_space(num_samples)
        theta_samples = self.sample_Theta_space(num_samples)

        samples = torch.cat((x_samples, theta_samples), dim=1)

        # While the mask is violated, get violators and replace them
        # (give up after so many tries)
        for _ in range(max_tries):
            violations = torch.logical_not(mask_fn(x_samples, theta_samples))
            if not violations.any():
                break

            new_samples = int(violations.sum().item())
            x_samples[violations] = self.sample_state_space(new_samples)
            theta_samples[violations] = self.sample_Theta_space(new_samples)
            samples[violations] = torch.cat((x_samples[violations], theta_samples[violations]), dim=1)

        return samples, x_samples, theta_samples

    def sample_safe(self, num_samples: int, max_tries: int = 5000) -> [torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample uniformly from the safe space. May return some points that are not
        safe, so watch out (only a best-effort sampling).
        """
        return self.sample_with_mask(num_samples, self.safe_mask, max_tries)

    def sample_unsafe(self, num_samples: int, max_tries: int = 5000) -> [torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample uniformly from the unsafe space. May return some points that are not
        unsafe, so watch out (only a best-effort sampling).
        """
        return self.sample_with_mask(num_samples, self.unsafe_mask, max_tries)

    def sample_goal(self, num_samples: int, max_tries: int = 5000) -> [torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample uniformly from the goal. May return some points that are not in the
        goal, so watch out (only a best-effort sampling).
        """
        return self.sample_with_mask(num_samples, self.goal_mask, max_tries)

    def sample_boundary(self, num_samples: int, max_tries: int = 5000) -> [torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample uniformly from the state space between the safe and unsafe regions.
        May return some points that are not in this region safe, so watch out (only a
        best-effort sampling).
        """
        return self.sample_with_mask(num_samples, self.boundary_mask, max_tries)

    def control_affine_dynamics(
        self, x: torch.Tensor, theta: torch.Tensor,
        params: Optional[Scenario] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (f + F \theta, g + \sum_i  G_i \theta) representing the system dynamics in control-affine form:

            dx/dt = f(x) + F(x) \theta + { g(x) + \sum_i G(x) \theta_i } u

        args:
            x: bs x self.n_dims tensor of state
            theta: bs x self.n_params tensor of parameter data
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            f: bs x self.n_dims x 1 tensor representing the control-independent dynamics
            g: bs x self.n_dims x self.n_controls tensor representing the control-
               dependent dynamics
        """
        # Sanity check on input
        assert x.ndim == 2
        assert x.shape[1] == self.n_dims

        # Constants
        batch_size = x.shape[0]

        # If no params required, use nominal params
        if params is None:
            params = self.nominal_scenario

        theta_reshape = theta.reshape((theta.shape[0], theta.shape[1], 1))

        # f_like = torch.zeros((batch_size, self.n_dims, 1))
        # f_like = self._f(x, params) + torch.bmm(self._F(x, params), theta_reshape)

        g_like = torch.zeros((batch_size, self.n_dims, self.n_controls), device=self.device)
        g_like = self.input_gain_matrix(x, theta_reshape, params)

        F_x_params = self._F(x, params)

        return self._f(x, params) + torch.bmm(F_x_params, theta_reshape), self.input_gain_matrix(x, theta_reshape, params)

    def closed_loop_dynamics(
        self, x: torch.Tensor, u: torch.Tensor, theta: torch.Tensor, params: Optional[Scenario] = None
    ) -> torch.Tensor:
        """
        Return the state derivatives at state x and control input u

            dx/dt = f(x) + F(x) \theta + { g(x) + sum_i G(x) \theta_i } u

        args:
            x: bs x self.n_dims tensor of state
            u: bs x self.n_controls tensor of controls
            theta: bs x self.n_params tensor of parameters
            scenario: a dictionary giving the scenario parameter values for the system.
                        If None, default to the nominal parameters used at initialization
        returns:
            xdot: bs x self.n_dims tensor of time derivatives of x
        """
        # Get the control-affine dynamics
        f, g = self.control_affine_dynamics(x, theta, params=params)
        # print("f", f.shape)
        # print("g", g.shape)
        # print("u", u.shape)
        # Compute state derivatives using control-affine form
        xdot = f + torch.bmm(g, u.unsqueeze(-1))
        return xdot.view(x.shape)

    def zero_order_hold(
        self,
        x: torch.Tensor,
        theta: torch.Tensor,
        u: torch.Tensor,
        controller_dt: float,
        params: Optional[Scenario] = None,
    ) -> torch.Tensor:
        """
        Simulate dynamics forward for controller_dt, simulating at self.dt, with control
        held constant at u, starting from x.

        args:
            x: bs x self.n_dims tensor of state
            theta: bs x self.n_params tensor of parameters
            u: bs x self.n_controls tensor of controls
            controller_dt: the amount of time to hold for
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            x_next: bs x self.n_dims tensor of next states
        """
        num_steps = int(controller_dt / self.dt)
        for tstep in range(0, num_steps):
            # Get the derivatives for this control input
            xdot = self.closed_loop_dynamics(x, u, theta, params)

            # Simulate forward
            x = x + self.dt * xdot

        # Return the simulated state
        return x

    def simulate(
        self,
        x_init: torch.Tensor,
        theta: torch.Tensor,
        num_steps: int,
        controller: Callable[[torch.Tensor,torch.Tensor], torch.Tensor],
        controller_period: Optional[float] = None,
        guard: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        params: Optional[Scenario] = None,
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Simulate the system for the specified number of steps using the given controller

        args:
            x_init - bs x n_dims tensor of initial conditions
            theta - bs x n_params tensor of parameters
            num_steps - a positive integer
            controller - a mapping from state to control action
            controller_period - the period determining how often the controller is run
                                (in seconds). If none, defaults to self.dt
            guard - a function that takes a bs x n_dims tensor and returns a length bs
                    mask that's True for any trajectories that should be reset to x_init
            params - a dictionary giving the parameter values for the system. If None,
                     default to the nominal parameters used at initialization
        returns
            x_sim - bs x num_steps x self.n_dims tensor of simulated trajectories. If an error
                    occurs on any trajectory, the simulation of all trajectories will stop and
                    the second dimension will be less than num_steps
            th_sim -    bs x num_steps x self.n_params tensor of randomly chosen parameters
                        for each trajectory. Parameters should not change.
            th_h_sim -  bs x num_steps x self.n_params tensor of estimated parameters.
                        Estimate may change.
        usage
            x_sim, th_sim, th_h_sim = simulate(x, theta, N_sim, silly_control, 0.01)
        """
        # Create a tensor to hold the simulation results
        batch_size = x_init.shape[0]

        n_dims = self.n_dims
        n_params = self.n_params

        # P = self.P
        # P = P.reshape(
        #     self.n_dims + self.n_params,
        #     self.n_dims + self.n_params
        # )

        # Set up Simulator Variables
        x_sim = torch.zeros(batch_size, num_steps, self.n_dims, device=self.device).type_as(x_init)
        x_sim[:, 0, :] = x_init

        th_sim = torch.zeros(batch_size, num_steps, self.n_params, device=self.device, ).type_as(theta)
        th_sim[:, 0, :] = theta

        th_h_sim = torch.zeros(batch_size, num_steps, self.n_params, device=self.device).type_as(theta)
        th_h_samples = self.sample_Theta_space(batch_size) #self.get_N_samples_from_polytope(self.Theta, batch_size)
        th_h_sim[:, 0, :] = torch.tensor(th_h_samples, device=self.device).type_as(theta)

        u = torch.zeros(x_init.shape[0], self.n_controls, device=self.device).type_as(x_init)

        # Compute controller update frequency
        if controller_period is None:
            controller_period = self.dt
        controller_update_freq = int(controller_period / self.dt)

        # Run the simulation until it's over or an error occurs
        t_sim_final = 0
        for tstep in range(1, num_steps):
            try:
                # Get the current state, theta and theta_hat
                x_current = x_sim[:, tstep - 1, :]
                theta_current = th_sim[:, tstep - 1, :]
                theta_hat_current = th_h_sim[:, tstep - 1, :]

                # Get the control input at the current state if it's time
                if tstep == 1 or tstep % controller_update_freq == 0:
                    u = controller(x_current, theta_hat_current)

                # Simulate forward using the dynamics
                xdot = self.closed_loop_dynamics(x_current, u, theta, params)
                x_sim[:, tstep, :] = x_current + self.dt * xdot
                th_sim[:, tstep, :] = theta_current

                # Compute theta hat evolution
                th_h_dot = torch.zeros(theta_hat_current.shape, device=self.device).type_as(theta_hat_current) # TODO: Try to implement Least Squares for this.
                th_h_sim[:, tstep, :] = theta_hat_current + self.dt * th_h_dot

                # If the guard is activated for any trajectory, reset that trajectory
                # to a random state
                if guard is not None:
                    guard_activations = guard(x_sim[:, tstep, :])
                    n_to_resample = int(guard_activations.sum().item())
                    x_new = self.sample_state_space(n_to_resample).type_as(x_sim)
                    x_sim[guard_activations, tstep, :] = x_new

                # Update the final simulation time if the step was successful
                t_sim_final = tstep
            except ValueError:
                break

        return x_sim[:, : t_sim_final + 1, :], th_sim[:, : t_sim_final + 1, :], th_h_sim[:, : t_sim_final + 1, :]

    def nominal_simulator(self, x_init: torch.Tensor, theta: torch.Tensor, num_steps: int) -> torch.Tensor:
        """
        Simulate the system forward using the nominal controller

        args:
            x_init - bs x n_dims tensor of initial conditions
            theta - bs x n_params tensor of parameters
            num_steps - a positive integer
        returns
            x_sim - bs x num_steps x self.n_dims tensor of simulated trajectories. If an error
                    occurs on any trajectory, the simulation of all trajectories will stop and
                    the second dimension will be less than num_steps
            th_sim -    bs x num_steps x self.n_params tensor of randomly chosen parameters
                        for each trajectory. Parameters should not change.
            th_h_sim -  bs x num_steps x self.n_params tensor of estimated parameters.
                        Estimate may change.
        """
        # Call the simulate method using the nominal controller
        return self.simulate(
            x_init, theta, num_steps, self.u_nominal, guard=self.out_of_bounds_mask
        )

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    def input_gain_matrix(self, x: torch.Tensor, theta: torch.Tensor, params: Scenario) -> torch.Tensor:
        """
        Return the factor that multiplies the control value in the dynamics.

        args:
            x: bs x self.n_Dims tensor of state
            theta: bs x self.n_params x 1 tensor of state
            params: a dictionary giving the parameter values for the system.
                    If None, default to the nominal parameters used at initialization.
        returns
            g_like: bs x self.n_dims x self.n_controls tensor defining how input vector impacts the state
                    in each batch.
        """
        # Constants
        batch_size = x.shape[0]
        if len(theta.shape) == 2:
            theta = theta.reshape((theta.shape[0], theta.shape[1], 1))

        # Algorithm
        g_like = self._g(x, params)
        G = self._G(x, params)
        for param_index in range(self.n_params):

            theta_i = theta[:, param_index, 0].reshape((batch_size, 1, 1))
            G_i = torch.zeros((batch_size, self.n_dims, self.n_controls), device= self.device)
            G_i[:, :, :] = G[:, :, :, param_index]
            # Update g
            g_like = g_like + torch.mul(theta_i, G_i)

        return g_like

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
        goals = self.goal_point(theta_hat).type_as(x)
        u_nominal = -(K @ (x - goals).T).T

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

    def get_N_samples_from_polytope(self, P: pc.Polytope, N_samples):
        """
        get_N_samples_from_polytope
        Description:
            This function retrieves N samples from the polytope P.
            Used to more efficiently produce samples (only have to compute extremes once.)

        Returns:
            N_samples x P.Dim array containing all of the samples
        """

        # Compute V Representation
        V = pc.extreme(P)

        # print(V is None)
        if V is None:
            # I suspect this means that the polytope contains a singleton.
            # Select the element at the boundary to get the correct sample.
            return P.b[:P.dim].reshape((P.dim, 1))

        n_V = V.shape[0]

        # Random Variable
        comb_rand_var = np.random.exponential(size=(n_V, N_samples))
        for sample_index in range(N_samples):
            comb_rand_var[:, sample_index] = comb_rand_var[:, sample_index] / np.sum(
                comb_rand_var[:, sample_index])

        return np.dot(V.T, comb_rand_var).T

    def compute_simple_aCLF_estimator_dynamics(self, x:torch.Tensor, theta_hat:torch.Tensor, params:Scenario):
        """

        """
        # Constants
        batch_size = x.shape[0]

        n_dims = self.n_dims
        n_params = self.n_params

        # Compute Jacobian
        JthV = F.linear(theta_hat_current, P[n_dims:n_dims + n_params, n_dims:n_dims + n_params]) + \
               2 * F.linear(x, P[:n_dims, n_dims:n_dims + n_params])
        JthV = JthV.reshape(x_theta.shape[0], 1, n_params)

        # Get Lie Derivatives of stuff
        raise("Not yet implemented!") # TODO: Implement this function!

        Gamma = torch.eye(n_dims, device=self.device)