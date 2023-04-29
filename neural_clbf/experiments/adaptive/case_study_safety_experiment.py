"""
case_study_safety_experiment.py
Description
    Simulate a rollout and plot in state space
"""
import random
import time
from typing import List, Tuple, Optional, TYPE_CHECKING, Dict, Callable, Any

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
import seaborn as sns
import torch
import tqdm

from neural_clbf.experiments.adaptive.safety_case_study import (
    timing_data_to_latex_table, get_avg_computation_time_from_df,
    plot_rollouts, plot_error_to_goal, create_initial_states_parameters_and_estimates,
)

import control as ct
import control.optimal as opt

from neural_clbf.experiments import Experiment
from neural_clbf.systems.adaptive import (
    ControlAffineParameterAffineSystem,
)

import numpy as np

if TYPE_CHECKING:
    from neural_clbf.controllers import Controller, NeuralObsBFController  # noqa
    from neural_clbf.systems import ObservableSystem  # noqa


class CaseStudySafetyExperiment(Experiment):
    """
    experim = CaseStudySafetyExperiment("Case Study 1", start_x, n_sims_per_start=5, t_sim=5.0)
    experim = CaseStudySafetyExperiment("Case Study 2", start_x, n_sims_per_start=5, t_sim=5.0, x_indices=[0, 1], x_labels=["x", "y"])

    Description:
        An experiment for plotting rollout performance of controllers.

        Plots trajectories projected onto a 2D plane.
    """

    def __init__(
        self,
        name: str,
        start_x: torch.Tensor,
        start_theta_hat: torch.Tensor = None,
        n_sims_per_start: int = 5,
        t_sim: float = 5.0,
        plot_x_indices: List[int] = [],
        plot_x_labels: List[str] = [],
    ):
        """
        Description:
            Initialize an experiment for simulating controller performance.

        args:
            name: the name of this experiment
            plot_x_index: the index of the state dimension to plot on the x axis,
            plot_x_label: the label of the state dimension to plot on the x axis,
            scenarios: a list of parameter scenarios to sample from. If None, use the
                       nominal parameters of the controller's dynamical system
            n_sims_per_start: the number of simulations to run (with random parameters),
                              per row in start_x
            t_sim: the amount of time to simulate for
        """
        super(CaseStudySafetyExperiment, self).__init__(name)

        # Save parameters
        self.start_x = start_x
        self.start_theta_hat = start_theta_hat
        self.n_sims_per_start = n_sims_per_start
        self.t_sim = t_sim

        self.plot_x_indices = plot_x_indices
        self.plot_x_labels = plot_x_labels

        # if "x" in self.plot_x_labels:
        #     raise "There could be a problem using the plot_x_label value x; try using a different value"

    @torch.no_grad()
    def run(self, controller_under_test: "Controller") -> pd.DataFrame:
        return self.run_aclbf_controlled(controller_under_test)

    @torch.no_grad()
    def run_aclbf_controlled(self, controller_under_test: "Controller") -> pd.DataFrame:
        """
        Run the experiment, likely by evaluating the controller, but the experiment
        has freedom to call other functions of the controller as necessary (if these
        functions are not supported by all controllers, then experiments will be
        responsible for checking compatibility with the provided controller)

        args:
            controller_under_test: the controller with which to run the experiment
        returns:
            a pandas DataFrame containing the results of the experiment, in tidy data
            format (i.e. each row should correspond to a single observation from the
            experiment).
        """
        # Constants
        scenarios = [controller_under_test.dynamics_model.nominal_scenario]
        results = []  # Set up a dataframe to store the results

        # Compute the number of simulations to run
        n_sims = self.n_sims_per_start * self.start_x.shape[0]

        # Determine the parameter range to sample from
        parameter_ranges = {}
        for param_name in scenarios[0].keys():
            param_max = max([s[param_name] for s in scenarios])
            param_min = min([s[param_name] for s in scenarios])
            parameter_ranges[param_name] = (param_min, param_max)

        # Generate a tensor of start states
        n_dims = controller_under_test.dynamics_model.n_dims
        n_controls = controller_under_test.dynamics_model.n_controls
        n_theta = controller_under_test.dynamics_model.n_params
        Theta = controller_under_test.dynamics_model.Theta

        x_sim_start, theta_sim_start, theta_hat_sim_start = create_initial_states_parameters_and_estimates(
            controller_under_test.dynamics_model, self.start_x,
            n_sims_per_start=self.n_sims_per_start,
        )

        # Generate a random scenario for each rollout from the given scenarios
        random_scenarios = []
        for i in range(n_sims):
            random_scenario = {}
            for param_name in scenarios[0].keys():
                param_min = parameter_ranges[param_name][0]
                param_max = parameter_ranges[param_name][1]
                random_scenario[param_name] = random.uniform(param_min, param_max)
            random_scenarios.append(random_scenario)

        # Make sure everything's on the right device
        device = controller_under_test.dynamics_model.device
        if hasattr(controller_under_test, "device"):
            device = controller_under_test.device  # type: ignore
        x_current = x_sim_start.to(device)
        theta_current = theta_sim_start.to(device)
        theta_hat_current = theta_hat_sim_start.to(device)

        # Reset the controller if necessary
        if hasattr(controller_under_test, "reset_controller"):
            controller_under_test.reset_controller(x_current)  # type: ignore

        # See how long controller took
        controller_calls = 0
        controller_time = 0.0

        # Simulate!
        delta_t = controller_under_test.dynamics_model.dt
        num_timesteps = int(self.t_sim // delta_t)
        u_current = torch.zeros(x_sim_start.shape[0], n_controls, device=device)
        controller_update_freq = int(controller_under_test.controller_period / delta_t)
        prog_bar_range = tqdm.trange(
            0, num_timesteps, desc=self.name + ": aCLBF Simulation", leave=True
        )
        for tstep in prog_bar_range:
            # Get the control input at the current state if it's time
            if tstep % controller_update_freq == 0:
                start_time = time.time()
                u_current = controller_under_test.u(x_current, theta_hat_current)
                end_time = time.time()
                controller_calls += 1
                controller_time += end_time - start_time
                controller_time_current_tstep = end_time - start_time


            # Get the barrier function if applicable
            h: Optional[torch.Tensor] = None
            # if hasattr(controller_under_test, "h") and hasattr(
            #     controller_under_test.dynamics_model, "get_observations"
            # ):
            #     controller_under_test = cast(
            #         "NeuralObsBFController", controller_under_test
            #     )
            #     dynamics_model = cast(
            #         "ObservableSystem", controller_under_test.dynamics_model
            #     )
            #     obs = dynamics_model.get_observations(x_current)
            #     h = controller_under_test.h(x_current, obs)

            # Get the Lyapunov function if applicable
            V: Optional[torch.Tensor] = None
            if hasattr(controller_under_test, "V") and h is None:
                V = controller_under_test.V(x_current, theta_hat_current)  # type: ignore

            # Log the current state and control for each simulation
            for sim_index in range(n_sims):
                log_packet = {"t": tstep * delta_t, "Simulation": str(sim_index)}

                # Include the parameters
                param_string = ""
                for param_name, param_value in random_scenarios[sim_index].items():
                    param_value_string = "{:.3g}".format(param_value)
                    param_string += f"{param_name} = {param_value_string}, "
                    log_packet[param_name] = param_value
                log_packet["Parameters"] = param_string[:-2]

                # Pick out the states to log and save them
                for x_idx in range(len(self.plot_x_labels)):
                    plot_x_index = self.plot_x_indices[x_idx]
                    plot_x_label = self.plot_x_labels[x_idx]

                    x_value = x_current[sim_index, plot_x_index].cpu().numpy().item()
                    log_packet[plot_x_label] = x_value

                log_packet["state"] = x_current.clone()[sim_index, :].cpu().numpy()
                log_packet["theta_hat"] = theta_hat_current.clone()[sim_index, :].cpu().numpy()
                log_packet["theta"] = theta_current.clone()[sim_index, :].cpu().numpy()
                log_packet["u"] = u_current.clone()[sim_index, :].cpu().numpy()
                log_packet["controller_time"] = controller_time_current_tstep
                theta_error = theta_hat_current[sim_index, :] - theta_current[sim_index, :]
                log_packet["theta_error_norm"] = torch.norm(theta_error)

                # Log the barrier function if applicable
                if h is not None:
                    log_packet["h"] = h[sim_index].cpu().numpy().item()
                # Log the Lyapunov function if applicable
                if V is not None:
                    log_packet["V"] = V[sim_index].cpu().numpy().item()

                results.append(log_packet)

            # Simulate forward using the dynamics
            for i in range(n_sims):
                xdot = controller_under_test.dynamics_model.closed_loop_dynamics(
                    x_current[i, :].unsqueeze(0),
                    u_current[i, :].unsqueeze(0),
                    theta_current[i, :].unsqueeze(0),  # Theta should never change. Theta hat will.
                    random_scenarios[i],
                )
                x_current[i, :] = x_current[i, :] + delta_t * xdot.squeeze()

                theta_hat_dot = controller_under_test.closed_loop_estimator_dynamics(
                    x_current[i, :].unsqueeze(0),
                    theta_current[i, :].unsqueeze(0),  # Theta should never change. Theta hat will.
                    u_current[i, :].unsqueeze(0),
                    random_scenarios[i],
                )
                theta_hat_current[i, :] = theta_hat_current[i, :] + delta_t * theta_hat_dot.squeeze()

        return pd.DataFrame(results)

    @torch.no_grad()
    def run_nominal_controlled(
            self,
            dynamics: ControlAffineParameterAffineSystem,
            controller_period: float,
    ) -> pd.DataFrame:
        """
        df_nominal = self.run_nominal_controlled(dynamics)
        Description:
            Run the experiment which evaluates the "NOMINAL CONTROLLER" on a number of
            initial conditions.
            The experiment has freedom to call other functions of the controller as necessary (if these
            functions are not supported by all controllers, then experiments will be
            responsible for checking compatibility with the provided controller)

        args:
            dynamics: The dynamical system which contains the nominal controller needed to evaluate
             the experiment.
        returns:
            a pandas DataFrame containing the results of the experiment, in tidy data
            format (i.e. each row should correspond to a single observation from the
            experiment).
        """
        # Constants
        scenarios = [dynamics.nominal_scenario]
        results = []  # Set up a dataframe to store the results

        # Compute the number of simulations to run
        n_sims = self.n_sims_per_start * self.start_x.shape[0]

        # Determine the parameter range to sample from
        parameter_ranges = {}
        for param_name in scenarios[0].keys():
            param_max = max([s[param_name] for s in scenarios])
            param_min = min([s[param_name] for s in scenarios])
            parameter_ranges[param_name] = (param_min, param_max)

        # Generate a tensor of start states
        n_dims = dynamics.n_dims
        n_controls = dynamics.n_controls
        n_theta = dynamics.n_params
        Theta = dynamics.Theta

        x_sim_start, theta_sim_start, theta_hat_sim_start = create_initial_states_parameters_and_estimates(
            dynamics, self.start_x,
            n_sims_per_start=self.n_sims_per_start,
        )

        # Generate a random scenario for each rollout from the given scenarios
        random_scenarios = []
        for i in range(n_sims):
            random_scenario = {}
            for param_name in scenarios[0].keys():
                param_min = parameter_ranges[param_name][0]
                param_max = parameter_ranges[param_name][1]
                random_scenario[param_name] = random.uniform(param_min, param_max)
            random_scenarios.append(random_scenario)

        # Make sure everything's on the right device
        device = dynamics.device

        x_current = x_sim_start.to(device)
        theta_current = theta_sim_start.to(device)
        theta_hat_current = theta_hat_sim_start.to(device)

        # See how long controller took
        controller_calls = 0
        controller_time = 0.0

        # Simulate!
        delta_t = dynamics.dt
        num_timesteps = int(self.t_sim // delta_t)
        u_current = torch.zeros(x_sim_start.shape[0], n_controls, device=device)
        controller_update_freq = int(controller_period / delta_t)
        prog_bar_range = tqdm.trange(
            0, num_timesteps, desc=self.name + ": Nominal Simulation", leave=True
        )
        for tstep in prog_bar_range:
            # Get the control input at the current state if it's time
            if tstep % controller_update_freq == 0:
                start_time = time.time()
                u_current = dynamics.u_nominal(x_current, theta_hat_current)
                end_time = time.time()
                controller_calls += 1
                controller_time += end_time - start_time
                controller_time_current_tstep = end_time - start_time

            # Get the barrier function if applicable
            h: Optional[torch.Tensor] = None
            # if hasattr(controller_under_test, "h") and hasattr(
            #     controller_under_test.dynamics_model, "get_observations"
            # ):
            #     controller_under_test = cast(
            #         "NeuralObsBFController", controller_under_test
            #     )
            #     dynamics_model = cast(
            #         "ObservableSystem", controller_under_test.dynamics_model
            #     )
            #     obs = dynamics_model.get_observations(x_current)
            #     h = controller_under_test.h(x_current, obs)

            # Log the current state and control for each simulation
            for sim_index in range(n_sims):
                log_packet = {"t": tstep * delta_t, "Simulation": str(sim_index)}

                # Include the parameters
                param_string = ""
                for param_name, param_value in random_scenarios[sim_index].items():
                    param_value_string = "{:.3g}".format(param_value)
                    param_string += f"{param_name} = {param_value_string}, "
                    log_packet[param_name] = param_value
                log_packet["Parameters"] = param_string[:-2]

                # Pick out the states to log and save them
                for x_idx in range(len(self.plot_x_labels)):
                    plot_x_index = self.plot_x_indices[x_idx]
                    plot_x_label = self.plot_x_labels[x_idx]

                    x_value = x_current[sim_index, plot_x_index].cpu().numpy().item()
                    log_packet[plot_x_label] = x_value
                log_packet["state"] = x_current.clone()[sim_index, :].cpu().numpy()
                log_packet["theta_hat"] = theta_hat_current.clone()[sim_index, :].cpu().detach().numpy()
                log_packet["theta"] = theta_current.clone()[sim_index, :].cpu().detach().numpy()
                log_packet["u"] = u_current.clone()[sim_index, :].cpu().detach().numpy()
                log_packet["controller_time"] = controller_time_current_tstep
                theta_error = theta_hat_current[sim_index, :] - theta_current[sim_index, :]
                log_packet["theta_error_norm"] = torch.norm(theta_error)

                results.append(log_packet)

            # Simulate forward using the dynamics
            for i in range(n_sims):
                xdot = dynamics.closed_loop_dynamics(
                    x_current[i, :].unsqueeze(0),
                    u_current[i, :].unsqueeze(0),
                    theta_current[i, :].unsqueeze(0),  # Theta should never change. Theta hat will.
                    random_scenarios[i],
                )
                x_current[i, :] = x_current[i, :].clone() + delta_t * xdot.squeeze()

                # theta_hat_dot = controller_under_test.closed_loop_estimator_dynamics(
                #     x_current[i, :].unsqueeze(0),
                #     theta_current[i, :].unsqueeze(0),  # Theta should never change. Theta hat will.
                #     u_current[i, :].unsqueeze(0),
                #     random_scenarios[i],
                # )
                # Maintain constant belief about parameters
                theta_hat_current[i, :] = theta_hat_current[i, :] #+ delta_t * theta_hat_dot.squeeze()

        return pd.DataFrame(results)

    @torch.no_grad()
    def run_trajopt_controlled(
            self,
            dynamics: ControlAffineParameterAffineSystem,
            controller_period: float,
            control_sequence: torch.Tensor,
            theta_hat_sim_start_in: Optional[torch.Tensor] = None,
    ) -> pd.DataFrame:
        """
        df_trajopt = self.run_trajopt_controlled(dynamics, controller_period, control_sequence)
        Description:
            Run the experiment which evaluates the "NOMINAL CONTROLLER" on a number of
            initial conditions.
            The experiment has freedom to call other functions of the controller as necessary (if these
            functions are not supported by all controllers, then experiments will be
            responsible for checking compatibility with the provided controller)

        args:
            dynamics: The dynamical system which contains the nominal controller needed to evaluate
             the experiment.
            control_sequence: A n_x0s x timehorizon x n_controls tensor of control inputs to apply
        returns:
            a pandas DataFrame containing the results of the experiment, in tidy data
            format (i.e. each row should correspond to a single observation from the
            experiment).
        """
        # Constants
        scenarios = [dynamics.nominal_scenario]
        results = []  # Set up a dataframe to store the results
        horizon = control_sequence.shape[1]

        # Compute the number of simulations to run
        n_sims = self.n_sims_per_start * self.start_x.shape[0]

        # Determine the parameter range to sample from
        parameter_ranges = {}
        for param_name in scenarios[0].keys():
            param_max = max([s[param_name] for s in scenarios])
            param_min = min([s[param_name] for s in scenarios])
            parameter_ranges[param_name] = (param_min, param_max)

        # Generate a tensor of start states
        n_dims = dynamics.n_dims
        n_controls = dynamics.n_controls
        n_theta = dynamics.n_params
        Theta = dynamics.Theta

        x_sim_start, theta_sim_start, theta_hat_sim_start = create_initial_states_parameters_and_estimates(
            dynamics, self.start_x,
            n_sims_per_start=self.n_sims_per_start,
        )
        if theta_hat_sim_start_in is not None:  # Set value of theta_hat if provided
            theta_hat_sim_start = theta_hat_sim_start_in

        # Generate a random scenario for each rollout from the given scenarios
        random_scenarios = []
        for i in range(n_sims):
            random_scenario = {}
            for param_name in scenarios[0].keys():
                param_min = parameter_ranges[param_name][0]
                param_max = parameter_ranges[param_name][1]
                random_scenario[param_name] = random.uniform(param_min, param_max)
            random_scenarios.append(random_scenario)

        # Make sure everything's on the right device
        device = dynamics.device

        x_current = x_sim_start.to(device)
        theta_current = theta_sim_start.to(device)
        theta_hat_current = theta_hat_sim_start.to(device)

        # See how long controller took
        controller_calls = 0
        controller_time = 0.0

        # Simulate!
        delta_t = dynamics.dt
        num_timesteps = horizon #int(self.t_sim // delta_t)
        u_current = torch.zeros(x_sim_start.shape[0], n_controls, device=device)
        controller_update_freq = int(controller_period / delta_t)
        prog_bar_range = tqdm.trange(
            0, num_timesteps, desc=self.name + ": Trajopt Simulation", leave=True
        )
        for tstep in prog_bar_range:
            # Get the control input at the current state if it's time
            if tstep % controller_update_freq == 0:
                start_time = time.time()
                u_current = control_sequence[:, tstep, :].reshape(n_sims, n_controls)
                end_time = time.time()
                controller_calls += 1
                controller_time += end_time - start_time
                controller_time_current_tstep = end_time - start_time

            # Get the barrier function if applicable
            h: Optional[torch.Tensor] = None
            # if hasattr(controller_under_test, "h") and hasattr(
            #     controller_under_test.dynamics_model, "get_observations"
            # ):
            #     controller_under_test = cast(
            #         "NeuralObsBFController", controller_under_test
            #     )
            #     dynamics_model = cast(
            #         "ObservableSystem", controller_under_test.dynamics_model
            #     )
            #     obs = dynamics_model.get_observations(x_current)
            #     h = controller_under_test.h(x_current, obs)

            # Log the current state and control for each simulation
            for sim_index in range(n_sims):
                log_packet = {"t": tstep * delta_t, "Simulation": str(sim_index)}

                # Include the parameters
                param_string = ""
                for param_name, param_value in random_scenarios[sim_index].items():
                    param_value_string = "{:.3g}".format(param_value)
                    param_string += f"{param_name} = {param_value_string}, "
                    log_packet[param_name] = param_value
                log_packet["Parameters"] = param_string[:-2]

                # Pick out the states to log and save them
                for x_idx in range(len(self.plot_x_indices)):
                    plot_x_index = self.plot_x_indices[x_idx]
                    plot_x_label = self.plot_x_labels[x_idx]

                    x_value = x_current[sim_index, plot_x_index].cpu().numpy().item()
                    log_packet[plot_x_label] = x_value
                log_packet["state"] = x_current.clone()[sim_index, :].cpu().detach().numpy()
                log_packet["theta_hat"] = theta_hat_current.clone()[sim_index, :].cpu().detach().numpy()
                log_packet["theta"] = theta_current.clone()[sim_index, :].cpu().detach().numpy()
                log_packet["u"] = u_current.clone()[sim_index, :].cpu().detach().numpy()
                log_packet["controller_time"] = controller_time_current_tstep
                theta_error = theta_hat_current[sim_index, :] - theta_current[sim_index, :]
                log_packet["theta_error_norm"] = torch.norm(theta_error)

                results.append(log_packet)

            # Simulate forward using the dynamics
            for i in range(n_sims):
                xdot = dynamics.closed_loop_dynamics(
                    x_current[i, :].unsqueeze(0),
                    u_current[i, :].unsqueeze(0),
                    theta_current[i, :].unsqueeze(0),  # Theta should never change. Theta hat will.
                    random_scenarios[i],
                )
                x_current[i, :] = x_current[i, :] + delta_t * xdot.squeeze()

                # theta_hat_dot = controller_under_test.closed_loop_estimator_dynamics(
                #     x_current[i, :].unsqueeze(0),
                #     theta_current[i, :].unsqueeze(0),  # Theta should never change. Theta hat will.
                #     u_current[i, :].unsqueeze(0),
                #     random_scenarios[i],
                # )
                # Maintain constant belief about parameters
                theta_hat_current[i, :] = theta_hat_current[i, :]  # + delta_t * theta_hat_dot.squeeze()

        return pd.DataFrame(results)

    @torch.no_grad()
    def run_trajopt_with_synthesis(
            self,
            dynamics: ControlAffineParameterAffineSystem,
            controller_period: float,
            dynamics_update: Callable[[float, np.array, np.array, Dict], np.array],
            Tf: float = None,
            u0: np.array = None,
            uf: np.array = None,
            constraints: List[Any] = [],
            Q: np.diag = None,
            R: np.array = None,
            P: np.array = None,
            N_timepts: int = 100,
    ):
        """
        df_trajopt, trajopt_synth_time = self.run_trajopt_with_synthesis(
            dynamics, controller_dt, np_dynamics_update,
            x0, Tf,
        )
        Description:
            This function runs a trajectory optimization experiment that attempts to avoid the
            obstacle centered at obstacle_pose with radius obstacle_radius while reaching the goal
            state defined by the dynamics.
        Inputs:
            dynamics: The dynamics model to use for the experiment
            controller_period: The period of the controller to use for the experiment
            np_dynamics_update: The dynamics model to use for the trajectory optimization
                Example - def vehicle_update(t, x, u, params) -> xdot:
                This system will be
            x0: An (n_dims,) array of the initial state
            theta_hat: An (n_params,) array of the initial guess for the parameters
            constraints: A list of constraints to apply to the trajectory optimization
                         Note: do not include the input constraint! Will be added automatically.
            Q: The state cost matrix

        Returns:

        """
        # Input Processing
        if Q is None:
            Q = np.diag([1.0 for i in range(dynamics.n_dims)])
        if R is None:
            R = np.diag([1.0 for i in range(dynamics.n_controls)])
        if P is None:
            P = np.diag([10000.0 for i in range(dynamics.n_dims)])  # get close to final point
        if u0 is None:
            u0 = np.zeros((dynamics.n_controls,))
            u0[0] = 10.0
        if uf is None:
            uf = np.zeros((dynamics.n_controls,))
        if Tf is None:
            Tf = self.t_sim

        # Constants
        start_x = self.start_x

        # Setup trajectory optimization
        #==============================

        # Setup the output equation
        def output_equation(t, x, u, params):
            return x

        # Create optimized trajectory for each ic
        n_x0s = start_x.shape[0]
        theta_samples = dynamics.sample_Theta_space(n_x0s).numpy()
        traj_opt_times = []
        control_sequences = torch.zeros((n_x0s, N_timepts, dynamics.n_controls))
        for x0_index in range(n_x0s):
            # Get x0 and xf
            x0 = start_x[x0_index, :].numpy()
            theta_sample = theta_samples[x0_index, :].reshape((dynamics.n_params,))
            xf = dynamics.goal_point(
                torch.tensor(theta_sample).reshape(1, dynamics.n_params),
            ).reshape((dynamics.n_dims,))
            xf = xf.numpy()

            # Define system
            output_list = tuple([f'x_{i}' for i in range(dynamics.n_dims)])
            input_list = tuple([f'u_{i}' for i in range(dynamics.n_controls)])
            system = ct.NonlinearIOSystem(
                dynamics_update, output_equation,
                states=dynamics.n_dims, name='case-study-system',
                inputs=input_list, outputs=output_list,
                params={"theta": theta_sample},
            )

            # Setup the initial and final conditions

            # Setup the cost function
            traj_cost = opt.quadratic_cost(system, Q, R, x0=xf, u0=uf)
            term_cost = opt.quadratic_cost(system, P, 0, x0=xf)

            # Add constraint
            constraint_set_i = constraints.copy()
            constraint_set_i.append(
                opt.input_poly_constraint(system, dynamics.U.A, dynamics.U.b)
            )

            # Setup the trajectory optimization problem
            timepts = np.linspace(0.0, Tf, N_timepts, endpoint=True)
            traj_opt_start = time.time()
            result = opt.solve_ocp(
                system, timepts, x0,
                traj_cost, constraint_set_i,
                terminal_cost=term_cost, initial_guess=u0,
            )
            traj_opt_end = time.time()
            traj_opt_times.append(
                traj_opt_end - traj_opt_start,
            )

            # Simulate the system dynamics (open loop)
            resp = ct.input_output_response(
                system, timepts, result.inputs, x0,
                t_eval=timepts)
            t, y, u = resp.time, resp.outputs, resp.inputs

            print("y = ", y)
            # Compile all input trajectories
            control_sequences[x0_index, :, :] = torch.tensor(u.T)

        # Simulate the system with optimized trajectory
        print(u)
        df_trajopt = self.run_trajopt_controlled(
            dynamics,
            controller_period,
            control_sequences,
            theta_hat_sim_start_in=torch.tensor(theta_samples),
        )

        return df_trajopt, traj_opt_times


    @torch.no_grad()
    def run_mpc_controlled(
            self,
            dynamics: ControlAffineParameterAffineSystem,
            controller_period: float,
            state_sequence: torch.Tensor,
            control_sequence: torch.Tensor,
            mpc_horizon: int=5,
    ) -> pd.DataFrame:
        """
        df_nominal = self.run_mpc_controlled(dynamics)
        Description:
            Run the experiment which evaluates the "MPC CONTROLLER" on a number of
            initial conditions.
            The experiment has freedom to call other functions of the controller as necessary (if these
            functions are not supported by all controllers, then experiments will be
            responsible for checking compatibility with the provided controller)

        args:
            dynamics: The dynamical system which contains the nominal controller needed to evaluate
             the experiment.
            control_sequence: A n_x0s x timehorizon x n_controls tensor of control inputs to apply
        returns:
            a pandas DataFrame containing the results of the experiment, in tidy data
            format (i.e. each row should correspond to a single observation from the
            experiment).
        """
        # Constants
        scenarios = [dynamics.nominal_scenario]
        results = []  # Set up a dataframe to store the results
        horizon = control_sequence.shape[1]

        # Compute the number of simulations to run
        n_sims = self.n_sims_per_start * self.start_x.shape[0]

        # Determine the parameter range to sample from
        parameter_ranges = {}
        for param_name in scenarios[0].keys():
            param_max = max([s[param_name] for s in scenarios])
            param_min = min([s[param_name] for s in scenarios])
            parameter_ranges[param_name] = (param_min, param_max)

        # Generate a tensor of start states
        n_dims = dynamics.n_dims
        n_controls = dynamics.n_controls
        n_theta = dynamics.n_params
        Theta = dynamics.Theta

        x_sim_start, theta_sim_start, theta_hat_sim_start = create_initial_states_parameters_and_estimates(
            dynamics, self.start_x,
            n_sims_per_start=self.n_sims_per_start,
        )

        # Generate a random scenario for each rollout from the given scenarios
        random_scenarios = []
        for i in range(n_sims):
            random_scenario = {}
            for param_name in scenarios[0].keys():
                param_min = parameter_ranges[param_name][0]
                param_max = parameter_ranges[param_name][1]
                random_scenario[param_name] = random.uniform(param_min, param_max)
            random_scenarios.append(random_scenario)

        # Make sure everything's on the right device
        device = dynamics.device

        x_current = x_sim_start.to(device)
        theta_current = theta_sim_start.to(device)
        theta_hat_current = theta_hat_sim_start.to(device)

        # See how long controller took
        controller_calls = 0
        controller_time = 0.0

        # Simulate!
        delta_t = dynamics.dt
        num_timesteps = horizon - mpc_horizon  # int(self.t_sim // delta_t)
        u_current = torch.zeros(x_sim_start.shape[0], n_controls, device=device)
        controller_update_freq = int(controller_period / delta_t)
        prog_bar_range = tqdm.trange(
            0, num_timesteps, desc=self.name + ": MPC Simulation", leave=True
        )
        for tstep in prog_bar_range:
            # Get the control input at the current state if it's time
            if tstep % controller_update_freq == 0:
                start_time = time.time()
                u_current = dynamics.mpc_about_input_trajectory(
                    x_current,
                    theta_hat_current,
                    state_sequence,
                    horizon=mpc_horizon,
                )
                end_time = time.time()
                controller_calls += 1
                controller_time += end_time - start_time
                controller_time_current_tstep = end_time - start_time

            # Get the barrier function if applicable
            h: Optional[torch.Tensor] = None
            # if hasattr(controller_under_test, "h") and hasattr(
            #     controller_under_test.dynamics_model, "get_observations"
            # ):
            #     controller_under_test = cast(
            #         "NeuralObsBFController", controller_under_test
            #     )
            #     dynamics_model = cast(
            #         "ObservableSystem", controller_under_test.dynamics_model
            #     )
            #     obs = dynamics_model.get_observations(x_current)
            #     h = controller_under_test.h(x_current, obs)

            # Log the current state and control for each simulation
            for sim_index in range(n_sims):
                log_packet = {"t": tstep * delta_t, "Simulation": str(sim_index)}

                # Include the parameters
                param_string = ""
                for param_name, param_value in random_scenarios[sim_index].items():
                    param_value_string = "{:.3g}".format(param_value)
                    param_string += f"{param_name} = {param_value_string}, "
                    log_packet[param_name] = param_value
                log_packet["Parameters"] = param_string[:-2]

                # Pick out the states to log and save them
                # for x_idx in range(len(self.plot_x_indices)):
                #     plot_x_index = self.plot_x_indices[x_idx]
                #     plot_x_label = self.plot_x_labels[x_idx]
                #
                #     x_value = x_current[sim_index, plot_x_index].cpu().numpy().item()
                #     log_packet[plot_x_label] = x_value
                log_packet["state"] = x_current[sim_index, :].cpu().detach().numpy()
                log_packet["theta_hat"] = theta_hat_current[sim_index, :].cpu().detach().numpy()
                log_packet["theta"] = theta_current[sim_index, :].cpu().detach().numpy()
                log_packet["u"] = u_current[sim_index, :].cpu().detach().numpy()
                log_packet["controller_time"] = controller_time_current_tstep
                theta_error = theta_hat_current[sim_index, :] - theta_current[sim_index, :]
                log_packet["theta_error_norm"] = torch.norm(theta_error)

                results.append(log_packet)

            # Simulate forward using the dynamics
            for i in range(n_sims):
                xdot = dynamics.closed_loop_dynamics(
                    x_current[i, :].unsqueeze(0),
                    u_current[i, :].unsqueeze(0),
                    theta_current[i, :].unsqueeze(0),  # Theta should never change. Theta hat will.
                    random_scenarios[i],
                )
                x_current[i, :] = x_current[i, :] + delta_t * xdot.squeeze()

                # theta_hat_dot = controller_under_test.closed_loop_estimator_dynamics(
                #     x_current[i, :].unsqueeze(0),
                #     theta_current[i, :].unsqueeze(0),  # Theta should never change. Theta hat will.
                #     u_current[i, :].unsqueeze(0),
                #     random_scenarios[i],
                # )
                # Maintain constant belief about parameters
                theta_hat_current[i, :] = theta_hat_current[i, :]  # + delta_t * theta_hat_dot.squeeze()

        return pd.DataFrame(results)

    def plot(
        self,
        controller_under_test: "Controller",
        aclbf_results_df: pd.DataFrame = None,
        nominal_results_df: pd.DataFrame = None,
        trajopt2_results_df: pd.DataFrame = None,
        display_plots: bool = False,
    ) -> List[Tuple[str, figure]]:
        """
        Plot the results, and return the plot handles. Optionally
        display the plots.

        args:
            controller_under_test: the controller with which to run the experiment
            results_df: dataframe containing the results of previous experiments.
            display_plots: defaults to False. If True, display the plots (blocks until
                           the user responds).
        returns: a list of tuples containing the name of each figure and the figure
                 object.
        """

        # raise NotImplementedError(f"This method will eventually show bar graphs (or something else) reflecting data.")

        fig_handles = []

        # Set the color scheme
        sns.set_theme(context="talk", style="white")

        # Plot ACLBF results
        if aclbf_results_df is not None:
            fig_handle = self.plot_trajectory(
                aclbf_results_df,
                controller_under_test.dynamics_model,
                fig_name="Rollout (aclbf)",
            )

            if not display_plots:
                fig_handles.append(fig_handle)

        if nominal_results_df is not None:
            fig_handle2 = self.plot_trajectory(
                nominal_results_df,
                controller_under_test.dynamics_model,
                fig_name="Rollout (nominal)",
            )

            if not display_plots:
                fig_handles.append(fig_handle2)

        if trajopt2_results_df is not None:
            fig_handle3 = self.plot_trajectory(
                trajopt2_results_df,
                controller_under_test.dynamics_model,
                fig_name="Rollout (trajopt2)",
            )

            if not display_plots:
                fig_handles.append(fig_handle3)

        if display_plots:
            plt.show()
            return fig_handles
        else:
            return fig_handles

    def plot_trajectory(
            self,
            results_df: pd.DataFrame,
            dynamical_system: ControlAffineParameterAffineSystem,
            fig_name: str = "Rollout",
    ) -> Tuple[str, plt.Figure]:
        """
        fig_handle_out = self.plot_trajectory(results_df, ax)
        Description:
            This function is to plot the trajectory of the system in the state space
            from the given results dataframe.
        """
        assert len(self.plot_x_labels) == 3, f"This function requires for 3 values to be given in x_labels to make a 3D plots; Received {len(self.plot_x_labels)}"

        # Constants
        goal_tolerance = dynamical_system.goal_tolerance

        fig = plt.figure()

        num_plots = 2
        if "h" in results_df:
            num_plots += 1
        if "V" in results_df:
            num_plots += 1

        # Create figure and begin plotting
        fig.set_size_inches(9 * num_plots, 6)

        rollout_ax = fig.add_subplot(100+10*num_plots+1, projection="3d")
        plot_rollouts(results_df, rollout_ax, dynamical_system, self.plot_x_labels)


        # Plot the error
        error_ax = fig.add_subplot(100+10*num_plots+2)
        plot_error_to_goal(results_df, error_ax, dynamical_system)

        # Remove the legend -- too much clutter
        rollout_ax.legend([], [], frameon=False)


        # Plot the lyapunov function if applicable
        if "V" in results_df:
            # Create axis for lyapunov
            V_ax = fig.add_subplot(
                100+num_plots*10+num_plots,
            )

            for plot_idx, sim_index in enumerate(results_df["Simulation"].unique()):
                sim_mask = results_df["Simulation"] == sim_index
                V_ax.plot(
                    results_df[sim_mask]["t"].to_numpy(),
                    results_df[sim_mask]["V"].to_numpy(),
                    linestyle="-",
                    # marker="+",
                    markersize=5,
                    color=sns.color_palette()[plot_idx],
                )
            # sns.lineplot(
            #     ax=V_ax,
            #     x="t",
            #     y="V",
            #     style="Parameters",
            #     hue="Simulation",
            #     data=results_df,
            # )
            V_ax.set_ylabel("$V$")
            V_ax.set_xlabel("t")
            # Remove the legend -- too much clutter
            V_ax.legend([], [], frameon=False)

            # Plot a reference line at V = 0
            V_ax.plot([0, results_df.t.max()], [0, 0], color="k")

        # Plot the environment
        # controller_under_test.dynamics_model.plot_environment(rollout_ax)

        return (fig_name, fig)


    def plot_error(self, results_df: pd.DataFrame, error_ax: plt.axis):
        """

        """
        # Constants

        for plot_idx, sim_index in enumerate(results_df["Simulation"].unique()):
            sim_mask = results_df["Simulation"] == sim_index
            # rollout_ax.plot(
            #     results_df[sim_mask][self.plot_x_labels[0]].to_numpy(),
            #     results_df[sim_mask][self.plot_x_labels[1]].to_numpy(),
            #     zs = results_df[sim_mask][self.plot_x_labels[2]].to_numpy(),
            #     linestyle="-",
            #     # marker="+",
            #     markersize=5,
            #     color=sns.color_palette()[plot_idx],
            # )
            error_ax.plot(
                results_df[sim_mask]["t"].to_numpy(),
                results_df[sim_mask]["theta_error_norm"].to_numpy(),
                linestyle="-",
                # marker="+",
                markersize=5,
                color=sns.color_palette()[plot_idx],
            )
            error_ax.set_xlabel("$t$")
            error_ax.set_ylabel("$\| r(t) - r_{des}(t) \|$")

    def save_timing_data_table(
        self,
        table_name: str,
        commit_prefix: str,
        version_number: str,
        aclbf_results_df: pd.DataFrame = None,
        nominal_results_df: pd.DataFrame = None,
        trajopt2_results_df: pd.DataFrame = None,
        trajopt2_synthesis_times: List[float] = None,
    ):
        """
        save_timing_data_table
        Description:
            Saves a table of timing data to a txt file that can
            be copied into a latex table.
        """

        # Constants

        # Collect the data
        aclbf_data_dict = None
        if aclbf_results_df is not None:
            aclbf_data_dict = get_avg_computation_time_from_df(aclbf_results_df)

        nominal_data_dict = None
        if nominal_results_df is not None:
            nominal_data_dict = get_avg_computation_time_from_df(nominal_results_df)

        trajopt2_data_dict = None
        if trajopt2_results_df is not None:
            trajopt2_data_dict = get_avg_computation_time_from_df(trajopt2_results_df)

        # Save the data to txt file
        with open(table_name, "w") as f:
            comments = [f"n_sims_per_start={self.n_sims_per_start}"]
            comments += [f"n_x0={self.start_x.shape[0]}"]
            comments += [f"commit_prefix={commit_prefix}"]
            comments += [f"version_number={version_number}"]

            lines = timing_data_to_latex_table(
                aclbf_data_dict,
                nominal_timing=nominal_data_dict,
                trajopt2_timing=trajopt2_data_dict,
                trajopt2_synthesis_times=trajopt2_synthesis_times,
                comments=comments,
            )

            f.writelines(lines)