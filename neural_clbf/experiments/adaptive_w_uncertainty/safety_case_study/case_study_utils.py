"""
case_study_utils.py
Description:
    This file contains utility functions for the case study evaluation.
"""

from typing import cast, List, Tuple, Optional, TYPE_CHECKING, Dict, Callable, Any
import pandas as pd
import numpy as np
import torch
import seaborn as sns

import matplotlib.pyplot as plt

from neural_clbf.systems.adaptive import (
    ControlAffineParameterAffineSystem,
)

def get_avg_computation_time_from_df(
    results_df: pd.DataFrame,
):
    """
    dict_out = cs.get_avg_computation_time_from_df(results_df)
    """

    # Set up the dictionary
    data_dict = {}

    time_per_sim = []
    for plot_idx, sim_index in enumerate(results_df["Simulation"].unique()):
        sim_mask = results_df["Simulation"] == sim_index
        time_per_sim.append(
            (results_df[sim_mask]["controller_time"].to_numpy()).mean(),
        )

    data_dict["AveragePerSim"] = time_per_sim
    data_dict["OverallAverage"] = np.array(time_per_sim).mean()

    return data_dict

def timing_data_to_latex_table(
        aclbf_timing: Dict[str, int],
        nominal_timing: Dict[str, int] = None,
        trajopt_timing: Dict[str, int] = None,
        trajopt2_timing: Dict[str, int] = None,
        mpc_timing: Dict[str, int] = None,
        trajopt2_synthesis_times: Dict[str, int] = None,
        mpc_trajopt_synthesis_times: List[float] = None,
        comments: List[str] = None,
) -> str:
    """
    timing_data_to_latex_table(aclbf_timing, nominal_timing, trajopt2_timing=trajopt2_timing, comments=comments)
    Description:
        Format the counts of the number of times each state was reached
        into a string.

    args:
        aclbf_timing: Dict[str, int], the timing data for the aclbf controller.
            Example {'OverallAverage': 0.1}
        nominal_timing: Dict[str, int], the timing data for the nominal controller.
            Example {'OverallAverage': 0.1}
        trajopt_timing: Dict[str, int], the timing data for the trajopt controller.
            Example {'OverallAverage': 0.1, 'ComputeAverage': 0.1}
        trajopt2_timing: Dict[str, int], the timing data for the trajopt2 controller.
            Example {'OverallAverage': 0.1, 'ComputeAverage': 0.1}
    returns:
        lines_of_table: List[str], a string containing the counts.
    """
    # Constants
    lines_of_table = []

    # Start formatting the table
    lines_of_table += [r"\begin{center}" + f"\n"]
    lines_of_table += [f"\t" + r"\begin{tabular}{|c|c|c|}" + f"\n"]
    lines_of_table += [f"\t\t" + r"\hline" + f"\n"]
    lines_of_table += [f"\t\t" + r"Controller & t_{init} (ms) & $t_{comp}$ (ms) \\" + f"\n"]
    lines_of_table += [f"\t\t" + r"\hline" + f"\n"]

    # Format the counts

    # Add goal reached counts for:
    # - Nominal
    if nominal_timing is not None:
        nominal_overall_compute_avg_timing = "{:.2f}".format(nominal_timing['OverallAverage'] * 1000)
        lines_of_table += [f"\t\t" + f"Nominal & -- & {nominal_overall_compute_avg_timing} \\\\ \n"]
        lines_of_table += [f"\t\t" + r"\hline" + f"\n"]

    # - Trajopt
    if trajopt_timing is not None:
        raise NotImplementedError

    # - Trajopt2
    if trajopt2_timing is not None:
        trajopt2_overall_compute_avg_timing = "{:.2f}".format(trajopt2_timing['OverallAverage'] * 1000)
        trajopt_avg_synthesis_time = "{:.2f}".format(np.mean(np.array(trajopt2_synthesis_times)) * 1000)
        lines_of_table += [
            f"\t\t" + f"TrajOpt & {trajopt_avg_synthesis_time} & {trajopt2_overall_compute_avg_timing} \\\\ \n"]
        lines_of_table += [f"\t\t" + r"\hline" + f"\n"]

    # - (Hybrid) MPC about optimized trajectory
    if mpc_timing is not None:
        mpc_overall_compute_avg_timing = "{:.2f}".format(mpc_timing['OverallAverage'] * 1000)
        mpc_avg_traj_synthesis_time = "{:.2f}".format(np.mean(np.array(mpc_trajopt_synthesis_times)) * 1000)
        lines_of_table += [
            f"\t\t" + f"MPC & {mpc_avg_traj_synthesis_time} & {mpc_overall_compute_avg_timing} \\\\ \n"]
        lines_of_table += [f"\t\t" + r"\hline" + f"\n"]

    # # - (Hybrid) MPC about optimized trajectory
    # if mpc_counts is not None:
    #     mpc_gr_percentage = "{:.2f}".format(mpc_counts['goal_reached_percentage'])
    #     mpc_s_percentage = "{:.2f}".format(1-mpc_counts['unsafe_percentage'])
    #     lines_of_table += [
    #         f"\t\t" + f"MPC & {mpc_gr_percentage} & {mpc_s_percentage} \\\\ \n"
    #     ]
    #     lines_of_table += [f"\t\t" + r"\hline" + f"\n"]

    # - aCLBF
    if aclbf_timing is not None:
        aclbf_overall_compute_avg_timing = "{:.2f}".format(aclbf_timing['OverallAverage'] * 1000)
        lines_of_table += [f"\t\t" + f"Neural aCLBF & -- & {aclbf_overall_compute_avg_timing} \\\\ \n"]
        lines_of_table += [f"\t\t" + r"\hline" + f"\n"]

    # End formatting the table
    # lines_of_table += [f"\t\t" + r"\hline" + f"\n"]
    lines_of_table += [f"\t" + r"\end{tabular}" + f"\n"]
    lines_of_table += [r"\end{center}" + f"\n"]

    if comments != None:
        lines_of_table += [f" \n"]
        for comment in comments:
            lines_of_table += [f"% " + comment + f" \n"]

    return lines_of_table

def tabulate_number_of_reaches(
    results_df: pd.DataFrame,
    dynamics: ControlAffineParameterAffineSystem,
) -> Dict[str, float]:
    """
    tabulated_results = self.tabulate_number_of_reaches(results_df)
    Description:
        Tabulate whether or not a given system reached
        the goal region during the experiment.

    args:
        results_df: dataframe containing the results of previous experiments.
    returns: a dataframe containing the number of times each state was reached.
    """
    # Constants
    num_timesteps = len(
        results_df[
            results_df["Simulation"] == "0"
        ])

    # Compute the number of simulations to run
    n_sims = len(results_df["Simulation"].unique())

    # Create new results struct
    counts = {
        "goal_reached": 0, "goal_reached_percentage": 0.0,
        "unsafe": 0, "unsafe_percentage": 0.0,
    }

    # For each simulation, count the number
    # of times the system reached the goal region
    for plot_idx, sim_index in enumerate(results_df["Simulation"].unique()):
        sim_mask = results_df["Simulation"] == sim_index
        sim_results_x = np.vstack(
            np.array(results_df[sim_mask]["state"])
        )
        sim_results_x = torch.tensor(sim_results_x)

        sim_results_theta = np.vstack(
            np.array(results_df[sim_mask]["theta"])
        )
        sim_results_theta = torch.tensor(sim_results_theta)

        # Count the number of times the goal was reached
        goal_reached = dynamics.goal_mask(
            sim_results_x,
            sim_results_theta,
        )

        # Count the number of times that the system was unsafe
        unsafe = dynamics.unsafe_mask(
            sim_results_x,
            sim_results_theta,
        )

        if goal_reached.any():
            counts["goal_reached"] += 1

        if unsafe.any():
            counts["unsafe"] += 1

    # Compute the percentage of times:
    # - the goal was reached
    # - the system was unsafe
    counts["goal_reached_percentage"] = counts["goal_reached"] / n_sims
    counts["unsafe_percentage"] = counts["unsafe"] / n_sims

    # Convert to a dataframe
    return counts



def counts_to_latex_table(
        aclbf_counts: Dict[str, int] = None,
        nominal_counts: Dict[str, int] = None,
        trajopt_counts: Dict[str, int] = None,
        trajopt2_counts: Dict[str, int] = None,
        mpc_counts: Dict[str, int] = None,
        comments: List[str] = None,
) -> str:
    """
    Description:
        Format the counts of the number of times each state was reached
        into a string.

    args:
        counts: a dictionary containing the number of times each state was reached.
    returns: a string containing the counts.
    """
    # Constants
    lines_of_table = []

    # Start formatting the table
    lines_of_table += [r"\begin{center}" + f"\n"]
    lines_of_table += [f"\t" + r"\begin{tabular}{|c|c|c|}" + f"\n"]
    lines_of_table += [f"\t\t" + r"\hline" + f"\n"]
    lines_of_table += [f"\t\t" + r"Controller & Goal Reached & Safe \\" + f"\n"]


    # Format the counts

    # Add goal reached counts for:
    # - Nominal
    if nominal_counts is not None:
        nominal_gr_percentage = "{:.2f}".format(nominal_counts['goal_reached_percentage'])
        nominal_s_percentage = "{:.2f}".format(1-nominal_counts['unsafe_percentage'])
        lines_of_table += [f"\t\t" + r"\hline" + f"\n"]
        lines_of_table += [
            f"\t\t" + f"Nominal & {nominal_gr_percentage} & {nominal_s_percentage} \\\\ \n"
        ]


    # - Trajopt
    if trajopt_counts is not None:
        trajopt_gr_percentage = "{:.2f}".format(trajopt_counts['goal_reached_percentage'])
        trajopt_s_percentage = "{:.2f}".format(1 - trajopt_counts['unsafe_percentage'])
        lines_of_table += [f"\t\t" + r"\hline" + f"\n"]
        lines_of_table += [
            f"\t\t" + f"Trajopt (Trajax) & {trajopt_gr_percentage} & {trajopt_s_percentage} \\\\ \n"
        ]

    # - Trajopt2
    if trajopt2_counts is not None:
        trajopt2_gr_percentage = "{:.2f}".format(trajopt2_counts['goal_reached_percentage'])
        trajopt2_s_percentage = "{:.2f}".format(1 - trajopt2_counts['unsafe_percentage'])
        lines_of_table += [f"\t\t" + r"\hline" + f"\n"]
        lines_of_table += [
            f"\t\t" + f"Trajopt2 (control) & {trajopt2_gr_percentage} & {trajopt2_s_percentage} \\\\ \n"
        ]

    # - (Hybrid) MPC about optimized trajectory
    if mpc_counts is not None:
        mpc_gr_percentage = "{:.2f}".format(mpc_counts['goal_reached_percentage'])
        mpc_s_percentage = "{:.2f}".format(1-mpc_counts['unsafe_percentage'])
        lines_of_table += [f"\t\t" + r"\hline" + f"\n"]
        lines_of_table += [
            f"\t\t" + f"MPC & {mpc_gr_percentage} & {mpc_s_percentage} \\\\ \n"
        ]

    # - aCLBF
    if aclbf_counts is not None:
        aclbf_gr_percentage = "{:.2f}".format(aclbf_counts['goal_reached_percentage'])
        aclbf_s_percentage = "{:.2f}".format(1-aclbf_counts['unsafe_percentage'])
        lines_of_table += [f"\t\t" + r"\hline" + f"\n"]
        lines_of_table += [f"\t\t" + f"Neural aCLBF & {aclbf_gr_percentage} & {aclbf_s_percentage} \\\\ \n"]

    # End formatting the table
    lines_of_table += [f"\t\t" + r"\hline" + f"\n"]
    lines_of_table += [f"\t" + r"\end{tabular}" + f"\n"]
    lines_of_table += [r"\end{center}" + f"\n"]

    if comments != None:
        lines_of_table += [f" \n"]
        for comment in comments:
            lines_of_table += [f"% " + comment + f" \n"]

    return lines_of_table

def plot_rollouts(
    results_df: pd.DataFrame,
    rollout_ax: plt.Axes,
    dynamical_system: ControlAffineParameterAffineSystem,
    plot_x_labels: List[str],
    plot_x_indices: List[int] = [0, 1, 2],
    x_limits: List[float] = None,
    y_limits: List[float] = None,
    show_environment: bool = True,
):
    """
    plot_rollouts
    Description:
        Plots the rollouts of the case study
    """

    assert len(plot_x_labels) == 3, "Only 3D plots are supported for now."

    # Constants

    # Input Processing
    if x_limits is None:
        # TODO: Replace with better logic for finding limits
        x_limits = [-0.3, 0.3]
        rollout_ax.set_xlim(x_limits)

    if y_limits is None:
        y_limits = [-0.3, 0.3]
        rollout_ax.set_ylim(y_limits)

    # Iterate through each simulation and plot them.

    for plot_idx, sim_index in enumerate(results_df["Simulation"].unique()):
        sim_mask = results_df["Simulation"] == sim_index
        rollout_ax.plot(
            results_df[sim_mask][plot_x_labels[0]].to_numpy(),
            results_df[sim_mask][plot_x_labels[1]].to_numpy(),
            results_df[sim_mask][plot_x_labels[2]].to_numpy(),
            linestyle="-",
            # marker="+",
            markersize=5,
            color=sns.color_palette()[plot_idx % 10],
        )
        rollout_ax.set_xlabel(plot_x_labels[0])
        rollout_ax.set_ylabel(plot_x_labels[1])
        rollout_ax.set_zlabel(plot_x_labels[2])

        # Plot initial conditions
        rollout_ax.scatter(
            results_df[sim_mask][plot_x_labels[0]].to_numpy()[0],
            results_df[sim_mask][plot_x_labels[1]].to_numpy()[0],
            results_df[sim_mask][plot_x_labels[2]].to_numpy()[0],
        )

        # Plot Target Points
        theta = np.stack(results_df[sim_mask]["theta"].to_numpy())
        # print(sim_index)
        # print("theta.shape = ", theta.shape)
        goal_point = dynamical_system.goal_point(
            torch.tensor(theta[0, :]).reshape((1, dynamical_system.n_params))
        )

        rollout_ax.scatter(
            goal_point[0, plot_x_indices[0]],
            goal_point[0, plot_x_indices[1]],
            goal_point[0, plot_x_indices[2]],
            marker="s",
            s=20,
            color=sns.color_palette()[plot_idx % 10],
        )

    # Remove the legend -- too much clutter
    rollout_ax.legend([], [], frameon=False)

    # Show environment
    if show_environment:
        dynamical_system.plot_environment(
            rollout_ax, theta[0, :].reshape((1, dynamical_system.n_params)),
        )

def plot_error_to_goal(
    results_df: pd.DataFrame,
    error_ax: plt.Axes,
    dynamical_system: ControlAffineParameterAffineSystem,
):
    """
    plot_error_to_goal
    Description:
        Plots the error trajectories of a dataframe's simulations
    """

    # Constants
    goal_tolerance = dynamical_system.goal_tolerance

    # Input Processing

    # Iterate through each simulation and plot them.
    for plot_idx, sim_index in enumerate(results_df["Simulation"].unique()):
        sim_mask = results_df["Simulation"] == sim_index

        x_sim_i = torch.tensor(np.vstack(results_df[sim_mask]["state"]).T)
        theta_sim_i = torch.tensor(np.vstack(results_df[sim_mask]["theta"]).T)
        goal_sim_i = dynamical_system.goal_point(theta_sim_i.T).T

        error_sim_i = x_sim_i - goal_sim_i
        error_ax.plot(
            results_df[sim_mask]["t"].to_numpy(),
            np.linalg.norm(error_sim_i, axis=0),
            linestyle="-",
            # marker="+",
            markersize=5,
            color=sns.color_palette()[plot_idx % 10],
        )
        error_ax.set_xlabel("$t$")
        error_ax.set_ylabel("$e(t) = ||x(t) - x_g(\\theta)||$")

        # Plot desired error level
        error_ax.plot(
            results_df[sim_mask]["t"].to_numpy(),
            np.ones((error_sim_i.shape[1],)) * goal_tolerance,
            linestyle=":",
            # marker="+",
            markersize=5,
            color=sns.color_palette()[plot_idx % 10],
        )

    # Remove the legend -- too much clutter
    error_ax.legend([], [], frameon=False)

def create_initial_states_parameters_and_estimates(
    dynamics: ControlAffineParameterAffineSystem,
    start_x: torch.Tensor,
    n_sims_per_start: int = 1,
    start_theta_hat: torch.tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    x0, theta0, theta_hat0 = self.create_initial_states_parameters_and_estimates()
    Description:
        Create initial states, parameters, and estimates for the experiment
    Returns:
        x0: bs x self.dynamical_systems.n_dims tensor of initial states
        theta0: bs x self..dynamical_systems.n_params tensor of initial parameter
        theta_hat0: Initial estimate
    """
    # Constants
    n_dims = dynamics.n_dims
    n_controls = dynamics.n_controls
    n_theta = dynamics.n_params
    Theta = dynamics.Theta

    n_ics = start_x.shape[0]

    # Compute the number of simulations to run
    n_sims = n_sims_per_start * n_ics

    # Create ics
    x_sim_start = torch.zeros(n_sims, n_dims).type_as(start_x)
    for i in range(n_ics):
        for j in range(0, n_sims_per_start):
            x_sim_start[i * n_sims_per_start + j, :] = start_x[i, :]

    theta_sim_start = torch.zeros(n_sims, n_theta).type_as(start_x)
    theta_sim_start[:, :] = torch.Tensor(
        dynamics.sample_Theta_space(n_sims)
    )

    theta_hat_sim_start = torch.zeros(n_sims, n_theta).type_as(start_x)
    theta_hat_sim_start[:, :] = torch.Tensor(
        dynamics.sample_Theta_space(n_sims)
    )

    return x_sim_start, theta_sim_start, theta_hat_sim_start


def save_timing_data_table(
        table_name: str,
        commit_prefix: str,
        version_number: str,
        aclbf_results_df: pd.DataFrame = None,
        nominal_results_df: pd.DataFrame = None,
        trajopt2_results_df: pd.DataFrame = None,
        trajopt2_synthesis_times: List[float] = None,
        mpc_results_df: pd.DataFrame = None,
        mpc_trajopt_synthesis_times: List[float] = None,
        # Extra data for documentation
        n_sims_per_start: int = -1,
        n_x0s: int = -1,
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

        mpc_data_dict = None
        if mpc_results_df is not None:
            mpc_data_dict = get_avg_computation_time_from_df(mpc_results_df)

        # Save the data to txt file
        with open(table_name, "w") as f:
            comments = []
            if n_sims_per_start > 0:
                comments = [f"n_sims_per_start={n_sims_per_start}"]
            if n_x0s > 0:
                comments += [f"n_x0={n_x0s}"]

            comments += [f"commit_prefix={commit_prefix}"]
            comments += [f"version_number={version_number}"]

            lines = timing_data_to_latex_table(
                aclbf_data_dict,
                nominal_timing=nominal_data_dict,
                trajopt2_timing=trajopt2_data_dict,
                trajopt2_synthesis_times=trajopt2_synthesis_times,
                mpc_timing=mpc_data_dict,
                mpc_trajopt_synthesis_times=mpc_trajopt_synthesis_times,
                comments=comments,
            )

            f.writelines(lines)