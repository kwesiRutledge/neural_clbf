"""
eval_safety_and_timing_load_sharing_uncertainty_aware.py
Description:
    In this file, I will evaluate the safety and timing of the load sharing manipulator
    using the 4 different controllers we had in mind for this system.
"""

import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import yaml
import os
from argparse import ArgumentParser

from neural_clbf.controllers.adaptive import (
    NeuralaCLBFController3,
)
from neural_clbf.systems.adaptive import (
    LoadSharingManipulator, ControlAffineParameterAffineSystem,
)
from neural_clbf.datamodules import EpisodicDataModuleAdaptive
from neural_clbf.experiments import (
    AdaptiveCLFContourExperiment, RolloutStateParameterSpaceExperiment,
    ExperimentSuite, ACLFRolloutTimingExperiment,
)
from neural_clbf.experiments.adaptive.safety_case_study import (
    counts_to_latex_table, tabulate_number_of_reaches,
    CaseStudySafetyExperimentMPC, CaseStudySafetyExperimentTrajOpt2,
    create_initial_states_parameters_and_estimates,
    #CaseStudySafetyExperiment,
    save_timing_data_table,
)
from neural_clbf.experiments.adaptive_w_uncertainty.safety_case_study import (
    CaseStudySafetyExperiment,
)
from neural_clbf.experiments.adaptive import (
    RolloutParameterConvergenceExperiment,
)

from neural_clbf.evaluation.adaptive import (
    load_yaml_trajectory_data,
)

import numpy as np
import scipy.optimize as optimize

from typing import Dict

import polytope as pc

def extract_hyperparams_from_args(args):
    """
    controller_ckpt, saved_hyperparams, saved_Vnn, controller_pt, controller_state_dict = extract_hyperparams_from_args()
    Description:
        Load the following from the log file associate with the commit and version #:
        - Hyperparameters
        - Vnn
    Outputs:
        - controller_from_checkpoint: The controller loaded from the checkpoint file.
            If no checkpoint file name is given, then this will be none.
        - saved_hyperparams: The hyperparameters used to train the controller.
        - saved_Vnn: The Vnn used to define the controller.

    """
    # Constants
    commit_prefix = args.commit_prefix
    version_to_load = args.version

    scalar_capa2_log_file_dir = "../../training/adaptive/logs/load_sharing_manipulator/"
    scalar_capa2_log_file_dir += "commit_" + commit_prefix + "/version_" + str(version_to_load) + "/"

    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    controller_from_checkpoint = None
    # checkpoint_filenames = [each for each in os.listdir(scalar_capa2_log_file_dir + "checkpoints/") if each.endswith(".ckpt")]
    # selected_checkpoint = checkpoint_filenames[-1]
    #
    # ckpt_file = scalar_capa2_log_file_dir + "checkpoints/" + selected_checkpoint
    # controller_from_checkpoint = NeuralaCLBFController.load_from_checkpoint(
    #     ckpt_file,
    #     map_location=torch.device('cpu'),
    #     device="cpu",
    # )

    # Load the hyperparameters
    hyperparam_log_filename = scalar_capa2_log_file_dir + "hyperparams.pt"
    saved_hyperparams = torch.load(
        hyperparam_log_filename,
        map_location=torch.device('cpu'),
    )

    # Load the Vnn
    saved_Vnn = torch.load(
        scalar_capa2_log_file_dir + "Vnn.pt",
        map_location=torch.device('cpu'),
    )

    # Load the controller's state dict
    saved_state_dict = torch.load(
        scalar_capa2_log_file_dir + "state_dict.pt",
        map_location=torch.device('cpu'),
    )

    nominal_scenario = saved_hyperparams["nominal_scenario"]
    scenarios = [
        nominal_scenario,
    ]

    # Define the range of possible uncertain parameters
    lb = saved_hyperparams["Theta_lb"]
    ub = saved_hyperparams["Theta_ub"]
    Theta = pc.box2poly(np.array([lb, ub]).T)

    # Define the dynamics model
    dynamics_model = LoadSharingManipulator(
        nominal_scenario,
        Theta,
        dt=saved_hyperparams["simulation_dt"],
        controller_dt=saved_hyperparams["controller_period"],
        scenarios=scenarios,
    )

    # Initialize the DataModule
    initial_conditions = [
        (-0.4, 0.4),  # p_x
        (-0.4, 0.4),  # p_y
        (0.0, 0.7),  # p_z
        (-0.5, 0.5),  # v_x
        (-0.5, 0.5),  # v_y
        (-0.5, 0.5),  # v_z
    ]
    data_module = EpisodicDataModuleAdaptive(
        dynamics_model,
        initial_conditions,
        trajectories_per_episode=saved_hyperparams["trajectories_per_episode"],
        trajectory_length=saved_hyperparams["trajectory_length"],
        fixed_samples=saved_hyperparams["n_fixed_samples"],
        max_points=100000,
        val_split=0.1,
        batch_size=saved_hyperparams["batch_size"],
        quotas=saved_hyperparams["sample_quotas"],
        device=saved_hyperparams["accelerator"],
        num_workers=saved_hyperparams["num_cpu_cores"],
    )

    lb_Vcontour = lb[saved_hyperparams["contour_exp_theta_index"]]
    ub_Vcontour = ub[saved_hyperparams["contour_exp_theta_index"]]
    theta_range_Vcontour = ub_Vcontour - lb_Vcontour
    V_contour_experiment = AdaptiveCLFContourExperiment(
        "V_Contour",
        x_domain=[
            (dynamics_model.state_limits[1][LoadSharingManipulator.P_X],
             dynamics_model.state_limits[0][LoadSharingManipulator.P_X]),
        ],
        theta_domain=[(lb_Vcontour - 0.2 * theta_range_Vcontour, ub_Vcontour + 0.2 * theta_range_Vcontour)],
        n_grid=30,
        x_axis_index=LoadSharingManipulator.P_X,
        theta_axis_index=saved_hyperparams["contour_exp_theta_index"],
        x_axis_label="$r_1$",
        theta_axis_label="$\\theta_" + str(saved_hyperparams["contour_exp_theta_index"]) + "$",  # "$\\dot{\\theta}$",
        plot_unsafe_region=False,
    )

    clf_relaxation_penalty = 1e2
    if "clf_relaxation_penalty" in saved_hyperparams.keys():
        clf_relaxation_penalty = saved_hyperparams["clf_relaxation_penalty"]

    Q_u = np.diag([1.0 for i in range(dynamics_model.n_controls)])
    if "Q_u" in saved_hyperparams.keys():
        Q_u = saved_hyperparams["Q_u"]

    controller_from_state_dict = NeuralaCLBFController3(
        dynamics_model,
        scenarios,
        data_module,
        experiment_suite=ExperimentSuite([V_contour_experiment]),
        clbf_hidden_layers=saved_hyperparams["clbf_hidden_layers"],
        clbf_hidden_size=saved_hyperparams["clbf_hidden_size"],
        clf_lambda=saved_hyperparams["clf_lambda"],
        safe_level=saved_hyperparams["safe_level"],
        controller_period=saved_hyperparams["controller_period"],
        clf_relaxation_penalty=clf_relaxation_penalty,
        num_init_epochs=saved_hyperparams["num_init_epochs"],
        epochs_per_episode=100,
        barrier=saved_hyperparams["barrier"],
        Gamma_factor=saved_hyperparams["Gamma_factor"],
        include_oracle_loss=saved_hyperparams["include_oracle_loss"],
        Q_u=Q_u,
    )
    controller_from_state_dict.load_state_dict(
        saved_state_dict,
    )

    # Load the controller
    aclbf_controller = None
    # aclbf_controller = torch.load(
    #     scalar_capa2_log_file_dir + "controller.pt",
    #     map_location=torch.device('cpu'),
    # )
    # aclbf_controller.dynamics_model.device = "cpu"

    return controller_from_checkpoint, saved_hyperparams, saved_Vnn, aclbf_controller, controller_from_state_dict

def inflate_context_using_hyperparameters(hyperparams):
    """
    dynamics, scenarios, data_module, experiment_suite, x0 = inflate_context_using_hyperparameters(saved_hyperparams)
    Description:
        Inflates some useful constants from the hyperparameters file that was loaded.
    """

    # Constants
    simulation_dt = hyperparams["simulation_dt"]
    controller_period = hyperparams["controller_period"]

    # Define the scenarios
    nominal_scenario = hyperparams["nominal_scenario"]
    scenarios = [
        nominal_scenario,
    ]

    # Get initial conditions for the experiment
    obs = torch.tensor([
        nominal_scenario["obstacle_center_x"],
        nominal_scenario["obstacle_center_y"],
        nominal_scenario["obstacle_center_z"],
    ])
    obs_rad = nominal_scenario["obstacle_width"] / 2.0
    # start_x = torch.tensor(
    #     [
    #         [obs[0],                obs[1]-4.5*obs_rad,     obs[2], 0.1, 0.0, 0.0],
    #         [obs[0],                obs[1]-4.5*obs_rad,     obs[2]+0.5*obs_rad, 0.0, 0.1, 0.0],
    #         [obs[0],                obs[1]-4.5 * obs_rad,   obs[2]-0.5*obs_rad, 0.0, 0.1, 0.0],
    #         [obs[0]-0.5*obs_rad,    obs[1]-4.5*obs_rad,     obs[2], 0.0, 0.1, 0.0],
    #         [obs[0]-0.5*obs_rad,    obs[1]-4.5*obs_rad,     obs[2]-0.5*obs_rad, 0.0, 0.0, 0.1],
    #         [obs[0]-0.5*obs_rad,    obs[1]-4.5 * obs_rad,   obs[2]+0.5*obs_rad, 0.0, 0.1, 0.0],
    #         [obs[0]-0.5*obs_rad,    obs[1]-4.5*obs_rad,     obs[2]-0.5*obs_rad, 0.1, 0.0, 0.0],
    #         [obs[0]+0.5*obs_rad,    obs[1]-4.5*obs_rad,     obs[2], 0.0, 0.0, 0.0],
    #         [obs[0]+0.5 * obs_rad,  obs[1]-4.5*obs_rad,     obs[2]+0.5*obs_rad, 0.0, 0.0, 0.0],
    #         [obs[0]+0.5*obs_rad,    obs[1]-4.5*obs_rad,     obs[2]-0.5*obs_rad, 0.0, 0.0, 0.1],
    #         # [0.3,  -0.4, 0.3, 0.0, 0.0, 0.0],
    #         # [0.2, -0.1, 0.3, 0.0, 0.0, 0.0],
    #         # [0.1, -0.05, 0.3, 0.0, 0.0, 0.0],
    #         # [0.1, -0.1, 0.3, 0.0, 0.5, 0.0],
    #     ]
    # )
    start_x = torch.tensor(
        [
            [obs[0], obs[1] - 4.5 * obs_rad, obs[2], 0.1, 0.0, 0.0],
            [obs[0], obs[1] - 4.5 * obs_rad, obs[2] + 0.5 * obs_rad, 0.0, 0.1, 0.0],
            [obs[0], obs[1] - 4.5 * obs_rad, obs[2] - 0.5 * obs_rad, 0.0, 0.1, 0.0],
            [obs[0] + 0.5 * obs_rad, obs[1] - 4.5 * obs_rad, obs[2], 0.0, 0.1, 0.0],
            [obs[0] + 0.5 * obs_rad, obs[1] - 4.5 * obs_rad, obs[2] - 0.5 * obs_rad, 0.0, 0.0, 0.1],
            [obs[0] + 0.5 * obs_rad, obs[1] - 4.5 * obs_rad, obs[2] + 0.5 * obs_rad, 0.0, 0.1, 0.0],
            [obs[0] + 0.5 * obs_rad, obs[1] - 4.5 * obs_rad, obs[2] - 0.5 * obs_rad, 0.1, 0.0, 0.0],
            [obs[0] + 0.5 * obs_rad, obs[1] - 4.5 * obs_rad, obs[2], 0.0, 0.0, 0.0],
            [obs[0] + 1.5 * obs_rad, obs[1] - 4.5 * obs_rad, obs[2] + 0.5 * obs_rad, 0.0, 0.0, 0.0],
            [obs[0] + 0.5 * obs_rad, obs[1] - 4.5 * obs_rad, obs[2] - 0.5 * obs_rad, 0.0, 0.0, 0.1],
            # [0.3,  -0.4, 0.3, 0.0, 0.0, 0.0],
            # [0.2, -0.1, 0.3, 0.0, 0.0, 0.0],
            # [0.1, -0.05, 0.3, 0.0, 0.0, 0.0],
            # [0.1, -0.1, 0.3, 0.0, 0.5, 0.0],
        ]
    )

    # Define the range of possible uncertain parameters
    lb = hyperparams["Theta_lb"]
    ub = hyperparams["Theta_ub"]
    Theta = pc.box2poly(np.array([lb, ub]).T)

    # Define the dynamics model
    dynamics_model = LoadSharingManipulator(
        nominal_scenario,
        Theta,
        dt=simulation_dt,
        controller_dt=controller_period,
        scenarios=scenarios,
    )

    # Initialize the DataModule
    initial_conditions = [
        (-np.pi / 4, np.pi / 4),  # p_x
        (-1.0, 1.0),  # p_y
        (-np.pi / 4, np.pi / 4),  # p_z
        (-1.0, 1.0),  # v_x
        (-1.0, 1.0),  # v_y
        (-1.0, 1.0),  # v_z
    ]
    data_module = EpisodicDataModuleAdaptive(
        dynamics_model,
        initial_conditions,
        trajectories_per_episode=1,
        trajectory_length=1,
        fixed_samples=hyperparams["n_fixed_samples"],
        max_points=100000,
        val_split=0.1,
        batch_size=64,
        quotas=hyperparams["sample_quotas"],
        device=hyperparams["accelerator"],
    )

    # Define the experiment suite
    V_contour_experiment = AdaptiveCLFContourExperiment(
        "V_Contour",
        x_domain=[(-2.0, 2.0)],  # plotting domain
        theta_domain=[(-2.6, -1.4)],
        n_grid=30,
        x_axis_index=LoadSharingManipulator.P_X,
        theta_axis_index=LoadSharingManipulator.P_X_DES,
        x_axis_label="$p_x$",
        theta_axis_label="$\\hat{\\theta}$",  # "$\\dot{\\theta}$",
        plot_unsafe_region=False,
    )
    rollout_experiment = RolloutStateParameterSpaceExperiment(
        "Rollout",
        start_x,
        LoadSharingManipulator.P_X,
        "$x$",
        LoadSharingManipulator.P_X_DES,
        "$\\hat{\\theta}$",
        scenarios=scenarios,
        n_sims_per_start=1,
        t_sim=5.0,
    )
    rollout_experiment2 = ACLFRolloutTimingExperiment(
        "Rollout",
        start_x,
        LoadSharingManipulator.P_X,
        "$x$",
        LoadSharingManipulator.P_X_DES,
        "$\\hat{\\theta}$",
        scenarios=scenarios,
        n_sims_per_start=1,
        t_sim=5.0,
    )
    rollout_experiment3 = RolloutParameterConvergenceExperiment(
        "Rollout (Parameter Convergence)",
        start_x,
        [LoadSharingManipulator.P_X],
        ["$s$"],
        n_sims_per_start=1,
        hide_state_rollouts=True,
    )
    experiment_suite = ExperimentSuite([V_contour_experiment, rollout_experiment2, rollout_experiment3])

    return dynamics_model, scenarios, data_module, experiment_suite, start_x


def create_case_study_experiments(
        x0: torch.tensor,
        theta_hat0: torch.tensor,
        theta0: torch.tensor,
        t_sim: float,
        controller_to_test: "Controller",
):
    """
    safety_case_study_experiment, safety_trajopt_exp, safety_mpc_exp = create_case_study_experiments(
        x0, theta_hat0, theta0,
        t_sim, controller_to_test,
    )
    Description:

    """

    # Algorithms
    safety_case_study_experiment = CaseStudySafetyExperiment(
        "Safety Case Study",
        x0, theta_hat0, theta0,
        n_sims_per_start=1,
        t_sim=t_sim,
        plot_x_indices=[LoadSharingManipulator.P_X, LoadSharingManipulator.P_Y, LoadSharingManipulator.P_Z],
        plot_x_labels=["$p_x$", "$p_y$", "$p_z$"],
    )
    controller_to_test.experiment_suite = ExperimentSuite([safety_case_study_experiment])

    safety_trajopt_exp = CaseStudySafetyExperimentTrajOpt2(
        "Safety Case Study - TrajOpt2",
        x0, theta_hat0, theta0,
        n_sims_per_start=1,
        t_sim=t_sim,
        plot_x_indices=[LoadSharingManipulator.P_X, LoadSharingManipulator.P_Y, LoadSharingManipulator.P_Z],
        plot_x_labels=["$p_x$", "$p_y$", "$p_z$"],
    )

    safety_mpc_exp = CaseStudySafetyExperimentMPC(
        "Safety Case Study - MPC",
        x0, theta_hat0, theta0,
        n_sims_per_start=1,
        t_sim=t_sim,
        plot_x_indices=[LoadSharingManipulator.P_X, LoadSharingManipulator.P_Y, LoadSharingManipulator.P_Z],
        plot_x_labels=["$p_x$", "$p_y$", "$p_z$"],
    )

    return safety_case_study_experiment, safety_trajopt_exp, safety_mpc_exp

def main(args):
    """
    main
    Description:
        This is the main function for this script.
        Runs all timings and other things.
    """
    # Constants
    t_sim = 15.0

    # Load aclbf Data from file
    controller_ckpt, saved_hyperparams, saved_Vnn, controller_pt, controller_state_dict = extract_hyperparams_from_args(args)

    # Inflate context using hyperparams
    dynamics_model, scenarios, data_module, experiment_suite, x0 = inflate_context_using_hyperparameters(saved_hyperparams)

    controller_to_test = controller_state_dict
    controller_to_test.experiment_suite = experiment_suite

    # Create trajectories
    _, theta0, theta_hat0 = create_initial_states_parameters_and_estimates(
        dynamics_model, x0,
        n_sims_per_start=1,
    )

    # Load trajectory optimization data from file
    #yaml_filename = "data/safety_scalar_case_study_data_Apr-10-2023-02:49:17.yml"
    # yaml_filename = "data/safety_scalar_case_study_data_Apr-10-2023-03:30:46.yml"
    if args.yaml_filename is not None:
        U_trajopt, X_trajopt = load_yaml_trajectory_data(args.yaml_filename)

    # Create the safety case study experiment
    safety_case_study_experiment, safety_trajopt_exp, safety_mpc_exp = create_case_study_experiments(
        x0, theta_hat0, theta0,
        t_sim, controller_to_test,
    )

    # Run the experiments
    aclbf_counts, aclbf_results_df = None, None
    counts_nominal, nominal_results_df = None, None
    counts_trajopt2, trajopt2_results_df, trajopt2_synthesis_times = None, None, None
    counts_trajopt, trajopt_results_df = None, None
    counts_mpc, mpc_results_df = None, None

    # - aCLBF Controller Safety Testing
    #controller_to_test.clf_relaxation_penalty = 1e4
    aclbf_results_df = safety_case_study_experiment.run(controller_to_test)
    aclbf_counts = tabulate_number_of_reaches(
        aclbf_results_df, controller_to_test.dynamics_model,
    )

    # - Nominal Controller Safety Testing
    nominal_results_df = safety_case_study_experiment.run_nominal_controlled(
        controller_to_test.dynamics_model, controller_to_test.controller_period,
    )
    counts_nominal = tabulate_number_of_reaches(
        nominal_results_df, controller_to_test.dynamics_model,
    )

    # - Optimized Trajectory Safety Testing #2 (in-function traj opt)
    def lsm_update(t, x, u, params):
        """
        dxdt = lsm_update(t,x,u, params)
        Description:
            This function defines the dynamics of the system.
        """
        # Constants
        m = dynamics_model.m
        gravity = 9.81
        theta = params.get("theta", np.array([-0.15, 0.4, 0.1]))

        # Unpack the state
        p = x[0:3]
        v = x[3:6]

        # Algorithm
        f = np.zeros((6,))
        f[0:3] = v
        f[3:6] = (1.0 / m) * np.diag([dynamics_model.K_x, dynamics_model.K_y, dynamics_model.K_z]) @ p
        f[-1] = f[-1] - gravity

        F = (1.0 / m) * np.vstack(
            (np.zeros((3, dynamics_model.n_controls)), -np.diag([dynamics_model.K_x, dynamics_model.K_y, dynamics_model.K_z]))
        )

        g = (1.0 / m) * np.vstack(
            (np.zeros((3, dynamics_model.n_controls)), np.eye(dynamics_model.n_controls))
        )

        return f + F @ theta + g @ u

    obs_center = np.array([
        dynamics_model.nominal_scenario["obstacle_center_x"],
        dynamics_model.nominal_scenario["obstacle_center_y"],
        dynamics_model.nominal_scenario["obstacle_center_z"],
    ])
    constraints = []
    constraints.append(
        (
            optimize.NonlinearConstraint,
            lambda x, u: np.linalg.norm(x[0:3] - obs_center),
            1.5 * (dynamics_model.nominal_scenario["obstacle_width"] / 2.0),
            float('Inf'),
        )
    )

    trajopt2_results_df, trajopt2_synthesis_times, X_trajopt, U_trajopt, t_trajopt, trajopt_durations = safety_trajopt_exp.run(
        controller_to_test.dynamics_model, controller_to_test.controller_period,
        lsm_update,
        Tf=t_sim,
        P=np.diag([5.0e3, 5.0e3, 5.0e3, 5.0e3, 5.0e3, 5.0e3]),
        Q=np.diag([1.0e2, 1.0e2, 1.0e2, 3.0e1, 3.0e1, 3.0e1]),
        R=np.diag([0.0e-2, 0.0e-2, 1.0e-1]),
        N_timepts=20,
        u0=np.array([-1.0, 1.0, 100.0]),
        constraints=constraints,
    )
    counts_trajopt2 = tabulate_number_of_reaches(
        trajopt2_results_df, controller_to_test.dynamics_model,
    )

    # - MPC Controller Safety Testing
    mpc_results_df, mpc_traj_synthesis_times, X_mpc_trajopt, U_mpc_trajopt, t_mpc_trajopt, mpc_trajopt_durations = safety_mpc_exp.run(
        controller_to_test.dynamics_model, controller_to_test.controller_period,
        lsm_update,
        Tf=t_sim,
        P=np.diag([5.0e5, 5.0e5, 5.0e5, 5.0e5, 5.0e5, 5.0e5]),
        Q=np.diag([1.0e4, 1.0e4, 1.0e4, 3.0e3, 3.0e3, 3.0e3]),
        R=np.diag([0.0e-2, 0.0e-2, 0.0e-2]),
        N_timepts=20,
        u0=np.array([-1.0, 1.0, 100.0]),
        constraints=constraints,
        X_trajopt=X_trajopt,
        U_trajopt=U_trajopt,
        t_trajopt=t_trajopt,
        trajopt_durations=trajopt_durations,
    )
    counts_mpc = tabulate_number_of_reaches(
        mpc_results_df, controller_to_test.dynamics_model,
    )

    if args.yaml_filename is not None:

        # - Optimized Trajectory Safety Testing
        trajopt_results_df = safety_case_study_experiment.run_trajopt_controlled(
            controller_to_test.dynamics_model, controller_to_test.controller_period, U_trajopt,
        )
        counts_trajopt = tabulate_number_of_reaches(
            trajopt_results_df, controller_to_test.dynamics_model,
        )

        # - (Hybrid) MPC Safety Testing
        mpc_results_df = safety_case_study_experiment.run_mpc_controlled(
            controller_to_test.dynamics_model, controller_to_test.controller_period,
            X_trajopt, U_trajopt,
        )
        counts_mpc = tabulate_number_of_reaches(
            mpc_results_df, controller_to_test.dynamics_model,
        )


    with open("../datafiles/load_sharing/safety_case_study_results.txt", "w") as f:
        comments = [f"n_sims_per_start={safety_case_study_experiment.n_sims_per_start}"]
        comments += [f"n_x0={x0.shape[0]}"]
        comments += [f"commit_prefix={args.commit_prefix}"]
        comments += [f"version_number={args.version}"]

        lines = counts_to_latex_table(
            aclbf_counts=aclbf_counts,
            nominal_counts=counts_nominal,
            trajopt_counts=counts_trajopt, trajopt2_counts=counts_trajopt2,
            mpc_counts=counts_mpc,
            comments=comments,
        )
        f.writelines(lines)

    # Save Timing Results to Table
    save_timing_data_table(
        "../datafiles/load_sharing/safety_case_study_timing_results.txt",
        args.commit_prefix, args.version,
        aclbf_results_df=aclbf_results_df,
        nominal_results_df=nominal_results_df,
        trajopt2_results_df=trajopt2_results_df,
        trajopt2_synthesis_times=trajopt2_synthesis_times,
        mpc_results_df=mpc_results_df,
        mpc_trajopt_synthesis_times=trajopt2_synthesis_times,
        n_sims_per_start=safety_case_study_experiment.n_sims_per_start,
        n_x0s=x0.shape[0],
    )

    # fig_handles = controller_to_test.experiment_suite.run_all_and_plot(
    #     controller_to_test, display_plots=False
    # )
    fig_handles = safety_case_study_experiment.plot(
        controller_to_test,
        aclbf_results_df,
        nominal_results_df=nominal_results_df,
        trajopt2_results_df=trajopt2_results_df,
        mpc_results_df=mpc_results_df,
        display_plots=False,
    )

    fig_titles = [
        "case_study1-aclbf-perf",
        "case_study1-nominal-perf",
        "case_study1-trajopt2-perf",
        "case_study1-mpc-perf",
    ]
    for fh_idx, fh in enumerate(fig_handles):
        fig_name, fig_obj = fh
        matplotlib.pyplot.figure(fig_obj.number)
        matplotlib.pyplot.savefig("../datafiles/load_sharing/" + fig_titles[fh_idx] + ".png")


if __name__ == "__main__":
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument(
        "--commit_prefix", type=str, default="dfbf44c",
        help='First seven letters of the commit id of the code used to generate the data (default: "dfbf44c")',
    )
    parser.add_argument(
        '--version', type=str, default="0",
        help='Version number of the data to load (default: 0)',
    )
    parser.add_argument(
        '--checkpoint_filename', type=str, default="",
        help='Checkpoint filename to load (default: ""). (Example: \'epoch=5-step=845.ckpt\')',
    )
    parser.add_argument(
        '--yaml_filename', type=str, default=None,
        help='Checkpoint filename to load (default: None). (Example: \'epoch=5-step=845.ckpt\')',
    )
    args = parser.parse_args()

    # Begin Evaluation!
    main(args)