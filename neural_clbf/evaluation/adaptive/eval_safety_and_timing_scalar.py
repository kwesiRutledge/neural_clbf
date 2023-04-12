"""
eval_safety_and_timing_scalar.py
Description:
    In this file, I will evaluate the safety and timing of the pusher slider system.
"""

import torch
import torch.nn as nn
import matplotlib
import yaml
from argparse import ArgumentParser

from neural_clbf.controllers import (
    NeuralaCLBFController, NeuralCLBFController
)
from neural_clbf.systems.adaptive import (
    ScalarCAPA2Demo, ControlAffineParameterAffineSystem
)
from neural_clbf.datamodules import EpisodicDataModuleAdaptive
from neural_clbf.experiments import (
    AdaptiveCLFContourExperiment, RolloutStateParameterSpaceExperiment,
    ExperimentSuite, ACLFRolloutTimingExperiment,
)
from neural_clbf.experiments.adaptive import (
    RolloutParameterConvergenceExperiment, CaseStudySafetyExperiment,
)

import numpy as np

from typing import Dict

import polytope as pc

def extract_hyperparams_from_args(args):
    """
    controller_ckpt, saved_hyperparams, saved_Vnn, controller_pt = extract_hyperparams_from_args()
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
    version_to_load = args.version_number

    scalar_capa2_log_file_dir = "../../training/adaptive/logs/scalar_demo_capa2_system/"
    scalar_capa2_log_file_dir += "commit_" + commit_prefix + "/version_" + str(version_to_load) + "/"

    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    controller_from_checkpoint = None
    if (args.checkpoint_filename is not None) and (args.checkpoint_filename != ""):
        ckpt_file = scalar_capa2_log_file_dir + "checkpoints/" + args.checkpoint_filename
        controller_from_checkpoint = NeuralaCLBFController.load_from_checkpoint(
            ckpt_file,
        )

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

    # Load the controller
    aclbf_controller = torch.load(
        scalar_capa2_log_file_dir + "controller.pt",
        map_location=torch.device('cpu'),
    )

    return controller_from_checkpoint, saved_hyperparams, saved_Vnn, aclbf_controller

def inflate_context_using_hyperparameters(hyperparams):
    """
    dynamics, scenarios, data_module, experiment_suite, x0 = inflate_context_using_hyperparameters(saved_hyperparams)
    Description:
        Inflates some useful constants from the hyperparameters file that was loaded.
    """

    # Constants
    simulation_dt = hyperparams["simulation_dt"]
    controller_period = hyperparams["controller_period"]

    # Get initial conditions for the experiment
    start_x = torch.arange(-0.3, 2.1, 0.2)
    start_x = start_x.reshape((start_x.shape[0], 1))

    # Define the scenarios
    wall_pos = hyperparams["nominal_scenario_wall_pos"]
    nominal_scenario = {"wall_position": wall_pos}
    scenarios = [
        nominal_scenario,
    ]

    # Define the range of possible uncertain parameters
    lb = [hyperparams["Theta_lb"]]
    ub = [hyperparams["Theta_ub"]]
    Theta = pc.box2poly(np.array([lb, ub]).T)

    # Define the dynamics model
    dynamics_model = ScalarCAPA2Demo(
        nominal_scenario,
        Theta,
        dt=simulation_dt,
        controller_dt=controller_period,
        scenarios=scenarios,
    )

    # Initialize the DataModule
    initial_conditions = [
        (1.0, 3.0),  # p_x
    ]
    data_module = EpisodicDataModuleAdaptive(
        dynamics_model,
        initial_conditions,
        trajectories_per_episode=0,
        trajectory_length=1,
        fixed_samples=10000,
        max_points=100000,
        val_split=0.1,
        batch_size=64,
        # quotas={"safe": 0.2, "unsafe": 0.2, "goal": 0.4},
    )

    # Define the experiment suite
    V_contour_experiment = AdaptiveCLFContourExperiment(
        "V_Contour",
        x_domain=[(-2.0, 2.0)],  # plotting domain
        theta_domain=[(-2.6, -1.4)],
        n_grid=30,
        x_axis_index=ScalarCAPA2Demo.X_DEMO,
        theta_axis_index=ScalarCAPA2Demo.P_DEMO,
        x_axis_label="$p_x$",
        theta_axis_label="$\\hat{\\theta}$",  # "$\\dot{\\theta}$",
        plot_unsafe_region=False,
    )
    rollout_experiment = RolloutStateParameterSpaceExperiment(
        "Rollout",
        start_x,
        ScalarCAPA2Demo.X_DEMO,
        "$x$",
        ScalarCAPA2Demo.P_DEMO,
        "$\\hat{\\theta}$",
        scenarios=scenarios,
        n_sims_per_start=1,
        t_sim=5.0,
    )
    rollout_experiment2 = ACLFRolloutTimingExperiment(
        "Rollout",
        start_x,
        ScalarCAPA2Demo.X_DEMO,
        "$x$",
        ScalarCAPA2Demo.P_DEMO,
        "$\\hat{\\theta}$",
        scenarios=scenarios,
        n_sims_per_start=1,
        t_sim=5.0,
    )
    rollout_experiment3 = RolloutParameterConvergenceExperiment(
        "Rollout (Parameter Convergence)",
        start_x,
        [ScalarCAPA2Demo.X_DEMO],
        ["$s$"],
        n_sims_per_start=1,
        hide_state_rollouts=True,
    )
    experiment_suite = ExperimentSuite([V_contour_experiment, rollout_experiment2, rollout_experiment3])

    return dynamics_model, scenarios, data_module, experiment_suite, start_x


def main(args):
    """
    main
    Description:
        This is the main function for this script.
        Runs all timings and other things.
    """
    # Constants

    # Load aclbf Data from file
    controller_ckpt, saved_hyperparams, saved_Vnn, controller_pt = extract_hyperparams_from_args(args)

    # Inflate context using hyperparams
    dynamics_model, scenarios, data_module, experiment_suite, x0 = inflate_context_using_hyperparameters(saved_hyperparams)
    controller_pt.experiment_suite = experiment_suite

    # Load trajectory optimization data from file
    #yaml_filename = "data/safety_scalar_case_study_data_Apr-10-2023-02:49:17.yml"
    yaml_filename = "data/safety_scalar_case_study_data_Apr-10-2023-03:30:46.yml"
    trajopt_file_data = yaml.load(open(yaml_filename, "r"), Loader=yaml.FullLoader)
    U_trajopt = torch.zeros(
        (trajopt_file_data["num_x0s"], trajopt_file_data["horizon"], dynamics_model.n_controls)
    )
    X_trajopt = torch.zeros(
        (trajopt_file_data["num_x0s"], trajopt_file_data["horizon"]+1, dynamics_model.n_dims)
    )
    for ic_index in range(trajopt_file_data["num_x0s"]):
        U_trajopt[ic_index, :, :] = torch.tensor(trajopt_file_data["U"+str(ic_index)])
        X_trajopt[ic_index, :, :] = torch.tensor(trajopt_file_data["X"+str(ic_index)])

    # Create the safety case study experiment
    safety_case_study_experiment = CaseStudySafetyExperiment(
        "Safety Case Study",
        x0,
        n_sims_per_start=1,
    )
    controller_pt.experiment_suite = ExperimentSuite([safety_case_study_experiment])

    # Run the experiments
    # - aCLBF Controller Safety Testing
    results_df = safety_case_study_experiment.run(controller_pt)
    counts = safety_case_study_experiment.tabulate_number_of_reaches(
        results_df, controller_pt.dynamics_model,
    )

    # - Nominal Controller Safety Testing
    nominal_results_df = safety_case_study_experiment.run_nominal_controlled(
        controller_pt.dynamics_model, controller_pt.controller_period,
    )
    counts_nominal = safety_case_study_experiment.tabulate_number_of_reaches(
        nominal_results_df, controller_pt.dynamics_model,
    )

    # - Optimized Trajectory Safety Testing
    trajopt_results_df = safety_case_study_experiment.run_trajopt_controlled(
        controller_pt.dynamics_model, controller_pt.controller_period, U_trajopt,
    )
    counts_trajopt = safety_case_study_experiment.tabulate_number_of_reaches(
        trajopt_results_df, controller_pt.dynamics_model,
    )

    # - (Hybrid) MPC Safety Testing
    mpc_results_df = safety_case_study_experiment.run_mpc_controlled(
        controller_pt.dynamics_model, controller_pt.controller_period,
        X_trajopt, U_trajopt,
    )
    counts_mpc = safety_case_study_experiment.tabulate_number_of_reaches(
        mpc_results_df, controller_pt.dynamics_model,
    )


    with open("../datafiles/scalar_demo_capa2/safety_case_study_results.txt", "w") as f:
        lines = safety_case_study_experiment.counts_to_latex_table(
            counts,
            nominal_counts=counts_nominal, trajopt_counts=counts_trajopt,
            mpc_counts=counts_mpc,
            comments=[f"n_sims_per_start={safety_case_study_experiment.n_sims_per_start} n_x0={x0.shape[0]}"]
        )
        print(lines)
        f.writelines(lines)


if __name__ == "__main__":
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument(
        "--commit_prefix", type=str, default="dfbf44c",
        help='First seven letters of the commit id of the code used to generate the data (default: "dfbf44c")',
    )
    parser.add_argument(
        '--version_number', type=int, default=0,
        help='Version number of the data to load (default: 0)',
    )
    parser.add_argument(
        '--checkpoint_filename', type=str, default="",
        help='Checkpoint filename to load (default: ""). (Example: \'epoch=5-step=845.ckpt\')',
    )
    args = parser.parse_args()

    # Begin Evaluation!
    main(args)