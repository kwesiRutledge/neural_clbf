import torch
import torch.nn as nn
import matplotlib
from argparse import ArgumentParser

from neural_clbf.controllers.adaptive import (
    NeuralaCLBFController,
)
from neural_clbf.systems.adaptive import LoadSharingManipulator
from neural_clbf.datamodules import EpisodicDataModuleAdaptive
from neural_clbf.experiments import (
    AdaptiveCLFContourExperiment, RolloutStateParameterSpaceExperiment,
    ExperimentSuite, ACLFRolloutTimingExperiment,
    RolloutStateParameterSpaceExperimentMultiple,
    RolloutManipulatorConvergenceExperiment
)

import numpy as np

from typing import Dict

import polytope as pc

matplotlib.use('TkAgg')

def extract_hyperparams_from_args(args):
    """
    controller_ckpt, saved_hyperparams, saved_Vnn, controller_pt = extract_hyperparams_from_args(args)
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

    scalar_capa2_log_file_dir = "../../training/adaptive/logs/load_sharing_manipulator/"
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

def inflate_context_using_hyperparameters(hyperparams: Dict)->NeuralaCLBFController:
    """
    inflate_context_using_hyperparameters
    Description

    """
    # Constants
    simulation_dt = hyperparams["simulation_dt"]
    controller_period = hyperparams["controller_period"]

    # Get initial conditions for the experiment
    start_x = torch.tensor(
        [
            [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.7, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
            [0.0, -0.6, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -0.4, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.7, 0.0, 0.0, 0.0]
        ]
    )

    # Define the scenarios
    nominal_scenario = hyperparams["nominal_scenario"]
    scenarios = [
        nominal_scenario,
    ]

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
        (0.0, 0.5),  # p_x
        (0.5, 1.0),  # p_y
        (0.0, -0.5), # p_z
        (0.0, 0.0),  # v_x
        (0.0, 0.0),  # v_y
        (0.0, 0.0),  # v_z
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
    lb_Vcontour = lb[hyperparams["contour_exp_theta_index"]]
    ub_Vcontour = ub[hyperparams["contour_exp_theta_index"]]
    theta_range_Vcontour = ub_Vcontour - lb_Vcontour
    V_contour_experiment = AdaptiveCLFContourExperiment(
        "V_Contour",
        x_domain=[(-2.0, 2.0)],  # plotting domain
        theta_domain=[(lb_Vcontour-0.2*theta_range_Vcontour, ub_Vcontour+0.2*theta_range_Vcontour)],
        n_grid=30,
        x_axis_index=LoadSharingManipulator.P_X,
        theta_axis_index=LoadSharingManipulator.P_X_DES,
        x_axis_label="$p_x$",
        theta_axis_label="$\\hat{\\theta}$",  # "$\\dot{\\theta}$",
        plot_unsafe_region=False,
    )
    rollout_experiment2 = RolloutStateParameterSpaceExperimentMultiple(
        "Rollout (Multiple Slices)",
        hyperparams["start_x"],
        [LoadSharingManipulator.P_X, LoadSharingManipulator.V_X, LoadSharingManipulator.P_Y],
        ["$r_1$", "$v_1$", "$r_2$"],
        [LoadSharingManipulator.P_X_DES, LoadSharingManipulator.P_X_DES, LoadSharingManipulator.P_Y],
        ["$\\theta_1 (r_1^{(d)})$", "$\\theta_1 (r_1^{(d)})$", "$\\theta_1 (r_2^{(d)})$"],
        scenarios=scenarios,
        n_sims_per_start=1,
        t_sim=hyperparams["rollout_experiment_horizon"],
    )
    rollout_experiment3 = RolloutManipulatorConvergenceExperiment(
        "Rollout Manipulator Convergence",
        hyperparams["start_x"],
        [LoadSharingManipulator.P_X, LoadSharingManipulator.V_X, LoadSharingManipulator.P_Y],
        ["$r_1$", "$r_2$", "$r_3$"],
        scenarios=scenarios,
        n_sims_per_start=1,
        t_sim=3*hyperparams["rollout_experiment_horizon"],
    )
    rollout_experiment4 = ACLFRolloutTimingExperiment(
        "aCLF Rollout Timing",
        start_x,
        LoadSharingManipulator.P_X,
        "$x$",
        LoadSharingManipulator.P_X_DES,
        "$\\hat{\\theta}$",
        scenarios=scenarios,
        n_sims_per_start=1,
        t_sim=10.0,
    )
    # experiment_suite = ExperimentSuite([V_contour_experiment, rollout_experiment3])
    # experiment_suite = ExperimentSuite([V_contour_experiment, rollout_experiment4])
    experiment_suite = ExperimentSuite([V_contour_experiment, rollout_experiment2, rollout_experiment3, rollout_experiment4])

    return dynamics_model, scenarios, data_module, experiment_suite

def plot_controlled_load_sharing(args):
    controller_ckpt, saved_hyperparams, saved_Vnn, controller_pt = extract_hyperparams_from_args(args)

    dynamics_model, scenarios, data_module, experiment_suite = inflate_context_using_hyperparameters(saved_hyperparams)
    controller_pt.experiment_suite = experiment_suite
    controller_pt.dynamics_model.device = "cpu"

    # Update parameters
    for experiment_idx in range(1, 1 + 1):
        controller_pt.experiment_suite.experiments[experiment_idx].start_x = torch.tensor(
        [
            [0.3, -0.3, 0.0, 0.0, 0.0, 0.0],
            # [0.7, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.3, -0.5, 0.0, 0.0, 0.0, 0.0],
            # [0.0, -0.6, 0.0, 0.0, 0.0, 0.0],
            [0.3, -0.3, 0.4, 0.0, 0.0, 0.0],
            # [0.0, 0.0, 0.7, 0.0, 0.0, 0.0]
        ]
    )

        controller_pt.experiment_suite.experiments[experiment_idx].t_sim = 45.0 #saved_hyperparams["rollout_experiment_horizon"]

    # Run the experiments and save the results
    fig_handles = controller_pt.experiment_suite.run_all_and_plot(
        controller_pt, display_plots=False
    )

    fig_titles = ["V-contour", "V-trajectories1", "V-trajectories2", "V-trajectories3", "u-trajectories", "x-convergence", "bad-estimator-traj", "u-trajectories-again", "pt-comparison-cloud1"]
    for fh_idx, fh in enumerate(fig_handles):
        fig_name, fig_obj = fh
        matplotlib.pyplot.figure(fig_obj.number)
        matplotlib.pyplot.savefig("../datafiles/load_sharing/" + fig_titles[fh_idx] + ".png")


if __name__ == "__main__":
    # Parse arguments
    parser = ArgumentParser(description="Evaluating and plotting some data from the Load Sharing Manipulator Example.")
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
    # Plot controlled load sharing
    plot_controlled_load_sharing(args)
