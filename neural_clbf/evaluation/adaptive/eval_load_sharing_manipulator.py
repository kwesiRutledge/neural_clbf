import torch
import torch.nn as nn
import matplotlib
from argparse import ArgumentParser

from neural_clbf.controllers.adaptive import (
    NeuralaCLBFController,
)
from neural_clbf.systems.adaptive import (
    LoadSharingManipulator, ControlAffineParameterAffineSystem,
)
from neural_clbf.datamodules import EpisodicDataModuleAdaptive
from neural_clbf.experiments import (
    AdaptiveCLFContourExperiment, RolloutStateParameterSpaceExperiment,
    ExperimentSuite, ACLFRolloutTimingExperiment,
    RolloutStateParameterSpaceExperimentMultiple,
    RolloutManipulatorConvergenceExperiment
)
from neural_clbf.experiments.adaptive import (
    aCLFCountourExperiment_StateSlices,
)

import numpy as np

from typing import Dict, List, Tuple, Union, Any

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
    version_to_load = args.version

    scalar_capa2_log_file_dir = "../../training/adaptive/logs/load_sharing_manipulator/"
    scalar_capa2_log_file_dir += "commit_" + commit_prefix + "/version_" + version_to_load + "/"

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

    # Load the controller's state dict
    saved_state_dict = torch.load(
        scalar_capa2_log_file_dir + "state_dict.pt",
        map_location=torch.device('cpu'),
    )

    # Load the controller
    aclbf_controller = None
    # aclbf_controller = torch.load(
    #     scalar_capa2_log_file_dir + "controller.pt",
    #     map_location=torch.device('cpu'),
    # )

    return controller_from_checkpoint, saved_hyperparams, saved_Vnn, aclbf_controller, saved_state_dict

def inflate_context_using_hyperparameters(hyperparams: Dict, args)->NeuralaCLBFController:
    """
    inflate_context_using_hyperparameters
    Description

    """
    # Constants
    simulation_dt = hyperparams["simulation_dt"]
    controller_period = hyperparams["controller_period"]

    highlight_level = args.highlight_level

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
    x_ub, x_lb = dynamics_model.state_limits
    obs_center = np.array([
        nominal_scenario["obstacle_center_x"],
        nominal_scenario["obstacle_center_y"],
        nominal_scenario["obstacle_center_z"],
    ])
    obs_width = nominal_scenario["obstacle_width"]
    V_contour_experiment5 = aCLFCountourExperiment_StateSlices(
        "V_Contour (state slices only) 1",
        x_domain=[
            (x_lb[LoadSharingManipulator.P_X], x_ub[LoadSharingManipulator.P_X]),
            (x_lb[LoadSharingManipulator.P_Y], x_ub[LoadSharingManipulator.P_Y])
        ],  # plotting domain
        n_grid=50,
        x_axis_index=LoadSharingManipulator.P_X,
        y_axis_index=LoadSharingManipulator.P_Y,
        x_axis_label="$s_x$",
        y_axis_label="$s_y$",
        default_param_estimate=torch.tensor(dynamics_model.Theta.chebXc).reshape(
            (LoadSharingManipulator.N_PARAMETERS, 1)),
        default_state=torch.tensor([0.3, 0.3, 0.3, 0.0, 0.0, 0.0]).reshape(
            (LoadSharingManipulator.N_DIMS, 1)),
        plot_highlight_region=highlight_level is not None,
        default_highlight_level=highlight_level,
        plot_goal_region=True,
    )
    # V_contour_experiment6 = aCLFCountourExperiment_StateSlices(
    #     "V_Contour (state slices only) 2",
    #     x_domain=[
    #         (x_lb[AdaptivePusherSliderStickingForceInput.S_X], x_ub[AdaptivePusherSliderStickingForceInput.S_X]),
    #         (x_lb[AdaptivePusherSliderStickingForceInput.S_Y], x_ub[AdaptivePusherSliderStickingForceInput.S_Y])
    #     ],  # plotting domain
    #     n_grid=50,
    #     x_axis_index=AdaptivePusherSliderStickingForceInput.S_X,
    #     y_axis_index=AdaptivePusherSliderStickingForceInput.S_Y,
    #     x_axis_label="$s_x$",
    #     y_axis_label="$s_y$",
    #     plot_unsafe_region=False,
    #     default_state=torch.tensor([-0.5, -0.5, torch.pi / 4]).reshape(
    #         (dynamics_model.n_dims, 1),
    #     ),
    #     default_param_estimate=torch.tensor([dynamics_model.s_width / 2.0, lb[1] * 0.5]).reshape(
    #         (AdaptivePusherSliderStickingForceInput.N_PARAMETERS, 1)),
    #     plot_highlight_region=args.highlight_level is not None,
    #     default_highlight_level=args.highlight_level,
    #     plot_goal_region=True,
    # )
    # V_contour_experiment7 = aCLFCountourExperiment_StateSlices(
    #     "V_Contour (state slices only) 3",
    #     x_domain=[
    #         (x_lb[AdaptivePusherSliderStickingForceInput.S_X], x_ub[AdaptivePusherSliderStickingForceInput.S_X]),
    #         (x_lb[AdaptivePusherSliderStickingForceInput.S_Y], x_ub[AdaptivePusherSliderStickingForceInput.S_Y])
    #     ],  # plotting domain
    #     n_grid=50,
    #     x_axis_index=AdaptivePusherSliderStickingForceInput.S_X,
    #     y_axis_index=AdaptivePusherSliderStickingForceInput.S_Y,
    #     x_axis_label="$s_x$",
    #     y_axis_label="$s_y$",
    #     plot_unsafe_region=False,
    #     default_state=torch.tensor([-0.5, -0.5, torch.pi / 4]).reshape(
    #         (dynamics_model.n_dims, 1),
    #     ),
    #     default_param_estimate=torch.tensor([dynamics_model.s_width / 2.0, ub[1] * 0.9]).reshape(
    #         (AdaptivePusherSliderStickingForceInput.N_PARAMETERS, 1)),
    #     plot_highlight_region=args.highlight_level is not None,
    #     default_highlight_level=args.highlight_level,
    #     plot_goal_region=True,
    # )
    # experiment_suite = ExperimentSuite([V_contour_experiment, rollout_experiment3])
    # experiment_suite = ExperimentSuite([V_contour_experiment, rollout_experiment4])
    experiment_suite = ExperimentSuite([
        V_contour_experiment,
        rollout_experiment2, rollout_experiment3, rollout_experiment4,
        V_contour_experiment5,
    ])

    return dynamics_model, scenarios, data_module, experiment_suite

def extract_controller_from_state_dict(
    dynamics_model: ControlAffineParameterAffineSystem,
    scenarios: List[Dict[str, torch.Tensor]],
    data_module: EpisodicDataModuleAdaptive,
    saved_hyperparams: Dict[str, Any],
    experiment_suite: ExperimentSuite,
    saved_state_dict,
)->NeuralaCLBFController:
    """
    Extracts the controller from the state dict
    """

    # Input Processing
    clf_relaxation_penalty = 1e2
    if "clf_relaxation_penalty" in saved_hyperparams.keys():
        clf_relaxation_penalty = saved_hyperparams["clf_relaxation_penalty"]

    Q_u = np.diag([1.0, 1.0, 1.0])
    if "Q_u" in saved_hyperparams.keys():
        Q_u = saved_hyperparams["Q_u"]

    # algorithm
    controller_from_state_dict = NeuralaCLBFController(
        dynamics_model,
        scenarios,
        data_module,
        experiment_suite=experiment_suite,
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

    return controller_from_state_dict

def plot_controlled_load_sharing(args):
    controller_ckpt, saved_hyperparams, saved_Vnn, controller_pt, saved_state_dict = extract_hyperparams_from_args(args)

    dynamics_model, scenarios, data_module, experiment_suite = inflate_context_using_hyperparameters(
        saved_hyperparams,
        args,
    )
    dynamics_model.device = "cpu"
    # controller_pt.experiment_suite = experiment_suite
    # controller_pt.dynamics_model.device = "cpu"

    # Load new controller
    controller_from_state_dict = extract_controller_from_state_dict(
        dynamics_model,
        scenarios,
        data_module,
        saved_hyperparams,
        experiment_suite,
        saved_state_dict,
    )
    controller_from_state_dict.experiment_suite = experiment_suite
    controller_from_state_dict.dynamics_model.device = "cpu"

    # SELECT CONTROLLER UNDER TEST
    controller_under_test = controller_from_state_dict

    # Update parameters
    for experiment_idx in range(1, 1 + 1):
        controller_under_test.experiment_suite.experiments[experiment_idx].start_x = torch.tensor(
        [
            [0.3, -0.3, 0.0, 0.0, 0.0, 0.0],
            # [0.7, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.3, -0.5, 0.0, 0.0, 0.0, 0.0],
            # [0.0, -0.6, 0.0, 0.0, 0.0, 0.0],
            [0.3, -0.3, 0.4, 0.0, 0.0, 0.0],
            # [0.0, 0.0, 0.7, 0.0, 0.0, 0.0]
        ]
    )

        controller_under_test.experiment_suite.experiments[experiment_idx].t_sim = 45.0 #saved_hyperparams["rollout_experiment_horizon"]

    # Run the experiments and save the results
    fig_handles = controller_under_test.experiment_suite.run_all_and_plot(
        controller_under_test, display_plots=False
    )

    fig_titles = [
        "V-contour",
        "V-trajectories1", "V-trajectories2", "V-trajectories3",
        "u-trajectories",
        "x-convergence", "bad-estimator-traj", "u-trajectories-again", "pt-comparison-cloud1",
        "V-contour-slices1"
    ]
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
        '--version', type=str, default="0",
        help='Version number of the data to load (default: 0)',
    )
    parser.add_argument(
        '--checkpoint_filename', type=str, default="",
        help='Checkpoint filename to load (default: ""). (Example: \'epoch=5-step=845.ckpt\')',
    )
    parser.add_argument(
        '--highlight_level', type=float, default=0.1,
        help='Level to highlight in the V-contour plot (default: None)',
    )
    args = parser.parse_args()
    # Plot controlled load sharing
    plot_controlled_load_sharing(args)
