import torch
import torch.nn as nn
import matplotlib
from argparse import ArgumentParser

from neural_clbf.controllers import (
    NeuralaCLBFController, NeuralCLBFController
)
from neural_clbf.systems.adaptive import AdaptivePusherSliderStickingForceInput
from neural_clbf.datamodules import EpisodicDataModuleAdaptive
from neural_clbf.experiments import (
    AdaptiveCLFContourExperiment, RolloutStateParameterSpaceExperiment,
    ExperimentSuite, ACLFRolloutTimingExperiment,
    RolloutStateParameterSpaceExperimentMultiple,
    RolloutManipulatorConvergenceExperiment
)
from neural_clbf.experiments.adaptive import (
    aCLFCountourExperiment_StateSlices
)

import numpy as np

from typing import Dict

import polytope as pc

matplotlib.use('TkAgg')

def inflate_context_using_hyperparameters(hyperparams: Dict, args)->NeuralaCLBFController:
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
            [-0.5, -0.5, 0.0],
            [-0.5, -0.5, torch.pi/4.0],
            [-0.4, -0.5, 0.0],
            [-0.5, -0.3, 0.0],
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
    dynamics_model = AdaptivePusherSliderStickingForceInput(
        nominal_scenario,
        Theta,
        dt=simulation_dt,
        controller_dt=controller_period,
        scenarios=scenarios,
    )

    # Initialize the DataModule
    initial_conditions = [
        (0.0, 0.5),  # s_x
        (0.5, 1.0),  # s_y
        (0.0, -0.5), # s_theta
    ]
    data_module = EpisodicDataModuleAdaptive(
        dynamics_model,
        initial_conditions,
        trajectories_per_episode=0,
        trajectory_length=1,
        fixed_samples=hyperparams["n_fixed_samples"],
        max_points=100000,
        val_split=0.1,
        batch_size=hyperparams["batch_size"],
        # quotas={"safe": 0.2, "unsafe": 0.2, "goal": 0.4},
    )

    # Define the experiment suite
    lb_Vcontour = lb[hyperparams["contour_exp_theta_index"]]
    ub_Vcontour = ub[hyperparams["contour_exp_theta_index"]]
    theta_range_Vcontour = ub_Vcontour - lb_Vcontour

    x_ub, x_lb = dynamics_model.state_limits

    V_contour_experiment = AdaptiveCLFContourExperiment(
        "V_Contour",
        x_domain=[(x_lb[AdaptivePusherSliderStickingForceInput.S_X], x_ub[AdaptivePusherSliderStickingForceInput.S_X])],  # plotting domain
        theta_domain=[(lb_Vcontour-0.2*theta_range_Vcontour, ub_Vcontour+0.2*theta_range_Vcontour)],
        n_grid=30,
        x_axis_index=AdaptivePusherSliderStickingForceInput.S_X,
        theta_axis_index=AdaptivePusherSliderStickingForceInput.C_X,
        x_axis_label="$p_x$",
        theta_axis_label="$\\hat{c}_x$",  # "$\\dot{\\theta}$",
    )
    rollout_experiment2 = RolloutStateParameterSpaceExperimentMultiple(
        "Rollout (Multiple Slices)",
        hyperparams["start_x"],
        [AdaptivePusherSliderStickingForceInput.S_X, AdaptivePusherSliderStickingForceInput.S_Y, AdaptivePusherSliderStickingForceInput.S_X],
        ["$r_1$", "$v_1$", "$r_2$"],
        [AdaptivePusherSliderStickingForceInput.C_X, AdaptivePusherSliderStickingForceInput.C_X, AdaptivePusherSliderStickingForceInput.C_Y],
        ["$\\theta_1 (c_x)$", "$\\theta_1 (c_x)$", "$\\theta_1 (c_y)$"],
        scenarios=scenarios,
        n_sims_per_start=1,
        t_sim=hyperparams["rollout_experiment_horizon"],
    )
    rollout_experiment3 = RolloutManipulatorConvergenceExperiment(
        "Rollout Manipulator Convergence",
        hyperparams["start_x"],
        [AdaptivePusherSliderStickingForceInput.S_X, AdaptivePusherSliderStickingForceInput.S_Y, AdaptivePusherSliderStickingForceInput.S_THETA],
        ["$s_x$", "$s_y$", "$s_{\\theta}$"],
        scenarios=scenarios,
        n_sims_per_start=1,
        t_sim=3*hyperparams["rollout_experiment_horizon"],
    )

    start_x4 = torch.tensor([
        [0.0, 0.5, 0.0],
        [0.2, -0.2, np.pi / 2.0]
    ])
    rollout_experiment6 = RolloutManipulatorConvergenceExperiment(
        "Rollout Manipulator Convergence (Shorter travel time)",
        start_x4,
        [AdaptivePusherSliderStickingForceInput.S_X, AdaptivePusherSliderStickingForceInput.S_Y, AdaptivePusherSliderStickingForceInput.S_THETA],
        ["$s_x$", "$s_y$", "$s_{\\theta}$"],
        scenarios=scenarios,
        n_sims_per_start=1,
        t_sim=3*hyperparams["rollout_experiment_horizon"],
    )

    start_x5 = torch.tensor([
        [0.0, 0.5, 0.0],
        [0.2, -0.2, np.pi/2.0]
    ])
    rollout_experiment5 = RolloutStateParameterSpaceExperimentMultiple(
        "Rollout (Multiple Slices)",
        start_x5,
        [AdaptivePusherSliderStickingForceInput.S_X, AdaptivePusherSliderStickingForceInput.S_Y, AdaptivePusherSliderStickingForceInput.S_X],
        ["$r_1$", "$v_1$", "$r_2$"],
        [AdaptivePusherSliderStickingForceInput.C_X, AdaptivePusherSliderStickingForceInput.C_X, AdaptivePusherSliderStickingForceInput.C_Y],
        ["$\\theta_1 (c_x)$", "$\\theta_1 (c_x)$", "$\\theta_1 (c_y)$"],
        scenarios=scenarios,
        n_sims_per_start=1,
        t_sim=hyperparams["rollout_experiment_horizon"],
    )

    V_contour_experiment5 = aCLFCountourExperiment_StateSlices(
        "V_Contour (state slices only) 1",
        x_domain=[
            (x_lb[AdaptivePusherSliderStickingForceInput.S_X], x_ub[AdaptivePusherSliderStickingForceInput.S_X]),
            (x_lb[AdaptivePusherSliderStickingForceInput.S_Y], x_ub[AdaptivePusherSliderStickingForceInput.S_Y])
        ],  # plotting domain
        n_grid=50,
        x_axis_index=AdaptivePusherSliderStickingForceInput.S_X,
        y_axis_index=AdaptivePusherSliderStickingForceInput.S_Y,
        x_axis_label="$s_x$",
        y_axis_label="$s_y$",
        default_param_estimate=torch.tensor([dynamics_model.s_width/2, 0.0]).reshape((AdaptivePusherSliderStickingForceInput.N_PARAMETERS, 1)),
        default_state=torch.tensor([-0.5, -0.5, torch.pi/4]).reshape((AdaptivePusherSliderStickingForceInput.N_DIMS, 1)),
        plot_highlight_region=args.highlight_level is not None,
        default_highlight_level=args.highlight_level,
        plot_goal_region=True,
    )
    V_contour_experiment6 = aCLFCountourExperiment_StateSlices(
        "V_Contour (state slices only) 2",
        x_domain=[
            (x_lb[AdaptivePusherSliderStickingForceInput.S_X], x_ub[AdaptivePusherSliderStickingForceInput.S_X]),
            (x_lb[AdaptivePusherSliderStickingForceInput.S_Y], x_ub[AdaptivePusherSliderStickingForceInput.S_Y])
        ],  # plotting domain
        n_grid=50,
        x_axis_index=AdaptivePusherSliderStickingForceInput.S_X,
        y_axis_index=AdaptivePusherSliderStickingForceInput.S_Y,
        x_axis_label="$s_x$",
        y_axis_label="$s_y$",
        plot_unsafe_region=False,
        default_state=torch.tensor([-0.5, -0.5, torch.pi/4]).reshape(
            (dynamics_model.n_dims, 1),
        ),
        default_param_estimate=torch.tensor([dynamics_model.s_width/2.0, lb[1]*0.5]).reshape((AdaptivePusherSliderStickingForceInput.N_PARAMETERS, 1)),
        plot_highlight_region=args.highlight_level is not None,
        default_highlight_level=args.highlight_level,
        plot_goal_region=True,
    )
    V_contour_experiment7 = aCLFCountourExperiment_StateSlices(
        "V_Contour (state slices only) 3",
        x_domain=[
            (x_lb[AdaptivePusherSliderStickingForceInput.S_X], x_ub[AdaptivePusherSliderStickingForceInput.S_X]),
            (x_lb[AdaptivePusherSliderStickingForceInput.S_Y], x_ub[AdaptivePusherSliderStickingForceInput.S_Y])
        ],  # plotting domain
        n_grid=50,
        x_axis_index=AdaptivePusherSliderStickingForceInput.S_X,
        y_axis_index=AdaptivePusherSliderStickingForceInput.S_Y,
        x_axis_label="$s_x$",
        y_axis_label="$s_y$",
        plot_unsafe_region=False,
        default_state=torch.tensor([-0.5, -0.5, torch.pi / 4]).reshape(
            (dynamics_model.n_dims, 1),
        ),
        default_param_estimate=torch.tensor([dynamics_model.s_width / 2.0, ub[1] * 0.9]).reshape(
            (AdaptivePusherSliderStickingForceInput.N_PARAMETERS, 1)),
        plot_highlight_region=args.highlight_level is not None,
        default_highlight_level=args.highlight_level,
        plot_goal_region=True,
    )
    rollout_experiment4 = ACLFRolloutTimingExperiment(
        "aCLF Rollout Timing",
        start_x,
        AdaptivePusherSliderStickingForceInput.S_X,
        "$x$",
        AdaptivePusherSliderStickingForceInput.C_X,
        "$\\hat{c}_x$",
        scenarios=scenarios,
        n_sims_per_start=1,
        t_sim=10.0,
    )
    # experiment_suite = ExperimentSuite([V_contour_experiment, rollout_experiment3])
    # experiment_suite = ExperimentSuite([V_contour_experiment, rollout_experiment4])

    #experiment_suite = ExperimentSuite([V_contour_experiment, rollout_experiment2, rollout_experiment3, rollout_experiment4])
    experiment_suite = ExperimentSuite(
        [V_contour_experiment,
         V_contour_experiment5, V_contour_experiment6, V_contour_experiment7,
         rollout_experiment2, rollout_experiment3, rollout_experiment5, rollout_experiment6]
    )

    return dynamics_model, scenarios, data_module, experiment_suite

def plot_pusher_slider_data(args):
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    pusher_slider_log_file_dir = "../../training/adaptive/logs/pusher_slider_sticking_force_input/"
    # ckpt_file = pusher_slider_log_file_dir + "commit_bd8ad31/version_25/checkpoints/epoch=5-step=845.ckpt"

    commit_prefix = args.commit_prefix
    version_to_load = args.version_number
    hyperparam_log_file = pusher_slider_log_file_dir + "commit_" + commit_prefix + "/version_" + str(version_to_load) + "/hyperparams.pt"

    saved_Vnn = torch.load(
        pusher_slider_log_file_dir + "commit_" + commit_prefix + "/version_" + str(version_to_load) + "/Vnn.pt",
        map_location=torch.device('cpu'),
    )
    saved_hyperparams = torch.load(
        hyperparam_log_file,
        map_location=torch.device('cpu'),
    )

    dynamics_model, scenarios, data_module, experiment_suite = inflate_context_using_hyperparameters(saved_hyperparams, args)

    aclbf_controller = torch.load(
        pusher_slider_log_file_dir + "commit_" + commit_prefix + "/version_" + str(version_to_load) + "/controller.pt",
        map_location=torch.device('cpu'),
    )
    aclbf_controller.experiment_suite = experiment_suite
    aclbf_controller.dynamics_model = dynamics_model # Replace the CUDA-based dynamics model with our new, CPU-based one

    # Update parameters
    for experiment_idx in range(1, 3 +1):
    #     aclbf_controller.experiment_suite.experiments[experiment_idx].start_x = 50.0* torch.tensor([
    #         [0.5, 0.0, 0.0],
    #         [0.0, 0.5, 0.0],
    #         [0.0, 0.5, -0.4],
    #     ])

        aclbf_controller.experiment_suite.experiments[experiment_idx].t_sim = 15.0 #saved_hyperparams["rollout_experiment_horizon"]

    # Run the experiments and save the results
    fig_handles = aclbf_controller.experiment_suite.run_all_and_plot(
        aclbf_controller, display_plots=False
    )

    fig_titles = [
        "V-contour",
        "V-contour-xSlices-only-theta0",
        "V-contour-xSlices-only-theta_minus",
        "V-contour-xSlices-only-theta_plus",
        "V-trajectories1", "V-trajectories2", "V-trajectories3", "u-trajectories",
        "x-convergence",
        "V-trajectories4", "V-trajectories5", "V-trajectories6", "u-trajectories4",
        "x-convergence4",
    ]
    for fh_idx, fh in enumerate(fig_handles):
        fig_name, fig_obj = fh
        matplotlib.pyplot.figure(fig_obj.number)
        matplotlib.pyplot.savefig("../datafiles/pusher_slider_sticking/" + fig_titles[fh_idx] + ".png")


if __name__ == "__main__":
    # eval_inverted_pendulum()
    parser = ArgumentParser(
        description="This script evaluates a trained aCLBF controller for the AdaptivePusherSliderStickingForceInput system.",
    )
    parser.add_argument(
        '--commit_prefix', type=str, default="supercloud3",
        help='First seven letters of the commit id of the code used to generate the data (default: "supercloud1")'
    )
    parser.add_argument(
        '--version_number', type=int, default=1,
        help='Number of the experiment that was run under this commit (default: 1)',
    )
    parser.add_argument(
        '--highlight_level', type=float, default=None,
        help='Level to highlight in the V-contour plot (default: None)',
    )
    args = parser.parse_args()

    plot_pusher_slider_data(args)
