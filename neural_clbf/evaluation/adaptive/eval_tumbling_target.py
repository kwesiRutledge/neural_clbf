import torch
import torch.nn as nn
import matplotlib

from neural_clbf.controllers import (
    NeuralaCLBFController, NeuralCLBFController
)
from neural_clbf.systems.adaptive import TumblingTarget2
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
            [8.0, 5.0, 0.5, -0.1, 0.5, 0.5, 0.5],
            [5.0, 5.0, 0.0, 0.1, 0.0, 0.0, 0.0],
            [5.2, -10.0, 0.0, -0.1, 0.0, 0.0, 0.0],
            [-0.2, -1.0, 0.0, 0.1, 0.0, 0.0, 0.0],
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
    dynamics_model = TumblingTarget2(
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
        (-0.1, 0.1), # phi_d
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
    Theta_range = hyperparams["Theta_ub"][0] - hyperparams["Theta_lb"][0]
    V_contour_experiment = AdaptiveCLFContourExperiment(
        "V_Contour",
        x_domain=[(-2.0, 2.0)],  # plotting domain
        theta_domain=[(hyperparams["Theta_lb"][0]-0.5*Theta_range, hyperparams["Theta_ub"][0]+0.5*Theta_range)],
        n_grid=30,
        x_axis_index=TumblingTarget2.P_X,
        theta_axis_index=TumblingTarget2.OMEGA_0,
        x_axis_label="$p_x$",
        theta_axis_label="$\\hat{\\theta}$",  # "$\\dot{\\theta}$",
        plot_unsafe_region=False,
    )
    rollout_experiment2 = RolloutStateParameterSpaceExperimentMultiple(
        "Rollout",
        hyperparams["start_x"],
        [TumblingTarget2.P_X, TumblingTarget2.V_X, TumblingTarget2.P_Y],
        ["$r_1$", "$v_1$", "$r_2$"],
        [TumblingTarget2.OMEGA_0, TumblingTarget2.OMEGA_0, TumblingTarget2.OMEGA_0],
        ["$\\theta_1 (r_1^{(d)})$", "$\\theta_1 (r_1^{(d)})$", "$\\theta_1 (r_2^{(d)})$"],
        scenarios=scenarios,
        n_sims_per_start=1,
        t_sim=hyperparams["rollout_experiment_horizon"],
    )
    rollout_experiment3 = RolloutManipulatorConvergenceExperiment(
        "Rollout",
        hyperparams["start_x"],
        [TumblingTarget2.P_X, TumblingTarget2.V_X, TumblingTarget2.P_Y],
        ["$r_1$", "$r_2$", "$r_3$"],
        scenarios=scenarios,
        n_sims_per_start=1,
        t_sim=hyperparams["rollout_experiment_horizon"],
    )
    rollout_experiment4 = ACLFRolloutTimingExperiment(
        "Rollout",
        start_x,
        TumblingTarget2.P_X,
        "$x$",
        TumblingTarget2.OMEGA_0,
        "$\\hat{\\theta}$",
        scenarios=scenarios,
        n_sims_per_start=1,
        t_sim=10.0,
    )
    # experiment_suite = ExperimentSuite([V_contour_experiment, rollout_experiment3])
    # experiment_suite = ExperimentSuite([V_contour_experiment, rollout_experiment4])
    experiment_suite = ExperimentSuite([V_contour_experiment, rollout_experiment2, rollout_experiment3, rollout_experiment4])

    return dynamics_model, scenarios, data_module, experiment_suite

def plot_controlled_load_sharing():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    scalar_capa2_log_file_dir = "../../training/adaptive/logs/tumbling_target/"
    # ckpt_file = scalar_capa2_log_file_dir + "commit_bd8ad31/version_25/checkpoints/epoch=5-step=845.ckpt"

    version_to_load = 11
    commit_name = "commit_41749ec"
    hyperparam_log_file = scalar_capa2_log_file_dir + commit_name + "/version_" + str(version_to_load) + "/hyperparams.pt"

    saved_Vnn = torch.load(scalar_capa2_log_file_dir + commit_name + "/version_" + str(version_to_load) + "/Vnn.pt")
    saved_hyperparams = torch.load(hyperparam_log_file)

    dynamics_model, scenarios, data_module, experiment_suite = inflate_context_using_hyperparameters(saved_hyperparams)

    aclbf_controller = torch.load(scalar_capa2_log_file_dir + commit_name + "/version_" + str(version_to_load) + "/controller.pt")
    aclbf_controller.experiment_suite = experiment_suite

    # Update parameters
    for experiment_idx in range(1, 1 + 1):
        aclbf_controller.experiment_suite.experiments[experiment_idx].start_x = 10.0 * torch.tensor(
        [
            [8.0, 5.0, 0.5, -0.1, 0.5, 0.5, 0.5],
            [5.0, 5.0, 0.0, 0.1, 0.0, 0.0, 0.0],
            [5.2, -10.0, 0.0, -0.1, 0.0, 0.0, 0.0],
            [-0.2, -1.0, 0.0, 0.1, 0.0, 0.0, 0.0],
        ]
    )

        aclbf_controller.experiment_suite.experiments[experiment_idx].t_sim = 45.0 #saved_hyperparams["rollout_experiment_horizon"]

    # Run the experiments and save the results
    fig_handles = aclbf_controller.experiment_suite.run_all_and_plot(
        aclbf_controller, display_plots=False
    )

    fig_titles = ["V-contour", "V-trajectories1", "V-trajectories2", "V-trajectories3", "u-trajectories", "x-convergence", "bad-estimator-traj", "u-trajectories-again" ,"pt-comparison-cloud1"]
    for fh_idx, fh in enumerate(fig_handles):
        fig_name, fig_obj = fh
        matplotlib.pyplot.figure(fig_obj.number)
        matplotlib.pyplot.savefig("datafiles/tumbling_target/" + fig_titles[fh_idx] + ".png")


if __name__ == "__main__":
    # eval_inverted_pendulum()
    plot_controlled_load_sharing()
