import torch
import torch.nn as nn
import matplotlib

from neural_clbf.controllers import (
    NeuralaCLBFController, NeuralCLBFController
)
from neural_clbf.systems.adaptive import ScalarCAPA2Demo
from neural_clbf.datamodules import EpisodicDataModuleAdaptive
from neural_clbf.experiments import (
    AdaptiveCLFContourExperiment, RolloutStateParameterSpaceExperiment,
    ExperimentSuite, ACLFRolloutTimingExperiment
)

import numpy as np

from typing import Dict

import polytope as pc

matplotlib.use('TkAgg')

def inflate_context_using_hyperparameters(hyperparams: Dict)->NeuralaCLBFController:
    """
    get_aclbf_controller
    Description

    """
    # Constants
    print(hyperparams)
    simulation_dt = hyperparams["simulation_dt"]
    controller_period = hyperparams["controller_period"]

    # Get initial conditions for the experiment
    start_x = torch.tensor(
        [
            [0.5],
            [0.7],
            [0.9],
            [1.5],
            [-0.5],
            [-0.7]
        ]
    )

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
        theta_domain=[(-2.6,-1.4)],
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
    experiment_suite = ExperimentSuite([V_contour_experiment, rollout_experiment2])

    return dynamics_model, scenarios, data_module, experiment_suite

def plot_controlled_scalar_capa2():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    scalar_capa2_log_file_dir = "../training/adaptive/logs/scalar_demo_capa2_system/"
    ckpt_file = scalar_capa2_log_file_dir + "commit_98c4671/version_25/checkpoints/epoch=5-step=845.ckpt"

    version_to_load = 43
    hyperparam_log_file = scalar_capa2_log_file_dir + "commit_98c4671/version_" + str(version_to_load) + "/hyperparams.pt"

    saved_Vnn = torch.load(scalar_capa2_log_file_dir + "commit_98c4671/version_" + str(version_to_load) + "/Vnn.pt")
    saved_hyperparams = torch.load(hyperparam_log_file)

    # print(saved_hyperparams)
    # print(saved_Vnn)
    # print(saved_Vnn[0])
    # for layer in saved_Vnn:
    #     print("layer: ", layer)
    #     if isinstance(layer, nn.Linear):
    #         print("layer.weight: ", layer.weight)

    dynamics_model, scenarios, data_module, experiment_suite = inflate_context_using_hyperparameters(saved_hyperparams)

    aclbf_controller = torch.load(scalar_capa2_log_file_dir + "commit_98c4671/version_" + str(version_to_load) + "/controller.pt")
    aclbf_controller.experiment_suite = experiment_suite

    # aclbf_controller = NeuralaCLBFController(
    #     dynamics_model,
    #     scenarios,
    #     data_module,
    #     experiment_suite=experiment_suite,
    #     clbf_hidden_layers=saved_hyperparams["clbf_hidden_layers"],
    #     clbf_hidden_size=saved_hyperparams["clbf_hidden_size"],
    #     clf_lambda=saved_hyperparams["clf_lambda"],
    #     safe_level=0.5,
    #     controller_period=saved_hyperparams["controller_period"],
    #     clf_relaxation_penalty=1e2,
    #     num_init_epochs=5,
    #     epochs_per_episode=100,
    #     barrier=False,
    #     saved_Vnn=saved_Vnn,
    # )

    # Update parameters
    for experiment_idx in range(1, 1 +1):
        aclbf_controller.experiment_suite.experiments[experiment_idx].start_x = torch.tensor(
            [
                [1.5],
                [0.9],
                [0.3],
                [0.0],
                [-0.3],
                [-0.9],
            ]
        )

    # Run the experiments and save the results
    fig_handles = aclbf_controller.experiment_suite.run_all_and_plot(
        aclbf_controller, display_plots=False
    )

    fig_titles = ["V-contour", "V-trajectories", "u-trajectories", "pt-comparison-cloud1"]
    for fh_idx, fh in enumerate(fig_handles):
        fig_name, fig_obj = fh
        matplotlib.pyplot.figure(fig_obj.number)
        matplotlib.pyplot.savefig("datafiles/scalar_demo_capa2/" + fig_titles[fh_idx] + ".png")


if __name__ == "__main__":
    # eval_inverted_pendulum()
    plot_controlled_scalar_capa2()
