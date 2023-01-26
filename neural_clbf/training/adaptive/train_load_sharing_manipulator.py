"""
train_load_sharing_manipulator.py
Description:
    This script trains an aCLBF for the load sharing manipulator system defined in systems/load_sharing_manipulator.py.
"""

from argparse import ArgumentParser

import torch
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import numpy as np

from neural_clbf.controllers import (
    NeuralCLBFController, NeuralaCLBFController
)
from neural_clbf.datamodules import (
    EpisodicDataModule, EpisodicDataModuleAdaptive
)
from neural_clbf.systems.adaptive import LoadSharingManipulator
from neural_clbf.experiments import (
    ExperimentSuite,
    CLFContourExperiment, AdaptiveCLFContourExperiment,
    RolloutStateSpaceExperiment,
)
from neural_clbf.training.utils import current_git_hash
import polytope as pc

from typing import Dict

import time

torch.multiprocessing.set_sharing_strategy("file_system")

batch_size = 64
controller_period = 0.05


simulation_dt = 0.01

def create_training_hyperparams()-> Dict:
    """
    create_hyperparams
    Description
        Creates a dictionary containing all hyperparameters used in
        Neural aCLBF training.
    """

    # Get initial conditions for the experiment
    start_x = torch.tensor(
        [
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            [-0.2, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, -1.0, 0.0, 0.0, 0.0, 0.0],
            [-0.2, -1.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    hyperparams_for_evaluation = {
        "batch_size": 64,
        "controller_period": 0.05,
        "start_x": start_x,
        "simulation_dt": 0.01,
        "nominal_scenario_obstacle_center": [1.0, 1.0, -1.0],
        "nominal_scenario_obstacle_width": 1.0,
        "Theta_lb": [0.5, 0.0, -0.5],
        "Theta_ub": [1.0, 0.5, 0.5],
        "clf_lambda": 1.0,
        # layer specifications
        "clbf_hidden_size": 64,
        "clbf_hidden_layers": 2,
        # Training parameters
        "max_epochs": 6,
        # Contour Experiment Parameters
        "contour_exp_x_index": 0,
        "contour_exp_theta_index": LoadSharingManipulator.P_X,
        # Random Seed Info
        "pt_manual_seed": 30,
        "np_manual_seed": 51,
        # Device
        "device": device,
        "sample_quotas": {"safe": 0.2, "unsafe": 0.2, "goal": 0.2},
    }

    return hyperparams_for_evaluation

def main(args):
    # Constants

    # Get hyperparameters for training
    t_hyper = create_training_hyperparams()

    # Set Constants
    torch.manual_seed(t_hyper["pt_manual_seed"])
    np.random.seed(t_hyper["np_manual_seed"])
    device = torch.device(t_hyper["device"])

    # Define the scenarios
    nominal_scenario = {
        "obstacle_center": t_hyper["nominal_scenario_obstacle_center"],
        "obstacle_width": t_hyper["nominal_scenario_obstacle_width"],
    }
    scenarios = [
        nominal_scenario,
        # {"m": 1.25, "L": 1.0, "b": 0.01},  # uncomment to add robustness
        # {"m": 1.0, "L": 1.25, "b": 0.01},
        # {"m": 1.25, "L": 1.25, "b": 0.01},
    ]

    # Define the range of possible goal region centers
    lb = t_hyper["Theta_lb"]
    ub = t_hyper["Theta_ub"]
    Theta = pc.box2poly(np.array([lb, ub]).T)
    print(Theta)

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
        (-np.pi / 4, np.pi / 4),# p_x
        (-1.0, 1.0),            # p_y
        (-np.pi / 4, np.pi / 4),# p_z
        (-1.0, 1.0),            # v_x
        (-1.0, 1.0),            # v_y
        (-1.0, 1.0),            # v_z
    ]
    data_module = EpisodicDataModuleAdaptive(
        dynamics_model,
        initial_conditions,
        trajectories_per_episode=1,
        trajectory_length=1,
        fixed_samples=10000,
        max_points=100000,
        val_split=0.1,
        batch_size=64,
        quotas=t_hyper["sample_quotas"],
    )

    # Define the experiment suite
    V_contour_experiment = AdaptiveCLFContourExperiment(
        "V_Contour",
        x_domain=[(-2.0, 2.0)],
        theta_domain=[(-2.0, 2.0)],
        n_grid=30,
        x_axis_index=LoadSharingManipulator.P_X,
        theta_axis_index=t_hyper["contour_exp_theta_index"],
        x_axis_label="$p_x$",
        theta_axis_label="$\\theta_" + str(t_hyper["contour_exp_theta_index"]) + "$", #"$\\dot{\\theta}$",
        plot_unsafe_region=False,
    )
    # rollout_experiment = RolloutStateSpaceExperiment(
    #     "Rollout",
    #     start_x,
    #     InvertedPendulum.THETA,
    #     "$\\theta$",
    #     InvertedPendulum.THETA_DOT,
    #     "$\\dot{\\theta}$",
    #     scenarios=scenarios,
    #     n_sims_per_start=1,
    #     t_sim=5.0,
    # )
    #experiment_suite = ExperimentSuite([V_contour_experiment, rollout_experiment])
    experiment_suite = ExperimentSuite([V_contour_experiment])

    # Initialize the controller
    aclbf_controller = NeuralaCLBFController(
        dynamics_model,
        scenarios,
        data_module,
        experiment_suite=experiment_suite,
        clbf_hidden_layers=2,
        clbf_hidden_size=64,
        clf_lambda=1.0,
        safe_level=1.0,
        controller_period=controller_period,
        clf_relaxation_penalty=1e2,
        num_init_epochs=5,
        epochs_per_episode=100,
        barrier=False,
    )
    aclbf_controller.to(device)

    # Initialize the logger and trainer
    tb_logger = pl_loggers.TensorBoardLogger(
        "logs/load_sharing_manipulator",
        name=f"commit_{current_git_hash()}",
    )
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=tb_logger,
        reload_dataloaders_every_epoch=True,
        max_epochs=t_hyper["max_epochs"],
    )

    # Train
    torch.autograd.set_detect_anomaly(True)
    training_time_start = time.time()
    trainer.fit(aclbf_controller)
    training_time_end = time.time()

    # Logging
    tb_logger.log_metrics({"pytorch random seed": t_hyper["pt_manual_seed"]})
    tb_logger.log_metrics({"numpy random seed": t_hyper["np_manual_seed"]})
    tb_logger.log_metrics({"training time": training_time_end - training_time_start})

    # Saving Data
    torch.save(
        aclbf_controller.V_nn,
        tb_logger.save_dir + "/" + tb_logger.name +
        "/version_" + str(tb_logger.version) + "/Vnn.pt"
    )

    # Record Hyperparameters in small pytorch format
    torch.save(
        t_hyper,
        tb_logger.save_dir + "/" + tb_logger.name +
        "/version_" + str(tb_logger.version) + "/hyperparams.pt"
    )

    # Save model
    torch.save(
        aclbf_controller.state_dict(),
        tb_logger.save_dir + "/" + tb_logger.name +
        "/version_" + str(tb_logger.version) + "/state_dict.pt"
    )

    torch.save(
        aclbf_controller,
        tb_logger.save_dir + "/" + tb_logger.name +
        "/version_" + str(tb_logger.version) + "/controller.pt"
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
