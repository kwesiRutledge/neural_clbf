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
from neural_clbf.systems.adaptive import TumblingTarget, TumblingTarget2
from neural_clbf.experiments import (
    ExperimentSuite,
    CLFContourExperiment, AdaptiveCLFContourExperiment,
    RolloutStateSpaceExperiment, RolloutStateParameterSpaceExperiment,
    RolloutStateParameterSpaceExperimentMultiple,
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
            [8.0, 5.0, 0.5, -0.1, 0.5, 0.5, 0.5],
            [5.0, 5.0, 0.0, 0.1, 0.0, 0.0, 0.0],
            [5.2, -10.0, 0.0, -0.1, 0.0, 0.0, 0.0],
            [-0.2, -1.0, 0.0, 0.1, 0.0, 0.0, 0.0],
        ]
    )

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    nominal_scenario = {
        "obstacle_center_x": 5.0,
        "obstacle_center_y": -1.0,
        "obstacle_center_z": 0.0,
        "obstacle_width": 2.0,
    }

    hyperparams_for_evaluation = {
        "batch_size": 128,
        "controller_period": 0.05,
        "start_x": start_x,
        "simulation_dt": 0.01,
        "nominal_scenario": nominal_scenario,
        "Theta_lb": [-0.01],
        "Theta_ub": [0.03],
        "clf_lambda": 1.0,
        "Gamma_factor": 0.1,
        # layer specifications
        "clbf_hidden_size": 64,
        "clbf_hidden_layers": 2,
        # Training parameters
        "max_epochs": 16,
        "trajectories_per_episode": 100,
        "trajectory_length": 100,
        "n_fixed_samples": 50000,
        # Contour Experiment Parameters
        "contour_exp_x_index": 0,
        "contour_exp_theta_index": TumblingTarget.P_X,
        # Rollout Experiment Parameters
        "rollout_experiment_horizon": 15.0,
        # Random Seed Info
        "pt_manual_seed": 61,
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
    scenarios = [
        t_hyper["nominal_scenario"],
        # {"m": 1.25, "L": 1.0, "b": 0.01},  # uncomment to add robustness
        # {"m": 1.0, "L": 1.25, "b": 0.01},
        # {"m": 1.25, "L": 1.25, "b": 0.01},
    ]

    # Define the range of possible goal region centers
    lb = t_hyper["Theta_lb"]
    ub = t_hyper["Theta_ub"]
    Theta = pc.box2poly(np.array([lb, ub]).T)

    # Define the dynamics model
    dynamics_model = TumblingTarget2(
        t_hyper["nominal_scenario"],
        Theta,
        dt=simulation_dt,
        controller_dt=controller_period,
        scenarios=scenarios,
    )

    # Initialize the DataModule
    initial_conditions = [
        (-np.pi / 4, np.pi / 4),# p_x
        (-1.0, 1.0),            # p_y
        (-1.0, 1.0),            # p_z
        (-np.pi / 4, np.pi / 4),# phi_des
        (-1.0, 1.0),            # v_x
        (-1.0, 1.0),            # v_y
        (-1.0, 1.0),            # v_z
    ]
    data_module = EpisodicDataModuleAdaptive(
        dynamics_model,
        initial_conditions,
        trajectories_per_episode=t_hyper["trajectories_per_episode"],
        trajectory_length=t_hyper["trajectory_length"],
        fixed_samples=t_hyper["n_fixed_samples"],
        max_points=100000,
        val_split=0.1,
        batch_size=64,
        quotas=t_hyper["sample_quotas"],
    )

    # Define the experiment suite
    Theta_range = t_hyper["Theta_ub"][0] - t_hyper["Theta_lb"][0]
    V_contour_experiment = AdaptiveCLFContourExperiment(
        "V_Contour",
        x_domain=[(-2.0, 2.0)],
        theta_domain=[(t_hyper["Theta_lb"][0]-0.5*Theta_range, t_hyper["Theta_ub"][0]+0.5*Theta_range)],
        n_grid=30,
        x_axis_index=TumblingTarget.P_X,
        theta_axis_index=t_hyper["contour_exp_theta_index"],
        x_axis_label="$p_x$",
        theta_axis_label="$\\theta_" + str(t_hyper["contour_exp_theta_index"]) + "$", #"$\\dot{\\theta}$",
        plot_unsafe_region=False,
    )
    rollout_experiment = RolloutStateParameterSpaceExperiment(
        "Rollout",
        t_hyper["start_x"],
        TumblingTarget.P_X,
        "$r_1$",
        TumblingTarget.OMEGA_0,
        "$\\theta_1 (r_1^{(d)})$",
        scenarios=scenarios,
        n_sims_per_start=1,
        t_sim=t_hyper["rollout_experiment_horizon"],
    )
    rollout_experiment2 = RolloutStateParameterSpaceExperimentMultiple(
        "Rollout",
        t_hyper["start_x"],
        [TumblingTarget.P_X, TumblingTarget.V_X, TumblingTarget.P_Y],
        ["$r_1$", "$v_1$", "$r_2$"],
        [TumblingTarget.OMEGA_0, TumblingTarget.OMEGA_0, TumblingTarget.OMEGA_0],
        ["$\\omega_0 $", "$\\omega_0$", "$\\omega_0$"],
        scenarios=scenarios,
        n_sims_per_start=1,
        t_sim=t_hyper["rollout_experiment_horizon"],
    )
    experiment_suite = ExperimentSuite([V_contour_experiment, rollout_experiment2])
    #experiment_suite = ExperimentSuite([V_contour_experiment])

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
        Gamma_factor=t_hyper["Gamma_factor"],
        device=t_hyper["device"],
    )
    aclbf_controller.to(device)

    # Initialize the logger and trainer
    tb_logger = pl_loggers.TensorBoardLogger(
        "logs/tumbling_target",
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
