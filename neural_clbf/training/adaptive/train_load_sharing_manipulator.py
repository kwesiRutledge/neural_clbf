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
    RolloutStateSpaceExperiment, RolloutStateParameterSpaceExperiment,
    RolloutStateParameterSpaceExperimentMultiple,
)
from neural_clbf.training.utils import (
    current_git_hash, initialize_training_arg_parser
)

import polytope as pc

from typing import Dict

import time

torch.multiprocessing.set_sharing_strategy("file_system")

def create_training_hyperparams(args)-> Dict:
    """
    create_hyperparams
    Description
        Creates a dictionary containing all hyperparameters used in
        Neural aCLBF training.
    """

    #device = "mps" if torch.backends.mps.is_available() else "cpu"
    accelerator = "cpu"
    if torch.cuda.is_available():
        accelerator = "cuda"
    elif torch.backends.mps.is_available():
        torch.set_default_dtype(torch.float32)
        # accelerator = "mps"
        accelerator = "cpu"

    # Get initial conditions for the experiment
    start_x = torch.tensor(
        [
            [0.3, -0.3, 0.4, 0.0, 0.0, 0.0],
            [0.25, -0.25, 0.4, 0.1, 0.0, 0.0],
            [0.35, -0.25, 0.3, 0.0, 0.0, 0.0],
            [0.35, -0.35, 0.3, 0.0, 0.0, 0.0],
            [0.25, -0.35, 0.3, 0.0, 0.0, 0.0],
        ]
    ).to(accelerator)

    nominal_scenario = {
        "obstacle_center_x": 0.2,
        "obstacle_center_y": 0.1,
        "obstacle_center_z": 0.3,
        "obstacle_width": 0.2,
    }

    hyperparams_for_evaluation = {
        "batch_size": 32,
        "controller_period": 0.1,
        "start_x": start_x,
        "simulation_dt": 0.025,
        "nominal_scenario": nominal_scenario,
        "Theta_lb": [-0.15, 0.4, 0.1],
        "Theta_ub": [0.15, 0.45, 0.3],
        "clf_lambda": args.clf_lambda,
        "Gamma_factor": 0.1,
        "safe_level": 1.0,
        # layer specifications
        "clbf_hidden_size": 64,
        "clbf_hidden_layers": 2,
        # Training parameters
        #"max_epochs": args.max_epochs,
        "n_fixed_samples": 30000,
        "trajectories_per_episode": 500,
        "trajectory_length": 20,
        "accelerator": accelerator,
        "num_init_epochs": 20,
        #"use_oracle_loss": args.use_oracle_loss,
        #"barrier": args.barrier,
        #"gradient_clip_val": args.gradient_clip_val,
        # "gradient_clip_val": args.gradient_clip_val,
        # "checkpoint_path": args.checkpoint_path,
        # Contour Experiment Parameters
        "contour_exp_x_index": 0,
        "contour_exp_theta_index": LoadSharingManipulator.P_X,
        # Rollout Experiment Parameters
        "rollout_experiment_horizon": 15.0,
        # Random Seed Info
        "pt_manual_seed": args.pt_random_seed,
        "np_manual_seed": args.np_random_seed,
        # Device
        "sample_quotas": {"safe": 0.2, "unsafe": 0.2, "goal": 0.2},
    }

    for k in args.__dict__:
        hyperparams_for_evaluation[k] = args.__dict__[k]

    return hyperparams_for_evaluation

def main(args):
    # Constants

    # Get hyperparameters for training
    t_hyper = create_training_hyperparams(args)

    # Set Constants
    torch.manual_seed(t_hyper["pt_manual_seed"])
    np.random.seed(t_hyper["np_manual_seed"])
    device = torch.device(t_hyper["accelerator"])

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
    dynamics_model = LoadSharingManipulator(
        t_hyper["nominal_scenario"],
        Theta,
        dt=t_hyper["simulation_dt"],
        controller_dt=t_hyper["controller_period"],
        scenarios=scenarios,
        device=t_hyper["accelerator"],
    )

    # Initialize the DataModule
    initial_conditions = [
        (-0.4, 0.4),# p_x
        (-0.4, 0.4),             # p_y
        (0.0, 0.7),             # p_z
        (-0.5, 0.5),            # v_x
        (-0.5, 0.5),            # v_y
        (-0.5, 0.5),            # v_z
    ]
    data_module = EpisodicDataModuleAdaptive(
        dynamics_model,
        initial_conditions,
        trajectories_per_episode=t_hyper["trajectories_per_episode"],
        trajectory_length=t_hyper["trajectory_length"],
        fixed_samples=t_hyper["n_fixed_samples"],
        max_points=100000,
        val_split=0.1,
        batch_size=t_hyper["batch_size"],
        quotas=t_hyper["sample_quotas"],
        device=t_hyper["accelerator"],
        num_workers=t_hyper["num_cpu_cores"],
    )

    # Define the experiment suite
    lb_Vcontour = lb[t_hyper["contour_exp_theta_index"]]
    ub_Vcontour = ub[t_hyper["contour_exp_theta_index"]]
    theta_range_Vcontour = ub_Vcontour - lb_Vcontour
    V_contour_experiment = AdaptiveCLFContourExperiment(
        "V_Contour",
        x_domain=[
            (dynamics_model.state_limits[1][LoadSharingManipulator.P_X],
             dynamics_model.state_limits[0][LoadSharingManipulator.P_X]),
        ],
        theta_domain=[(lb_Vcontour-0.2*theta_range_Vcontour, ub_Vcontour+0.2*theta_range_Vcontour)],
        n_grid=30,
        x_axis_index=LoadSharingManipulator.P_X,
        theta_axis_index=t_hyper["contour_exp_theta_index"],
        x_axis_label="$r_1$",
        theta_axis_label="$\\theta_" + str(t_hyper["contour_exp_theta_index"]) + "$", #"$\\dot{\\theta}$",
        plot_unsafe_region=False,
    )
    rollout_experiment = RolloutStateParameterSpaceExperiment(
        "Rollout",
        t_hyper["start_x"],
        LoadSharingManipulator.P_X,
        "$r_1$",
        LoadSharingManipulator.P_X_DES,
        "$\\theta_1 (r_1^{(d)})$",
        scenarios=scenarios,
        n_sims_per_start=1,
        t_sim=t_hyper["rollout_experiment_horizon"],
    )
    rollout_experiment2 = RolloutStateParameterSpaceExperimentMultiple(
        "Rollout (Multiple Slices)",
        t_hyper["start_x"],
        [LoadSharingManipulator.P_X, LoadSharingManipulator.V_X, LoadSharingManipulator.P_Y],
        ["$r_1$", "$v_1$", "$r_2$"],
        [LoadSharingManipulator.P_X_DES, LoadSharingManipulator.P_X_DES, LoadSharingManipulator.P_Y],
        ["$\\theta_1 (r_1^{(d)})$", "$\\theta_1 (r_1^{(d)})$", "$\\theta_1 (r_2^{(d)})$"],
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
        clbf_hidden_layers=t_hyper["clbf_hidden_layers"],
        clbf_hidden_size=t_hyper["clbf_hidden_size"],
        clf_lambda=t_hyper["clf_lambda"],
        safe_level=t_hyper["safe_level"],
        controller_period=t_hyper["controller_period"],
        clf_relaxation_penalty=1e2,
        num_init_epochs=t_hyper["num_init_epochs"],
        epochs_per_episode=100,
        barrier=t_hyper["barrier"],
        Gamma_factor=t_hyper["Gamma_factor"],
        include_oracle_loss=t_hyper["include_oracle_loss"],
    )

    # Initialize the logger and trainer
    tb_logger = pl_loggers.TensorBoardLogger(
        "logs/load_sharing_manipulator",
        name=f"commit_{current_git_hash()}",
    )

    if t_hyper["number_of_gpus"] <= 1:
        print("Using CPU or Single GPU")
        trainer = pl.Trainer(
            logger=tb_logger,
            # reload_dataloaders_every_epoch=True,
            max_epochs=t_hyper["max_epochs"],
            accelerator=t_hyper["accelerator"],
            gradient_clip_val=t_hyper["gradient_clip_val"],
        )
    else:
        print("Using DDP")
        trainer = pl.Trainer(
            logger=tb_logger,
            max_epochs=t_hyper["max_epochs"],
            accelerator=t_hyper["accelerator"],
            gradient_clip_val=t_hyper["gradient_clip_val"],
            devices=t_hyper["number_of_gpus"],
            strategy="ddp",
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
    tb_logger.log_metrics({"gamma factor": t_hyper["Gamma_factor"]})

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
    parser = ArgumentParser(
        description="Train a CLBF Controller to perform safe control of the load sharing manipulator task."
    )
    initialize_training_arg_parser(parser)
    args = parser.parse_args()

    main(args)
