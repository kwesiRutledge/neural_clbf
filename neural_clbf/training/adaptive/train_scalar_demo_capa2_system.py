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

from neural_clbf.controllers import NeuralaCLBFController
from neural_clbf.datamodules import (
    EpisodicDataModule, EpisodicDataModuleAdaptive
)
from neural_clbf.systems.adaptive import ScalarCAPA2Demo
from neural_clbf.experiments import (
    ExperimentSuite,
    CLFContourExperiment,
    RolloutStateSpaceExperiment,
    AdaptiveCLFContourExperiment
)
from neural_clbf.training.utils import current_git_hash
import polytope as pc

torch.multiprocessing.set_sharing_strategy("file_system")

batch_size = 64
controller_period = 0.05

start_x = torch.tensor(
    [
        [0.5, 1.5],
    ]
)
simulation_dt = 0.01


def main(args):
    # Define the scenarios
    nominal_scenario = {"wall_position": -0.5}
    scenarios = [
        nominal_scenario,
    ]

    # Define the range of possible uncertain parameters
    lb = [0.5]
    ub = [0.8]
    Theta = pc.box2poly(np.array([lb, ub]).T)
    print(Theta)

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
        (1.0, 3.0),# p_x
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
        domain=[(-2.0, 2.0), (-2.0, 2.0)], #plotting domain
        n_grid=30,
        x_axis_index=ScalarCAPA2Demo.X_DEMO,
        theta_axis_index=ScalarCAPA2Demo.P_DEMO,
        x_axis_label="$p_x$",
        theta_axis_label="$\theta$", #"$\\dot{\\theta}$",
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
    clbf_controller = NeuralaCLBFController(
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

    # Initialize the logger and trainer
    tb_logger = pl_loggers.TensorBoardLogger(
        "logs/scalar_demo_capa2_system",
        name=f"commit_{current_git_hash()}",
    )
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=tb_logger,
        reload_dataloaders_every_epoch=True,
        max_epochs=51,
    )

    # Train
    torch.autograd.set_detect_anomaly(True)
    trainer.fit(clbf_controller)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
