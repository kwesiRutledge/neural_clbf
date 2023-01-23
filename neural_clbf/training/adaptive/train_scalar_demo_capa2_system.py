"""
train_load_sharing_manipulator.py
Description:
    This script trains an aCLBF for the load sharing manipulator system defined in systems/load_sharing_manipulator.py.
"""
from typing import Dict

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
    AdaptiveCLFContourExperiment,
    RolloutStateParameterSpaceExperiment
)
from neural_clbf.training.utils import current_git_hash
import polytope as pc

torch.multiprocessing.set_sharing_strategy("file_system")



def create_hyperparam_struct()-> Dict:

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

    hyperparams_for_evaluation = {
        "batch_size": 64,
        "controller_period": 0.05,
        "start_x": start_x,
        "simulation_dt": 0.01,
        "nominal_scenario_wall_pos": -0.5,
        "Theta_lb": -2.5,
        "Theta_ub": -1.5,
        "clf_lambda": 1.0,
        # layer specifications
        "clbf_hidden_size": 64,
        "clbf_hidden_layers": 2,
        "max_epochs": 10,
    }

    return hyperparams_for_evaluation

def main(args):
    # Random Seed
    pt_manual_seed = 30
    torch.manual_seed(pt_manual_seed)
    np_manual_seed = pt_manual_seed
    np.random.seed(np_manual_seed)

    hyperparams_for_evaluation = create_hyperparam_struct()

    batch_size = hyperparams_for_evaluation["batch_size"]
    controller_period = hyperparams_for_evaluation["controller_period"]

    start_x = hyperparams_for_evaluation["start_x"]
    simulation_dt = hyperparams_for_evaluation["simulation_dt"]

    # Define the scenarios
    wall_pos = hyperparams_for_evaluation["nominal_scenario_wall_pos"]
    nominal_scenario = {"wall_position": wall_pos}
    scenarios = [
        nominal_scenario,
    ]

    # Define the range of possible uncertain parameters
    lb = [hyperparams_for_evaluation["Theta_lb"]]
    ub = [hyperparams_for_evaluation["Theta_ub"]]
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
        batch_size=batch_size,
        # quotas={"safe": 0.2, "unsafe": 0.2, "goal": 0.4},
    )

    # Define the experiment suite
    V_contour_experiment = AdaptiveCLFContourExperiment(
        "V_Contour",
        x_domain=[(-2.0, 2.0)], #plotting domain
        n_grid=30,
        x_axis_index=ScalarCAPA2Demo.X_DEMO,
        theta_axis_index=ScalarCAPA2Demo.P_DEMO,
        x_axis_label="$p_x$",
        theta_axis_label="$\\theta$", #"$\\dot{\\theta}$",
        plot_unsafe_region=False,
    )
    rollout_experiment = RolloutStateParameterSpaceExperiment(
        "Rollout",
        start_x,
        ScalarCAPA2Demo.X_DEMO,
        "$x$",
        ScalarCAPA2Demo.P_DEMO,
        "$\\theta$",
        scenarios=scenarios,
        n_sims_per_start=1,
        t_sim=5.0,
    )
    experiment_suite = ExperimentSuite([V_contour_experiment, rollout_experiment])
    #experiment_suite = ExperimentSuite([V_contour_experiment])

    # Initialize the controller
    clbf_controller = NeuralaCLBFController(
        dynamics_model,
        scenarios,
        data_module,
        experiment_suite=experiment_suite,
        clbf_hidden_layers=hyperparams_for_evaluation["clbf_hidden_layers"],
        clbf_hidden_size=hyperparams_for_evaluation["clbf_hidden_size"],
        clf_lambda=hyperparams_for_evaluation["clf_lambda"],
        safe_level=0.5,
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
        max_epochs=hyperparams_for_evaluation["max_epochs"],
    )

    # Train
    torch.autograd.set_detect_anomaly(True)
    trainer.fit(clbf_controller)

    # End of Training Sequence
    # ========================

    # Logging
    tb_logger.log_metrics({"pytorch random seed": pt_manual_seed})
    tb_logger.log_metrics({"numpy random seed": np_manual_seed})

    # Saving Data
    torch.save(
        clbf_controller.V_nn,
        tb_logger.save_dir + "/" + tb_logger.name +
        "/version_" + str(tb_logger.version) + "/Vnn.pt"
    )

    for layer in clbf_controller.V_nn:
        print(layer)
        if isinstance(layer, torch.nn.Linear):
            print(layer.weight)

    # Record Hyperparameters in small pytorch format
    torch.save(
        hyperparams_for_evaluation,
        tb_logger.save_dir + "/" + tb_logger.name +
        "/version_" + str(tb_logger.version) + "/hyperparams.pt"
    )

    # Save model
    torch.save(
        clbf_controller.state_dict(),
        tb_logger.save_dir + "/" + tb_logger.name +
        "/version_" + str(tb_logger.version) + "/state_dict.pt"
    )

    torch.save(
        clbf_controller,
        tb_logger.save_dir + "/" + tb_logger.name +
        "/version_" + str(tb_logger.version) + "/controller.pt"
    )

if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
