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

def create_hyperparam_struct(args)-> Dict:
    # Device declaration
    accelerator_name = "cpu"
    if torch.cuda.is_available():
        accelerator_name = "cuda"
    elif torch.backends.mps.is_available():
        torch.set_default_dtype(torch.float32)
        # accelerator_name = "mps"
        accelerator_name = "cpu"

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
    ).to(accelerator_name)

    #device = "mps" if torch.backends.mps.is_available() else "cpu"

    hyperparams_for_evaluation = {
        "batch_size": 64,
        "controller_period": 0.05,
        "start_x": start_x,
        "simulation_dt": 0.01,
        "nominal_scenario_wall_pos": -0.5,
        "Theta_lb": -2.5,
        "Theta_ub": -1.5,
        "clf_lambda": 1.0,
        # Training Parameters
        "sample_quotas": {"safe": 0.2, "unsafe": 0.2, "goal": 0.2},
        "use_oracle": args.use_oracle,
        "barrier": args.barrier,
        "safe_level": args.safe_level,
        # layer specifications
        "clbf_hidden_size": 64,
        "clbf_hidden_layers": 2,
        "max_epochs": args.max_epochs,
        # Random Seed Info
        "pt_manual_seed": args.pt_random_seed,
        "np_manual_seed": args.np_random_seed,
        # Device
        "accelerator": accelerator_name,
    }

    return hyperparams_for_evaluation

def main(args):

    hyperparams = create_hyperparam_struct(args)
    # Random Seed Setting
    torch.manual_seed(hyperparams["pt_manual_seed"])
    np.random.seed(hyperparams["np_manual_seed"])

    device = torch.device(hyperparams["accelerator"])

    batch_size = hyperparams["batch_size"]
    controller_period = hyperparams["controller_period"]

    start_x = hyperparams["start_x"]
    simulation_dt = hyperparams["simulation_dt"]

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
        device=hyperparams["accelerator"],
    )

    # Initialize the DataModule
    initial_conditions = [
        (1.0, 3.0),# p_x
    ]
    data_module = EpisodicDataModuleAdaptive(
        dynamics_model,
        initial_conditions,
        trajectories_per_episode=1,
        trajectory_length=1,
        fixed_samples=10000,
        max_points=100000,
        val_split=0.1,
        batch_size=batch_size,
        quotas=hyperparams["sample_quotas"],
        device=hyperparams["accelerator"],
        # quotas={"safe": 0.2, "unsafe": 0.2, "goal": 0.4},
    )

    # Define the experiment suite
    x_ub, x_lb = dynamics_model.state_limits
    V_contour_experiment = AdaptiveCLFContourExperiment(
        "V_Contour",
        n_grid=30,
        # X axis details
        x_domain=[(x_lb[ScalarCAPA2Demo.X_DEMO], x_ub[ScalarCAPA2Demo.X_DEMO])], #plotting domain
        x_axis_index=ScalarCAPA2Demo.X_DEMO,
        x_axis_label="$p_x$",
        # Theta axis details
        theta_axis_index=ScalarCAPA2Demo.P_DEMO,
        theta_domain=[(lb[0], ub[0])], # plotting domain for theta
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
    if args.checkpoint_path is None:
        aclbf_controller = NeuralaCLBFController(
            dynamics_model,
            scenarios,
            data_module,
            experiment_suite=experiment_suite,
            clbf_hidden_layers=hyperparams["clbf_hidden_layers"],
            clbf_hidden_size=hyperparams["clbf_hidden_size"],
            clf_lambda=hyperparams["clf_lambda"],
            safe_level=hyperparams["safe_level"],
            controller_period=controller_period,
            clf_relaxation_penalty=1e2,
            num_init_epochs=10,
            epochs_per_episode=100,
            barrier=hyperparams["barrier"],
            Gamma_factor=0.1,
            include_oracle_loss=hyperparams["use_oracle"],
        )
    else:

        aclbf_controller = NeuralaCLBFController.load_from_checkpoint(
            args.checkpoint_path,
        )

    # Initialize the logger and trainer
    tb_logger = pl_loggers.TensorBoardLogger(
        "logs/scalar_demo_capa2_system",
        name=f"commit_{current_git_hash()}",
    )
    # trainer = pl.Trainer.from_argparse_args(
    #     args,
    #     logger=tb_logger,
    #     reload_dataloaders_every_epoch=True,
    #     max_epochs=hyperparams["max_epochs"],
    # )
    trainer = pl.Trainer(
        logger=tb_logger,
        max_epochs=hyperparams["max_epochs"],
        # reload_dataloaders_every_n_epochs=1,
        # val_check_interval=1.0,
        # log_every_n_steps=1,
        accelerator=hyperparams["accelerator"],
    )

    # Train
    torch.autograd.set_detect_anomaly(True)
    trainer.fit(aclbf_controller)

    # End of Training Sequence
    # ========================

    # Logging
    tb_logger.log_metrics({"pytorch random seed": hyperparams["pt_manual_seed"]})
    tb_logger.log_metrics({"numpy random seed": hyperparams["np_manual_seed"]})

    # Saving Data
    torch.save(
        aclbf_controller.V_nn,
        tb_logger.save_dir + "/" + tb_logger.name +
        "/version_" + str(tb_logger.version) + "/Vnn.pt"
    )

    # for layer in aclbf_controller.V_nn:
    #     print(layer)
    #     if isinstance(layer, torch.nn.Linear):
    #         print(layer.weight)

    # Record Hyperparameters in small pytorch format
    torch.save(
        hyperparams,
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
    parser.add_argument(
        '--pt_random_seed', type=int, default=31,
        help='Integer used as PyTorch\'s random seed (default: 31)',
    ),
    parser.add_argument(
        '--np_random_seed', type=int, default=30,
        help='Integer used as Numpy\'s random seed (default: 30)'
    )
    # parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=6)
    parser.add_argument(
        '--checkpoint_path', type=str, default=None,
        help='Path to checkpoint to load from (default: None)',
    )
    parser.add_argument(
        '--use_oracle', type=bool, default=False,
        help='Whether to use the oracle loss in training(default: False)',
    )
    parser.add_argument(
        '--barrier', type=bool, default=False,
        help='Whether to use the barrier loss in training(default: False)',
    )
    parser.add_argument(
        '--safe_level', type=float, default=0.5,
        help='Safe level for the CLBF (default: 0.5)',
    )
    args = parser.parse_args()

    main(args)
