"""
train_pusher_slider_force_input.py
Description:
    This script trains an aCLBF for the pusher-slider system defined in
    systems/adaptive_w_scenarios/load_sharing_manipulator.py.
    Uses the certainty-equivalent controller from NeuralaCLBFController5.
"""

from argparse import ArgumentParser

import torch
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import numpy as np

from neural_clbf.controllers.adaptive_with_observed_parameters import (
    NeuralaCLBFControllerV5,
)
from neural_clbf.datamodules.adaptive_w_scenarios import (
    EpisodicDataModuleAdaptiveWScenarios,
)
from neural_clbf.systems.adaptive_w_scenarios import AdaptivePusherSliderStickingForceInput_NObstacles as PusherSlider
from neural_clbf.experiments import (
    ExperimentSuite,
    CLFContourExperiment, AdaptiveCLFContourExperiment,
    RolloutStateSpaceExperiment, RolloutStateParameterSpaceExperiment,
    # RolloutStateParameterSpaceExperimentMultiple,
)
from neural_clbf.experiments.adaptive_w_observed_parameters import (
    AdaptiveCLFContourExperiment3, RolloutStateParameterSpaceExperimentMultiple, aCLFCountourExperiment_StateSlices3,
)
from neural_clbf.experiments.adaptive import (
    aCLFCountourExperiment_StateSlices
)
from neural_clbf.experiments.adaptive_w_uncertainty import (
    RolloutStateParameterSpaceExperimentMultipleUncertainty, AdaptiveCLFContourExperimentUncertainty,
    aCLFCountourExperiment_StateSlices_Uncertainty,
)
from neural_clbf.training.utils import (
    current_git_hash, initialize_training_arg_parser
)
import polytope as pc

from typing import Dict

import time
import datetime

torch.multiprocessing.set_sharing_strategy("file_system")

def create_training_hyperparams(args)-> Dict:
    """
    create_hyperparams
    Description
        Creates a dictionary containing all hyperparameters used in
        Neural aCLBF training.
    """

    # Get initial conditions for the experiment
    start_x = torch.tensor([
        [-0.5, -0.5, 0.0],
        [-0.4, -0.5, 0.0],
        [-0.5, -0.4, 0.0],
    ])

    #device = "mps" if torch.backends.mps.is_available() else "cpu"
    accelerator_name = "cpu"
    if torch.cuda.is_available():
        accelerator_name = "cuda"
    elif torch.backends.mps.is_available():
        torch.set_default_dtype(torch.float32)
        accelerator_name = "mps"
        # accelerator_name = "cpu"

    # Create the nominal scenario
    nominal_scenario = {
        "obstacle_0_center_x": 0.0,
        "obstacle_0_center_y": 0.0,
        "obstacle_1_center_x": 0.0,
        "obstacle_1_center_y": -0.5,
        "goal_x": 0.5,
        "goal_y": 0.2,
    }

    s_width = 0.09

    # Create default number of maximum epochs
    hyperparams_for_evaluation = {
        "batch_size": 64,
        "controller_period": 0.1,
        "start_x": start_x,
        "simulation_dt": 0.025,
        "nominal_scenario": nominal_scenario,
        "Theta_lb": [-0.03 + s_width/2.0, -0.03],
        "Theta_ub": [ 0.03 + s_width/2.0, 0.03],
        "scenario_lb": [-0.1, -0.1, 0.2, 0.2],
        "scenario_ub": [ 0.1,  0.1, 0.3, 0.3],
        # "clf_lambda": args.clf_lambda,
        "Gamma_factor": 0.01,
        # "safe_level": args.safe_level,
        # layer specifications
        "clbf_hidden_size": 64,
        "clbf_hidden_layers": 2,
        # Training parameters
        # "max_epochs": args.max_epochs,
        "trajectories_per_episode": 200, #1000,
        "trajectory_length": 30,
        "n_fixed_samples": 4000, # 20000,
        "num_init_epochs": 15,
        "goal_loss_weight": 1e2,
        "safe_loss_weight": 1e2,
        "unsafe_loss_weight": 1e2,
        # "max_iters_cvxpylayer": int(1e7),  # default = 50000000 = 50 million
        # "include_oracle_loss": True,
        # "include_estimation_error_loss": args.use_estimation_error_loss,
        # "barrier": args.barrier,
        "gradient_clip_val": args.gradient_clip_val,
        "checkpoint_path": args.checkpoint_path,
        # Contour Experiment Parameters
        "contour_exp_x_index": 0,
        "contour_exp_theta_index": PusherSlider.S_X,
        # Rollout Experiment Parameters
        "rollout_experiment_horizon": 5.0,
        # Random Seed Info
        "pt_manual_seed": args.pt_random_seed,
        "np_manual_seed": args.np_random_seed,
        # Device
        "accelerator": accelerator_name,
        "sample_quotas": {"safe": 0.2, "unsafe": 0.2, "goal": 0.2},
        # Observation error
        "observation_error": 1e-2,
    }

    for k in args.__dict__:
        hyperparams_for_evaluation[k] = args.__dict__[k]

    # Set default datatype
    # torch.set_default_dtype(torch.float64)

    return hyperparams_for_evaluation

def main(args):
    # Constants

    # Get hyperparameters for training
    t_hyper = create_training_hyperparams(args)

    # Set Constants
    torch.manual_seed(t_hyper["pt_manual_seed"])
    np.random.seed(t_hyper["np_manual_seed"])

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

    P_scenario = pc.box2poly(
        np.array([t_hyper["scenario_lb"], t_hyper["scenario_ub"]]).T,
    )

    # Define the dynamics model
    dynamics_model = PusherSlider(
        t_hyper["nominal_scenario"],
        Theta,
        P_scenario=P_scenario,
        dt=t_hyper["simulation_dt"],
        controller_dt=t_hyper["controller_period"],
        device=t_hyper["accelerator"],
    )

    # Initialize the DataModule
    initial_conditions = [
        (-0.7, -0.4),  # s_x
        (-0.8, -0.4),  # s_y
        (0.0, np.pi / 2),  # s_theta
    ]
    data_module = EpisodicDataModuleAdaptiveWScenarios(
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
    )

    # Define the experiment suite
    lb_Vcontour = lb[t_hyper["contour_exp_theta_index"]]
    ub_Vcontour = ub[t_hyper["contour_exp_theta_index"]]
    theta_range_Vcontour = ub_Vcontour - lb_Vcontour
    x_ub, x_lb = dynamics_model.state_limits

    V_contour_experiment = AdaptiveCLFContourExperiment3(
        "V_Contour",
        x_domain=[(x_lb[PusherSlider.S_X], x_ub[PusherSlider.S_X])],
        theta_domain=[(lb_Vcontour-0.2*theta_range_Vcontour, ub_Vcontour+0.2*theta_range_Vcontour)],
        n_grid=30,
        x_axis_index=PusherSlider.S_X,
        theta_axis_index=t_hyper["contour_exp_theta_index"],
        x_axis_label="$p_1$",
        theta_axis_label="$\\theta_" + str(t_hyper["contour_exp_theta_index"]) + "$", #"$\\dot{\\theta}$",
        plot_unsafe_region=False,
    )
    rollout_experiment = RolloutStateParameterSpaceExperiment(
        "Rollout",
        t_hyper["start_x"],
        PusherSlider.S_X,
        "$r_1$",
        PusherSlider.C_X,
        "$\\theta_1 (r_1^{(d)})$",
        scenarios=scenarios,
        n_sims_per_start=1,
        t_sim=t_hyper["rollout_experiment_horizon"],
    )
    rollout_experiment2 = RolloutStateParameterSpaceExperimentMultiple(
        "Rollout (Multiple Slices)",
        t_hyper["start_x"],
        [PusherSlider.S_X, PusherSlider.S_Y, PusherSlider.S_X],
        ["$r_1$", "$v_1$", "$r_2$"],
        [PusherSlider.C_X, PusherSlider.C_X, PusherSlider.C_Y],
        ["$\\theta_1 (c_x)$", "$\\theta_1 (c_x)$", "$\\theta_1 (c_y)$"],
        n_sims_per_start=1,
        t_sim=t_hyper["rollout_experiment_horizon"],
    )
    V_contour_experiment3 = aCLFCountourExperiment_StateSlices3(
        "V_Contour (state slices only)",
        x_domain=[
            (x_lb[0], x_ub[0]),
            (x_lb[1], x_ub[1]),
        ],  # plotting domain
        n_grid=50,
        x_axis_index=PusherSlider.S_X,
        y_axis_index=PusherSlider.S_Y,
        x_axis_label="$p_1$",
        y_axis_label="$p_2$",
        default_param_estimate=torch.tensor([dynamics_model.s_width, 0.0]).reshape((PusherSlider.N_PARAMETERS, 1)),
    )
    experiment_suite = ExperimentSuite([V_contour_experiment, rollout_experiment2, V_contour_experiment3])
    #experiment_suite = ExperimentSuite([V_contour_experiment])

    # Initialize the controller
    if (args.checkpoint_path is None) and (t_hyper["saved_Vnn_subpath"] is None):
        aclbf_controller = NeuralaCLBFControllerV5(
            dynamics_model,
            data_module,
            experiment_suite=experiment_suite,
            clbf_hidden_layers=2,
            clbf_hidden_size=64,
            clf_lambda=t_hyper["clf_lambda"],
            safe_level=t_hyper["safe_level"],
            controller_period=t_hyper["controller_period"],
            clf_relaxation_penalty=t_hyper["clf_relaxation_penalty"],
            num_init_epochs=t_hyper["num_init_epochs"],
            learn_shape_epochs=t_hyper["learn_shape_epochs"],
            learn_boundary_epochs=t_hyper["learn_boundary_epochs"],
            epochs_per_episode=100,
            barrier=t_hyper["barrier"],
            Gamma_factor=t_hyper["Gamma_factor"],
            include_oracle_loss=t_hyper["include_oracle_loss"],
            include_estimation_error_loss=t_hyper["include_estimation_error_loss"],
            include_radially_unbounded_loss1=t_hyper["include_radially_unbounded_loss1"],
            include_radially_unbounded_loss2=t_hyper["include_radially_unbounded_loss2"],
            max_iters_cvxpylayer=t_hyper["max_iters_cvxpylayer"],
        )
    elif args.checkpoint_path is not None:
        aclbf_controller = NeuralaCLBFControllerV5.load_from_checkpoint(
            args.checkpoint_path,
        )
        print(aclbf_controller)
        print(f"Loaded controller from {args.checkpoint_path}", "green")
    elif t_hyper["saved_Vnn_subpath"] is not None:
        pusher_slider_log_file_dir = "logs/pusher_slider_sticking_force_input/"
        # Load V_nn
        saved_Vnn = torch.load(
            pusher_slider_log_file_dir + t_hyper["saved_Vnn_subpath"] + "/Vnn.pt",
            map_location=torch.device(t_hyper["device"]),
        )







    # Initialize the logger and trainer
    t = datetime.datetime.now()
    tb_logger = pl_loggers.TensorBoardLogger(
        "logs/pusher_slider_sticking_force_input",
        name=f"commit_{current_git_hash()}",
        version=f"version_{t.strftime('%m%d%Y_%H_%M_%S')}",
    )
    # trainer = pl.Trainer.from_argparse_args(
    #     args,
    #     logger=tb_logger,
    #     reload_dataloaders_every_epoch=True,
    #     max_epochs=t_hyper["max_epochs"],
    # )

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
        "/" + str(tb_logger.version) + "/Vnn.pt"
    )

    # Record Hyperparameters in small pytorch format
    torch.save(
        t_hyper,
        tb_logger.save_dir + "/" + tb_logger.name +
        "/" + str(tb_logger.version) + "/hyperparams.pt"
    )

    # Save model
    torch.save(
        aclbf_controller.state_dict(),
        tb_logger.save_dir + "/" + tb_logger.name +
        "/" + str(tb_logger.version) + "/state_dict.pt"
    )

    torch.save(
        aclbf_controller,
        tb_logger.save_dir + "/" + tb_logger.name +
        "/" + str(tb_logger.version) + "/controller.pt"
    )

if __name__ == "__main__":
    parser = ArgumentParser(
        description="This script trains the PusherSlider controller based on adaptive control Lyapunov function principles.",
    )
    initialize_training_arg_parser(parser)
    args = parser.parse_args()
    main(args)
