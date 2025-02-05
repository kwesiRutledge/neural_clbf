"""
test_episodic_datamodule_adaptive.py
Description:
    Tests the episodic datamodule for adaptive systems.
"""

import unittest
import numpy as np
import polytope as pc

from neural_clbf.datamodules import EpisodicDataModuleAdaptive
from neural_clbf.systems.adaptive import ScalarCAPA2Demo
from neural_clbf.experiments import ExperimentSuite
from neural_clbf.experiments.adaptive import (
    AdaptiveCLFContourExperiment
)
from neural_clbf.controllers import NeuralaCLBFController

class TestEpisodicDataModuleAdaptive(unittest.TestCase):
    def test_episodic_datamodule_adaptive_add_data1(self):
        """
        test_episodic_datamodule_adaptive_add_data1
        Description:
            This test verifies that the episodic datamodule for adaptive systems
            add_data method works as expected. Calling it for the scalar system.
        """

        # Constants
        wall_pos = -0.5
        nominal_scenario = {"wall_position": wall_pos}
        controller_period = 0.05
        simulation_dt = 0.01

        num_sim_steps = 10

        hyperparams = {
            "batch_size": 32,
            "sample_quotas": {"safe": 0.2, "unsafe": 0.2, "goal": 0.2},
            "accelerator": "cpu",
        }

        # Define the system
        # Define the range of possible uncertain parameters
        lb = [-2.5]
        ub = [-1.5]
        Theta = pc.box2poly(np.array([lb, ub]).T)

        # Define the dynamics model
        dynamics_model = ScalarCAPA2Demo(
            nominal_scenario,
            Theta,
            dt=simulation_dt,
            controller_dt=controller_period,
            scenarios=[nominal_scenario],
            device="cpu",
        )

        # Initialize the DataModule
        initial_conditions = [
            (1.0, 3.0),  # p_x
        ]
        data_module = EpisodicDataModuleAdaptive(
            dynamics_model,
            initial_conditions,
            trajectories_per_episode=1,
            trajectory_length=1,
            fixed_samples=10000,
            max_points=100000,
            val_split=0.1,
            batch_size=hyperparams["batch_size"],
            quotas=hyperparams["sample_quotas"],
            device=hyperparams["accelerator"],
        )

        _, hyperparams, experiment_suite = self.define_datamodule_and_params_for_scalar(
            dynamics_model,
            lb, ub,
        )
        aclbf_controller = NeuralaCLBFController(
            dynamics_model,
            [nominal_scenario],
            data_module,
            experiment_suite=experiment_suite,
            clbf_hidden_layers=2,
            clbf_hidden_size=64,
            clf_lambda=0.1,
            safe_level=0.5,
            controller_period=controller_period,
            clf_relaxation_penalty=1e2,
            num_init_epochs=10,
            epochs_per_episode=100,
            barrier=True,
            Gamma_factor=0.1,
            include_oracle_loss=False,
            include_estimation_error_loss=False,
        )

        # Test to see if everything runs
        data_module.prepare_data() # Prepare data first

        # Then add
        data_module.add_data(
            aclbf_controller.simulator_fn,
        )

    def define_datamodule_and_params_for_scalar(
            self,
            dynamics_model,
            Theta_lb: np.array,
            Theta_ub: np.array,
    ):
        """
        define_datamodule_and_params_for_scalar
        Description:
            This function defines the datamodule and hyperparameters
            for the scalar CAPA-2 system.
        """

        # Define the hyperparameters
        hyperparams = {
            "batch_size": 64,
            "num_workers": 0,
            "pin_memory": False,
            "sample_quotas": {"safe": 0.2, "unsafe": 0.2, "goal": 0.2},
            "use_oracle": False,
            "barrier": True,
            "safe_level": 0.5,
            "use_estim_err_loss": False,
        }

        # Define the datamodule
        initial_conditions = [
            (1.0, 3.0),  # p_x
        ]
        datamodule = EpisodicDataModuleAdaptive(
            dynamics_model,
            initial_conditions,
            trajectories_per_episode=1,
            trajectory_length=1,
            fixed_samples=10000,
            max_points=100000,
            val_split=0.2,
            batch_size=hyperparams["batch_size"],
            device="cpu",
        )

        # Define the experiment suite
        x_ub, x_lb = dynamics_model.state_limits
        V_contour_experiment = AdaptiveCLFContourExperiment(
            "V_Contour",
            n_grid=30,
            # X axis details
            x_domain=[(x_lb[ScalarCAPA2Demo.X_DEMO], x_ub[ScalarCAPA2Demo.X_DEMO])],  # plotting domain
            x_axis_index=ScalarCAPA2Demo.X_DEMO,
            x_axis_label="$p_x$",
            # Theta axis details
            theta_axis_index=ScalarCAPA2Demo.P_DEMO,
            theta_domain=[(Theta_lb[0], Theta_ub[0])],  # plotting domain for theta
            theta_axis_label="$\\theta$",  # "$\\dot{\\theta}$",
            plot_unsafe_region=False,
        )
        e_suite = ExperimentSuite([V_contour_experiment])

        return datamodule, hyperparams, e_suite

if __name__ == "__main__":
    unittest.main()