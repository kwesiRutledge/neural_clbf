"""
test_neural_aclbf_controller3.py
Description:
    This tests the new object NeuralaCLBFController3.
"""

import torch
from typing import List

from neural_clbf.controllers.adaptive import NeuralaCLBFController3
from neural_clbf.systems.adaptive import LoadSharingManipulator, ControlAffineParameterAffineSystem
from neural_clbf.datamodules import EpisodicDataModuleAdaptive
from neural_clbf.experiments import RolloutStateParameterSpaceExperimentMultiple, ExperimentSuite
from neural_clbf.experiments.adaptive import (
    AdaptiveCLFContourExperiment,
)

import polytope as pc
import numpy as np

import unittest

class TestNeuralaCLBFController3(unittest.TestCase):
    def get_lsm1(
            self,
            theta_lb: List[float] = None,
            theta_ub: List[float] = None,
    )->(ControlAffineParameterAffineSystem):
        """
        Description:
            Creates a LoadSharingManipulator example system for tests.
        """
        # Input Processing
        if theta_lb is None:
            theta_lb = [0.175, 0.4, 0.2]
        if theta_ub is None:
            theta_ub = [0.225, 0.65, 0.3]

        # Define Scenario
        nominal_scenario = {
            "obstacle_center_x": 0.2,
            "obstacle_center_y": 0.1,
            "obstacle_center_z": 0.3,
            "obstacle_width": 0.2,
        }

        Theta = pc.box2poly(np.array([theta_lb, theta_ub]).T)

        # Define the dynamics model
        dynamics_model = LoadSharingManipulator(
            nominal_scenario,
            Theta,
            dt=0.025,
            controller_dt=0.025,
            scenarios=[nominal_scenario],
        )

        return dynamics_model

    def get_lsm_with_neuralaclbfcontroller1(self)->(ControlAffineParameterAffineSystem, NeuralaCLBFController3):
        """
        get_lsm_with_neuralaclbfcontroller1
        Description:
            Creates a LoadSharingManipulator example system for tests.
        """

        # Constants
        theta_lb = [0.175, 0.4, 0.2]
        theta_ub = [0.225, 0.65, 0.3]

        rollout_experiment_horizon = 10.0

        start_x = torch.tensor(
            [
                [0.3, -0.3, 0.4, 0.0, 0.0, 0.0],
                [0.25, -0.25, 0.4, 0.1, 0.0, 0.0],
                [0.35, -0.25, 0.3, 0.0, 0.0, 0.0],
                [0.35, -0.35, 0.3, 0.0, 0.0, 0.0],
                [0.25, -0.35, 0.3, 0.0, 0.0, 0.0],
                [0.15, -0.2, 0.2, 0.0, 0.0, 0.0],
            ]
        )

        # Get System
        lsm0 = self.get_lsm1(theta_lb=theta_lb, theta_ub=theta_ub)

        # Initialize the DataModule
        initial_conditions = [
            (-0.4, 0.4),  # p_x
            (-0.4, 0.4),  # p_y
            (0.0, 0.7),  # p_z
            (-0.5, 0.5),  # v_x
            (-0.5, 0.5),  # v_y
            (-0.5, 0.5),  # v_z
        ]
        datamodule = EpisodicDataModuleAdaptive(
            lsm0,
            initial_conditions,
            trajectories_per_episode=10,
            trajectory_length=20,
            fixed_samples=10000,
            max_points=100000,
            val_split=0.1,
            batch_size=64,
            quotas={"safe": 0.2, "unsafe": 0.2, "goal": 0.2},
            device="cpu",
            num_workers=10,
        )

        # Define Experiment Suite
        contour_exp_theta_index = 0
        lb_Vcontour = theta_lb[contour_exp_theta_index]
        ub_Vcontour = theta_ub[contour_exp_theta_index]
        theta_range_Vcontour = ub_Vcontour - lb_Vcontour
        V_contour_experiment = AdaptiveCLFContourExperiment(
            "V_Contour",
            x_domain=[
                (lsm0.state_limits[1][LoadSharingManipulator.P_X],
                 lsm0.state_limits[0][LoadSharingManipulator.P_X]),
            ],
            theta_domain=[(lb_Vcontour - 0.2 * theta_range_Vcontour, ub_Vcontour + 0.2 * theta_range_Vcontour)],
            n_grid=30,
            x_axis_index=LoadSharingManipulator.P_X,
            theta_axis_index=contour_exp_theta_index,
            x_axis_label="$r_1$",
            theta_axis_label="$\\theta_" + str(contour_exp_theta_index) + "$",  # "$\\dot{\\theta}$",
            plot_unsafe_region=False,
        )
        rollout_experiment2 = RolloutStateParameterSpaceExperimentMultiple(
            "Rollout (Multiple Slices)",
            start_x,
            [LoadSharingManipulator.P_X, LoadSharingManipulator.V_X, LoadSharingManipulator.P_Y],
            ["$r_1$", "$v_1$", "$r_2$"],
            [LoadSharingManipulator.P_X_DES, LoadSharingManipulator.P_X_DES, LoadSharingManipulator.P_Y],
            ["$\\theta_1 (r_1^{(d)})$", "$\\theta_1 (r_1^{(d)})$", "$\\theta_1 (r_2^{(d)})$"],
            scenarios=lsm0.scenarios,
            n_sims_per_start=1,
            t_sim=rollout_experiment_horizon,
        )
        experiment_suite = ExperimentSuite([V_contour_experiment, rollout_experiment2])

        # Define the controller
        controller = NeuralaCLBFController3(
            lsm0,
            lsm0.scenarios,
            datamodule,
            experiment_suite,
        )

        return lsm0, controller

    def test_boundary_loss1(self):
        """
        Description:
            Tests boundary loss for a single point.
        """

        # Constants
        lsm0, controller0 = self.get_lsm_with_neuralaclbfcontroller1()


        # Create data
        x = torch.tensor([
            [0.3, -0.3, 0.4, 0.0, 0.0, 0.0],
        ])
        theta_hat = torch.tensor([
            [0.2, 0.5, 0.25],
        ])
        theta = torch.tensor([
            [0.2, 0.5, 0.25],
        ])
        theta_err_hat = torch.tensor([
            [0.05, 0.05, 0.05],
        ])

        # Mask Data points
        goal_mask = lsm0.goal_mask(
            x, theta,
        )
        safe_mask = lsm0.safe_mask(
            x, theta,
        )
        unsafe_mask = lsm0.unsafe_mask(
            x, theta,
        )

        # Compute loss
        loss = controller0.boundary_loss(
            x, theta_hat, theta, theta_err_hat,
            goal_mask, safe_mask, unsafe_mask,
            accuracy=True,
        )

        print(loss)

        # Check loss
        self.assertGreaterEqual(
            loss[0][1], 0.0,
        )

        pass

    def test_descent_loss1(self):
        """
        test_descent_loss1
        Description:
            Tests boundary loss for a single point.
        """

        # Constants
        lsm0, controller0 = self.get_lsm_with_neuralaclbfcontroller1()

        # Create data
        x = torch.tensor([
            [0.3, -0.3, 0.4, 0.0, 0.0, 0.0],
        ])
        theta_hat = torch.tensor([
            [0.2, 0.5, 0.25],
        ])
        theta = torch.tensor([
            [0.2, 0.5, 0.25],
        ])
        theta_err_hat = torch.tensor([
            [0.05, 0.05, 0.05],
        ])

        # Mask Data points
        goal_mask = lsm0.goal_mask(
            x, theta,
        )
        safe_mask = lsm0.safe_mask(
            x, theta,
        )
        unsafe_mask = lsm0.unsafe_mask(
            x, theta,
        )

        # Compute loss
        loss = controller0.descent_loss(
            x, theta_hat, theta, theta_err_hat,
            goal_mask, safe_mask, unsafe_mask,
            accuracy=True,
        )

        print(loss)

        # Check loss
        self.assertGreaterEqual(
            loss[0][1], 0.0,
        )

        pass




if __name__ == '__main__':
    unittest.main()