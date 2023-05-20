"""
test_neural_aclbf_controller3.py
Description:
    This tests the new object NeuralaCLBFController3.
"""

import torch
from typing import List

from neural_clbf.controllers.adaptive import NeuralaCLBFController3
from neural_clbf.controllers.adaptive.adaptive_control_utils import define_set_valued_estimator_cvxpylayer1
from neural_clbf.systems.adaptive import LoadSharingManipulator, ControlAffineParameterAffineSystem
from neural_clbf.datamodules import EpisodicDataModuleAdaptive
from neural_clbf.experiments import RolloutStateParameterSpaceExperimentMultiple, ExperimentSuite
from neural_clbf.experiments.adaptive import (
    AdaptiveCLFContourExperiment,
)

import polytope as pc
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

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
            controller_period=lsm0.controller_dt,
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

        #print(loss)

        # Check loss
        self.assertGreaterEqual(
            loss[0][1], 0.0,
        )

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
            x, theta_hat, theta_err_hat, theta,
            goal_mask, safe_mask, unsafe_mask,
            accuracy=True,
        )

        # Check loss
        self.assertGreaterEqual(
            loss[0][1], 0.0,
        )

        # print(loss)
        # print([loss[i][0] for i in range(len(loss))])

        self.assertEqual(
            "Oracle CLBF descent term (simulated)" in [loss[i][0] for i in range(len(loss))],
            controller0.include_oracle_loss,
        )

    def test_estimation_error_update1_1(self):
        """
        estimation_error_update1_1
        Description:
            Tests estimation error update function #1.
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

        # Compute estimation error update
        u = controller0.u(x, theta_hat, theta_err_hat)
        x_dot = controller0.dynamics_model.closed_loop_dynamics(
            x, u, theta_err_hat,
        )
        theta_err_hat_next = controller0.estimation_error_update1(
            x, theta_hat, x_dot, u,
        )

        # Check loss
        self.assertTrue(
            torch.all(torch.ge(theta_err_hat_next, 0.0)),
        )

        # print("theta_err_hat_next")
        # print(theta_err_hat_next)

    def test_estimation_error_update1_2(self):
        """
        estimation_error_update1_2
        Description
            Testing the cvxpylayers logic that I need to use for the estimation error update.
        """

        # Constants
        lsm0, controller0 = self.get_lsm_with_neuralaclbfcontroller1()
        n_dims = lsm0.n_dims
        n_params = lsm0.n_params

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

        # Compute estimation error update
        u = controller0.u(x, theta_hat, theta_err_hat)
        x_dot = controller0.dynamics_model.closed_loop_dynamics(
            x, u, theta,
        )

        # Create the estimation error program
        delta_theta = cp.Variable((controller0.dynamics_model.n_params,))
        #theta_hat = cp.Parameter((controller0.dynamics_model.n_params,))
        F_plus_Gu = cp.Parameter((controller0.dynamics_model.n_dims, controller0.dynamics_model.n_params))
        D_minus_extras1 = cp.Parameter((controller0.dynamics_model.n_dims,))
        D_minus_extras2 = cp.Parameter((controller0.dynamics_model.n_dims,))

        # Create the constraints
        constraints = []
        constraints.append(
            - F_plus_Gu @ delta_theta <= D_minus_extras1
        )
        constraints.append(
            F_plus_Gu @ delta_theta <= D_minus_extras2
        )
        constraints.append(
            controller0.dynamics_model.Theta.A @ delta_theta <= controller0.dynamics_model.Theta.b,
        )

        # Create the objective
        objective_expression = np.ones((1, controller0.dynamics_model.n_params)) @ delta_theta
        objective = cp.Maximize(objective_expression)

        # Create the problem
        problem = cp.Problem(objective, constraints)
        self.assertTrue(problem.is_dpp())

        variables = [delta_theta]
        parameters = [F_plus_Gu, D_minus_extras1, D_minus_extras2]

        # Create the cvxpylayer
        layer0 = CvxpyLayer(problem, variables=variables, parameters=parameters)

        # Use Layer
        params = []
        dynamics_model = controller0.dynamics_model
        s = dynamics_model.nominal_scenario

        f = dynamics_model._f(x, s)
        F = dynamics_model._F(x, s)
        g = dynamics_model._g(x, s)
        G = dynamics_model._G(x, s)

        F_plus_Gu = F
        for theta_index in range(controller0.dynamics_model.n_params):
            F_plus_Gu = F_plus_Gu + torch.bmm(
                G[:, :, :, theta_index], u.unsqueeze(2),
            )

        D = 1e-3

        D_minus_extras1_val = D * torch.ones((x.shape[0], dynamics_model.n_dims)) - \
                              x_dot + \
                              f.squeeze(2) + \
                              torch.bmm(g, u.unsqueeze(2)).squeeze(2) #+ \
                              #torch.bmm(F_plus_Gu, theta_hat.unsqueeze(2)).squeeze(2)

        D_minus_extras2_val = D * torch.ones((x.shape[0], dynamics_model.n_dims)) + \
                              x_dot - \
                              f.squeeze(2) - \
                              torch.bmm(g, u.unsqueeze(2)).squeeze(2) #- \
                              #torch.bmm(F_plus_Gu, theta_hat.unsqueeze(2)).squeeze(2)

        result = layer0(
            *[F_plus_Gu, D_minus_extras1_val, D_minus_extras2_val],
            solver_args={"max_iters": int(1e6)},
        )

        theta_err_hat_next = result[0] - theta_hat

        print("theta_err_hat_next = ", theta_err_hat_next)


        # Check loss
        self.assertTrue(
            torch.all(torch.ge(theta_err_hat_next, 0.0)),
        )

    # def test_estimation_error_update1_3(self):
    #     """
    #     estimation_error_update1_3
    #     Description
    #         Testing the cvxpylayers logic that I need to use for the estimation error update.
    #
    #         This implementation uses the set containment idea.
    #     """
    #
    #     # Constants
    #     lsm0, controller0 = self.get_lsm_with_neuralaclbfcontroller1()
    #     n_dims = lsm0.n_dims
    #     n_params = lsm0.n_params
    #
    #     # Create data
    #     x = torch.tensor([
    #         [0.3, -0.3, 0.4, 0.0, 0.0, 0.0],
    #     ])
    #     theta_hat = torch.tensor([
    #         [0.2, 0.5, 0.25],
    #     ])
    #     theta = torch.tensor([
    #         [0.2, 0.5, 0.25],
    #     ])
    #     theta_err_hat = torch.tensor([
    #         [0.05, 0.05, 0.05],
    #     ])
    #
    #     # Mask Data points
    #     goal_mask = lsm0.goal_mask(
    #         x, theta,
    #     )
    #     safe_mask = lsm0.safe_mask(
    #         x, theta,
    #     )
    #     unsafe_mask = lsm0.unsafe_mask(
    #         x, theta,
    #     )
    #
    #     # Compute estimation error update
    #     u = controller0.u(x, theta_hat, theta_err_hat)
    #     x_dot = controller0.dynamics_model.closed_loop_dynamics(
    #         x, u, theta_err_hat,
    #     )
    #
    #     # Create the estimation error program
    #     delta_theta = cp.Variable((controller0.dynamics_model.n_params,), nonneg=True)
    #     #theta_hat = cp.Parameter((controller0.dynamics_model.n_params,))
    #     F_plus_Gu = cp.Parameter((controller0.dynamics_model.n_dims, controller0.dynamics_model.n_params))
    #     D_minus_extras1 = cp.Parameter((controller0.dynamics_model.n_dims,))
    #     D_minus_extras2 = cp.Parameter((controller0.dynamics_model.n_dims,))
    #
    #     Lambda = cp.Variable((2*n_params, n_dims), nonneg=True)
    #
    #     # Create the constraints
    #     constraints = []
    #     constraints.append(
    #         Lambda[:n_params, :] @ (- F_plus_Gu) == np.eye(n_params)
    #     )
    #     constraints.append(
    #         Lambda[n_params:, :] @ F_plus_Gu == -np.eye(n_params)
    #     )
    #
    #     constraints.append(
    #         Lambda[:n_params, :] @ D_minus_extras1 <= delta_theta
    #     )
    #     constraints.append(
    #         Lambda[n_params:, :] @ D_minus_extras2 <= delta_theta
    #     )
    #     # constraints.append(
    #     #     controller0.dynamics_model.Theta.A @ delta_theta <= controller0.dynamics_model.Theta.b,
    #     # )
    #
    #     print(controller0.dynamics_model.Theta)
    #
    #     # Create the objective
    #     objective_expression = np.ones((1, controller0.dynamics_model.n_params)) @ delta_theta
    #     objective = cp.Minimize(objective_expression)
    #
    #     # Create the problem
    #     problem = cp.Problem(objective, constraints)
    #     self.assertTrue(problem.is_dpp())
    #
    #     variables = [delta_theta, Lambda]
    #     parameters = [F_plus_Gu, D_minus_extras1, D_minus_extras2]
    #
    #     # Create the cvxpylayer
    #     layer0 = CvxpyLayer(problem, variables=variables, parameters=parameters)
    #
    #     # Use Layer
    #     params = []
    #     dynamics_model = controller0.dynamics_model
    #     s = dynamics_model.nominal_scenario
    #
    #     f = dynamics_model._f(x, s)
    #     F = dynamics_model._F(x, s)
    #     g = dynamics_model._g(x, s)
    #     G = dynamics_model._G(x, s)
    #
    #     F_plus_Gu = F
    #     for theta_index in range(controller0.dynamics_model.n_params):
    #         F_plus_Gu[:, :, theta_index] = F_plus_Gu[:, :, theta_index] + torch.bmm(
    #             G[:, :, :, theta_index], u.unsqueeze(2),
    #         ).squeeze()
    #
    #     D = 1e-1
    #
    #     D_minus_extras1_val = D * torch.ones((x.shape[0], dynamics_model.n_dims)) - \
    #                           x_dot + \
    #                           f.squeeze(2) + \
    #                           torch.bmm(g, u.unsqueeze(2)).squeeze(2) #+ \
    #                           #torch.bmm(F_plus_Gu, theta_hat.unsqueeze(2)).squeeze(2)
    #
    #     D_minus_extras2_val = D * torch.ones((x.shape[0], dynamics_model.n_dims)) + \
    #                           x_dot - \
    #                           f.squeeze(2) - \
    #                           torch.bmm(g, u.unsqueeze(2)).squeeze(2) #- \
    #                           #torch.bmm(F_plus_Gu, theta_hat.unsqueeze(2)).squeeze(2)
    #
    #     try:
    #         result = layer0(
    #             *[F_plus_Gu, D_minus_extras1_val, D_minus_extras2_val],
    #             solver_args={"max_iters": int(1e7)},
    #         )
    #
    #         theta_err_hat_next = result[0]
    #
    #         print("theta_err_hat_next = ", theta_err_hat_next)
    #
    #
    #         # Check loss
    #         self.assertTrue(
    #             torch.all(torch.ge(theta_err_hat_next, 0.0)),
    #         )
    #
    #
    #         print("theta_err_hat_next")
    #         print(theta_err_hat_next)
    #     except Exception as e:
    #         # We expect for an error to be caught
    #         self.assertTrue(True)

    def test_estimation_error_update1_4(self):
        """
        estimation_error_update1_2
        Description
            Testing the cvxpylayers logic that I need to use for the estimation error update.
            Half of logic is placed in function.
        """

        # Constants
        lsm0, controller0 = self.get_lsm_with_neuralaclbfcontroller1()
        n_dims = lsm0.n_dims
        n_params = lsm0.n_params

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

        # Compute estimation error update
        u = controller0.u(x, theta_hat, theta_err_hat)
        x_dot = controller0.dynamics_model.closed_loop_dynamics(
            x, u, theta,
        )

        layer1 = define_set_valued_estimator_cvxpylayer1(lsm0)

        # Use Layer
        params = []
        dynamics_model = controller0.dynamics_model
        s = dynamics_model.nominal_scenario

        f = dynamics_model._f(x, s)
        F = dynamics_model._F(x, s)
        g = dynamics_model._g(x, s)
        G = dynamics_model._G(x, s)

        F_plus_Gu = F
        for theta_index in range(controller0.dynamics_model.n_params):
            F_plus_Gu = F_plus_Gu + torch.bmm(
                G[:, :, :, theta_index], u.unsqueeze(2),
            )

        D = 1e-2

        D_minus_extras1_val = D * torch.ones((x.shape[0], dynamics_model.n_dims)) - \
                              x_dot + \
                              f.squeeze(2) + \
                              torch.bmm(g, u.unsqueeze(2)).squeeze(2)  # + \
        # torch.bmm(F_plus_Gu, theta_hat.unsqueeze(2)).squeeze(2)

        D_minus_extras2_val = D * torch.ones((x.shape[0], dynamics_model.n_dims)) + \
                              x_dot - \
                              f.squeeze(2) - \
                              torch.bmm(g, u.unsqueeze(2)).squeeze(2)  # - \
        # torch.bmm(F_plus_Gu, theta_hat.unsqueeze(2)).squeeze(2)

        result = layer1(
            *[F_plus_Gu, D_minus_extras1_val, D_minus_extras2_val],
            solver_args={"max_iters": int(1e6)},
        )

        theta_err_hat_next = result[0] - theta_hat
        theta_err_hat_next2 = theta_hat - result[1]

        print("theta_err_hat_next = ", theta_err_hat_next)
        print("theta_err_hat_next2 = ", theta_err_hat_next2)

        # Check loss
        self.assertTrue(
            torch.all(torch.ge(theta_err_hat_next, 0.0)),
        )

        theta_bar = torch.max(torch.stack([
            theta_err_hat_next, theta_err_hat_next2
        ], dim=2), dim=2)

        print(theta_bar[0])

    def test_initial_loss1(self):
        """
        test_initial_loss1
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

        # Compute loss
        loss = controller0.initial_loss(
            x, theta_hat, theta_err_hat,
        )

        # Check loss
        self.assertGreaterEqual(
            loss[0][1], 0.0,
        )

        self.assertEqual(
            len(loss), 1,
        )

    def test_training_step1(self):
        """
        Description:
            Tests training step function with a batch containing a single point.
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

        batch = ( x, theta, theta_hat, goal_mask, safe_mask, unsafe_mask )
        batch_idx = 2


        # Compute step
        dict_out = controller0.training_step(
            batch, batch_idx,
        )

        # Check loss
        print("batch dictionary: ", dict_out)
        self.assertIn(
            "loss", dict_out.keys(),
        )

    def test_validation_step1(self):
        """
        test_validation_step1
        Description:
            Tests validation step function with a batch containing a single point.
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

        batch = ( x, theta, theta_hat, goal_mask, safe_mask, unsafe_mask )
        batch_idx = 2


        # Compute step
        dict_out = controller0.validation_step(
            batch, batch_idx,
        )

        # Check loss
        #print("batch dictionary: ", dict_out)
        self.assertIn(
            "val_loss", dict_out.keys(),
        )
        self.assertNotIn(
            "loss", dict_out.keys(),
        )

    def test_simulate1(self):
        """
        test_simulate1
        Description:
            This test evaluates the simulate function for some simple input conditions.
        """

        # Constants
        lsm0, controller0 = self.get_lsm_with_neuralaclbfcontroller1()
        num_steps = 20

        # Create data
        x_init = torch.tensor([
            [0.3, -0.3, 0.4, 0.0, 0.0, 0.0],
            [0.3, -0.3, 0.4, 0.0, -0.1, 0.1],
        ])
        theta_hat = torch.tensor([
            [0.2, 0.5, 0.25],
        ])
        theta = torch.tensor([
            [0.2, 0.5, 0.25],
            [0.25, 0.55, 0.3]
        ])
        theta_err_hat = torch.tensor([
            [0.05, 0.05, 0.05],
        ])

        # Mask Data points
        goal_mask = lsm0.goal_mask(
            x_init, theta,
        )
        safe_mask = lsm0.safe_mask(
            x_init, theta,
        )
        unsafe_mask = lsm0.unsafe_mask(
            x_init, theta,
        )

        # Compute step
        tuple_out = controller0.simulate(
            x_init, theta,
            num_steps, controller0.u,
        )



if __name__ == '__main__':
    unittest.main()