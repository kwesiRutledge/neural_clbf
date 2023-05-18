"""
test_aclf_controller3.py
Description:
    This script is meant to test the adaptive controller that uses parameter estimation error in its computations.
"""

import numpy as np
from typing import List

from neural_clbf.systems.adaptive import (
    ControlAffineParameterAffineSystem, LoadSharingManipulator,
)
from neural_clbf.experiments.adaptive import (
    AdaptiveCLFContourExperiment,
)
from neural_clbf.experiments import (
    RolloutStateParameterSpaceExperimentMultiple, ExperimentSuite,
)

from neural_clbf.controllers.adaptive import (
    aCLFController3,
)

import torch
import polytope as pc
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

import unittest

class TestaCLFController3(unittest.TestCase):
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

    def get_lsm_with_aclfcontroller1(self)->(ControlAffineParameterAffineSystem, aCLFController3):
        """
        get_lsm_with_aclfcontroller1
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
        controller = aCLFController3(
            lsm0,
            lsm0.scenarios,
            experiment_suite,
        )

        return lsm0, controller

    def test_controller_V_with_jacobian1(self):
        """
        test_controller_V_with_Jacobian1
        Description:
            Tests how well the Jacobian computation works in this example.
        """
        # Constants
        lsm0, controller0 = self.get_lsm_with_aclfcontroller1()

        x0 = torch.tensor([0.3, -0.3, 0.4, 0.0, 0.0, 0.0]).reshape(1, -1)
        theta_hat0 = torch.tensor([0.2, 0.5, 0.25]).reshape(1, -1)
        theta_err0 = torch.tensor([0.05, 0.05, 0.10]).reshape(1, lsm0.n_params)

        # Algorithm
        V0, JxV0, JthetaV0, JthetaerrV0 = controller0.V_with_jacobian(x0, theta_hat0, theta_err0)

        print(controller0.V_with_jacobian(x0, theta_hat0, theta_err0))

        self.assertEqual(
            torch.norm(JthetaerrV0, p=2).item(),
            0.0,
        )

        self.assertGreaterEqual(V0[0], 0.0)

    def test_controller_V_with_jacobian2(self):
        """
        test_controller_V_with_Jacobian2
        Description:
            Tests how well the Jacobian computation works in this example.
        """
        # Constants
        lsm0, controller0 = self.get_lsm_with_aclfcontroller1()

        x0 = torch.tensor([0.2, 0.5, 0.25, 0.0, 0.0, 0.0]).reshape(1, -1)
        theta_hat0 = torch.tensor([0.2, 0.5, 0.25]).reshape(1, -1)
        theta_err0 = torch.tensor([0.0, 0.0, 0.0]).reshape(1, lsm0.n_params)

        # Algorithm
        V0, JxV0, JthetaV0, JthetaerrV0 = controller0.V_with_jacobian(x0, theta_hat0, theta_err0)

        print(controller0.V_with_jacobian(x0, theta_hat0, theta_err0))

        self.assertEqual(
            torch.norm(JthetaerrV0, p=2).item(),
            0.0,
        )

        self.assertGreaterEqual(V0[0], 0.0)

        self.assertEqual(V0[0], 0.0)

    def test_controller_V_with_jacobian3(self):
        """
        test_controller_V_with_Jacobian3
        Description:
            Tests how well the Jacobian computation works in this example.
            Want to choose a point where the state is at the goal point, but the uncertainty is too high.
        """
        # Constants
        lsm0, controller0 = self.get_lsm_with_aclfcontroller1()

        x0 = torch.tensor([0.2, 0.5, 0.25, 0.0, 0.0, 0.0]).reshape(1, -1)
        theta_hat0 = torch.tensor([0.2, 0.5, 0.25]).reshape(1, -1)
        theta_err0 = torch.tensor([0.05, 0.05, 0.05]).reshape(1, lsm0.n_params)

        # Algorithm
        V0, JxV0, JthetaV0, JthetaerrV0 = controller0.V_with_jacobian(x0, theta_hat0, theta_err0)

        print(controller0.V_with_jacobian(x0, theta_hat0, theta_err0))

        self.assertEqual(
            torch.norm(JthetaerrV0, p=2).item(),
            0.0,
        )

        self.assertGreaterEqual(V0[0], 0.0)

        self.assertNotEqual(V0[0], 0.0)

    def test_binary_padding1(self):
        """
        test_binary_padding1
        Description:
            Tests how well the binary padding works before inserting it into a function.
        """
        # Constants
        testInt = 11

        # Algorithm
        binaryString = bin(testInt)[2:]
        print(binaryString.zfill(6))

        self.assertTrue(len(binaryString) == 4)

    def test_initialize_cvxpylayers1(self):
        # Constants
        dynamics_model = self.get_lsm1()
        scenarios = dynamics_model.scenarios

        clf_lambda = 1.0
        Q_u = np.eye(dynamics_model.n_controls)

        # Since we want to be able to solve the CLF-QP differentiably, we need to set
        # up the CVXPyLayers optimization. First, we define variables for each control
        # input and the relaxation in each scenario
        u = cp.Variable(dynamics_model.n_controls)
        clf_relaxation = cp.Variable(1, nonneg=True)

        # Next, we define the parameters that will be supplied at solve-time: the value
        # of the Lyapunov function, its Lie derivatives, the relaxation penalty, and
        # the reference control input
        Va_param = cp.Parameter(1, nonneg=True)
        n_params = dynamics_model.n_params

        n_Theta_hat_vertices = 2 ** dynamics_model.n_params
        # theta_hat_param = cp.Parameter(n_params)
        # theta_err_param = cp.Parameter(n_params)

        Lf_Va_params = []
        Lg_Va_params = []
        LF_Va_thhat_params = []
        LFGammadV_Va_params = []
        LG_Va_thhat_params = []
        LGammadVaG_params = []  # LGammaVaG[scenario_idx] = (dVa/dx) * (\sum_i Gamma[i,:] * (dVa/dtheta).T * G_i)

        Theta_vertices = pc.extreme(dynamics_model.Theta)
        n_Theta_v = Theta_vertices.shape[0]
        for scenario in dynamics_model.scenarios:
            Lf_Va_param_cluster = []
            Lg_Va_param_cluster = []
            LF_Va_thhat_param_cluster = []
            LGammadVaG_param_cluster = []
            LFGammadV_Va_param_cluster = []
            LG_Va_thhat_cluster = []
            for theta_idx in range(n_Theta_v):

                Lf_Va_param_cluster.append(cp.Parameter(1))
                Lg_Va_param_cluster.append(cp.Parameter(dynamics_model.n_controls))
                LF_Va_thhat_param_cluster.append(cp.Parameter(1))
                LGammadVaG_param_cluster.append(cp.Parameter(dynamics_model.n_controls))

                # LFGammadV_Va_clusters = []
                # for v_Theta in Theta_vertices:
                #     LFGammadV_Va_clusters.append(cp.Parameter(1))
                # LFGammadV_Va_params.append(LFGammadV_Va_clusters)
                LFGammadV_Va_param_cluster.append(cp.Parameter(1))

                # LG_Va_clump = []
                # for theta_dim in range(dynamics_model.n_params):
                #     LG_Va_clump.append(cp.Parameter(dynamics_model.n_controls))
                LG_Va_thhat_cluster.append(cp.Parameter(dynamics_model.n_controls))

            # Add all clusters to main set of params
            Lf_Va_params.append(Lf_Va_param_cluster)
            Lg_Va_params.append(Lg_Va_param_cluster)
            LF_Va_thhat_params.append(LF_Va_thhat_param_cluster)
            LGammadVaG_params.append(LGammadVaG_param_cluster)
            LFGammadV_Va_params.append(LFGammadV_Va_param_cluster)
            LG_Va_thhat_params.append(LG_Va_thhat_cluster)

        clf_relaxation_penalty_param = cp.Parameter(1, nonneg=True)
        u_ref_param = cp.Parameter(dynamics_model.n_controls)

        # These allow us to define the constraints
        constraints = []
        for i in range(len(dynamics_model.scenarios)):
            for v_Theta_index in range(n_Theta_hat_vertices):
                # Get the current Theta_hat error bar
                # format_string = '{0:' + str(n_params) + 'b}'
                # binnum_as_str = bin(v_Theta_index)[2:].zfill(n_params)  # Convert number to binary string
                # binnum = [int(digit) for digit in binnum_as_str]  # Convert to list of digits

                # v_Theta = theta_hat_param + \
                #           np.diag(binnum) * theta_err_param - \
                #           (np.eye(len(binnum)) - np.diag(binnum)) * theta_err_param

                # Create the sum G_i term
                # sum_LG_i_Va = v_Theta[0] * LG_Va_params[i][v_Theta_index][0]
                # for theta_index in range(1, n_params):
                #     sum_LG_i_Va = sum_LG_i_Va + v_Theta[theta_index] * LG_Va_params[i][v_Theta_index][theta_index]

                # CLF decrease constraint (with relaxation)
                constraints.append(
                    Lf_Va_params[i][v_Theta_index]
                    + LF_Va_thhat_params[i][v_Theta_index] + LFGammadV_Va_params[i][v_Theta_index]
                    + (Lg_Va_params[i][v_Theta_index] + LG_Va_thhat_params[i][v_Theta_index] + LGammadVaG_params[i][v_Theta_index]) @ u
                    + clf_lambda * Va_param
                    - clf_relaxation
                    <= 0
                )

        # Relaxation Additional Constraint

        # Control limit constraints
        U = dynamics_model.U
        constraints.append(
            U.A @ u <= U.b,
        )

        # And define the objective
        objective_expression = cp.quad_form(u - u_ref_param, Q_u)
        objective_expression = objective_expression + cp.multiply(clf_relaxation_penalty_param, clf_relaxation)
        objective = cp.Minimize(objective_expression)

        # Finally, create the optimization problem
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()
        variables = [u] + [clf_relaxation]
        parameters = [Va_param, u_ref_param, clf_relaxation_penalty_param]
        for s_idx in range(len(scenarios)):
            parameters = parameters + Lf_Va_params[s_idx] + Lg_Va_params[s_idx] + LF_Va_thhat_params[s_idx]
            # for theta_dim_idx in range(n_params):
            parameters = parameters + LFGammadV_Va_params[s_idx]
            parameters = parameters + LG_Va_thhat_params[s_idx]
            # for LG_Va_cluster in LG_Va_params[s_idx]:
            #     parameters = parameters + LG_Va_cluster
            parameters = parameters + LG_Va_thhat_params[s_idx]
            parameters = parameters + LGammadVaG_params[s_idx]

        differentiable_qp_solver = CvxpyLayer(
            problem, variables=variables, parameters=parameters
        )

        self.assertTrue(True)

    def test__solve_CLF_QP_gurobi1(self):
        # Constants
        lsm0, controller0 = self.get_lsm_with_aclfcontroller1()

        x0 = torch.tensor([0.3, -0.3, 0.4, 0.0, 0.0, 0.0]).reshape(1, -1)
        theta_hat0 = torch.tensor([0.2, 0.5, 0.25]).reshape(1, -1)
        theta_err0 = torch.tensor([0.05, 0.05, 0.10]).reshape(1, lsm0.n_params)

        # Compute QP solution
        u, r = controller0._solve_CLF_QP_gurobi(
            x0, theta_hat0, theta_err0,
            controller0.u_reference(x0, theta_hat0, theta_err0),
            controller0.V(x0, theta_hat0, theta_err0),
            1e4,
        )

        self.assertGreaterEqual(
            r[0], -1e-6,
        )

    def test__solve_CLF_QP_cvxpylayers1(self):
        """
        test__solve_CLF_QP_cvxpylayers1
        Description:
            Test to make sure that the cvxpylayers problem solves the same problem that gurobi is solving.
        """
        # Constants
        lsm0, controller0 = self.get_lsm_with_aclfcontroller1()

        x0 = torch.tensor([0.3, -0.3, 0.4, 0.0, 0.0, 0.0]).reshape(1, -1)
        theta_hat0 = torch.tensor([0.2, 0.5, 0.25]).reshape(1, -1)
        theta_err0 = torch.tensor([0.05, 0.05, 0.10]).reshape(1, lsm0.n_params)

        relaxation_penalty = 1e4

        # Compute QP solution
        u, r = controller0._solve_CLF_QP_gurobi(
            x0, theta_hat0, theta_err0,
            controller0.u_reference(x0, theta_hat0, theta_err0),
            controller0.V(x0, theta_hat0, theta_err0),
            relaxation_penalty,
        )

        u_cvxpylayers, r_cvxpylayers = controller0._solve_CLF_QP_cvxpylayers(
            x0, theta_hat0, theta_err0,
            controller0.u_reference(x0, theta_hat0, theta_err0),
            controller0.V(x0, theta_hat0, theta_err0),
            relaxation_penalty,
        )

        self.assertGreaterEqual(
            r[0], -1e-6,
        )

        print("u_cvxpylayers = ", u_cvxpylayers)
        print("u = ", u)

        self.assertTrue(
            torch.allclose(u, u_cvxpylayers, atol=1e-3),
        )



if __name__ == '__main__':
    unittest.main()