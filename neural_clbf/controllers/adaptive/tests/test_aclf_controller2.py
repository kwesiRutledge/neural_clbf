"""
test_aclf_controller2.py
Description:
    This script is meant to test some of the basic functions of the ACLF controller object.
"""

import cvxpy as cp
import cvxpylayers as cvl
import numpy as np
import polytope as pc

from cvxpylayers.torch import CvxpyLayer

import torch
import torch.nn.functional as F

from neural_clbf.controllers.adaptive import aCLFController2
from neural_clbf.systems.adaptive import (
    ScalarCAPA2Demo, ControlAffineParameterAffineSystem, LoadSharingManipulator,
)
from neural_clbf.systems.utils import Scenario
from neural_clbf.experiments import (
    AdaptiveCLFContourExperiment, ExperimentSuite, RolloutStateParameterSpaceExperiment,
    RolloutStateParameterSpaceExperimentMultiple,
)

import unittest
from typing import List
import diffqcqp as dqcqp

from qpth.qp import QPFunction

class TestaCLFController2(unittest.TestCase):
    def test_aclfcontroller_layerconcept1(self):
        """
        Description:
            Tests some basic functionality of the layer.
        """

        # Constants

        # Algorithm (Copied from ChatGPT) (There were many bugs...)

        # Create a variable for the layer
        x = cp.Variable(shape=(2, 1))
        P = cp.Parameter(shape=(2, 2), symmetric=True)

        # Define the quadratic function
        f = 0.5 * cp.quad_form(x, P)
        obj = cp.Minimize(cp.sum_squares(x))

        # Create the quadratic layer
        problem = cp.Problem(obj, [])
        quad_layer = CvxpyLayer(problem, variables=[x], parameters=[])

        # Test (Created by ChatGPT for functionality) Many bugs...

        # check if the object has the expected properties
        assert isinstance(quad_layer, CvxpyLayer)
        assert len(quad_layer.variables) == 1
        assert quad_layer.variables[0] is x
        #assert quad_layer.shape == (1, 1)

        # check if it correctly represents the quadratic function
        # quad_layer_output = cvl.Parameter(quad_layer.size)
        # quad_layer_output.value = np.array([[1]])
        # assert cp.norm(quad_layer(x) - quad_layer_output, p=np.inf) < 1e-6

    def test_aclfcontroller_layerconcept2(self):
        """
        Description:
            Tests some basic functionality of the layer.
        """

        # Constants

        # Algorithm (Copied from ChatGPT) (There were many bugs...)

        z = np.ones((2, 1))

        # Create a variable for the layer
        x = cp.Variable(shape=(2, 1))
        P = cp.Parameter(shape=(2, 2), symmetric=True)

        # Define the quadratic function
        f = 0.5 * cp.quad_form(x, P)
        obj = cp.Minimize( cp.sum_squares(x) )

        # Create the quadratic layer
        problem = cp.Problem(obj, [z.T @ x >= 0.5])
        quad_layer = CvxpyLayer(problem, variables=[x], parameters=[])

        # Test (Created by ChatGPT for functionality) Many bugs...

        # check if the object has the expected properties
        assert isinstance(quad_layer, CvxpyLayer)
        assert len(quad_layer.variables) == 1
        assert quad_layer.variables[0] is x

    def test_aclfcontroller_V_with_jacobian1(self):
        """
        Description:
            Tests the inner components of the method V_with_jacobian of the aclf_controller object.
            Doesn't instantiate the object, but uses the logic with some dummy values.
        """

        # Constants
        # =========

        dynamics_model, _ = self.get_scalar_system()
        dynamics_model.P = torch.eye(dynamics_model.n_dims)

        # Create the data
        batch_size = 4
        x = torch.zeros((batch_size, 1))
        x[0, 0] = 0.5
        x[1, 0] = 0.6
        x[2, 0] = 0.7

        theta_hat = torch.zeros((batch_size, 1))
        theta_hat[0, 0] = 0.5
        theta_hat[1, 0] = 0.6
        theta_hat[2, 0] = 0.7

        # Create batches of x-theta pairs
        x_theta = torch.cat([x, theta_hat], dim=1)

        # First, get the Lyapunov function value and gradient at this state
        P_x = dynamics_model.P.type_as(x_theta)
        # Reshape to use pytorch's bilinear function
        P = torch.zeros(
            1,
            dynamics_model.n_dims + dynamics_model.n_params,
            dynamics_model.n_dims + dynamics_model.n_params
        )
        P[0, :dynamics_model.n_dims, :dynamics_model.n_dims] = P_x
        V = 0.5 * F.bilinear(x_theta, x_theta, P).squeeze()
        V = V.reshape(x_theta.shape[0])

        # Reshape again for the gradient calculation
        P = P.reshape(
            dynamics_model.n_dims + dynamics_model.n_params,
            dynamics_model.n_dims + dynamics_model.n_params
        )
        JV = F.linear(x_theta, P)
        JV = JV.reshape(x_theta.shape[0], 1, dynamics_model.n_dims + dynamics_model.n_params)

        # print("V: ", V)
        # print("JV: ", JV)

        assert torch.all(V >= 0.0)

    def get_scalar_system(self) -> [ControlAffineParameterAffineSystem, Scenario]:
        scenario0 = {
            "wall_position": -2.0,
        }

        th_dim = 1
        lb = 0.5
        ub = 0.8
        Theta = pc.box2poly(np.array([[lb], [ub]]).T)

        scalar_system = ScalarCAPA2Demo(scenario0, Theta)

        return scalar_system, scenario0

    def test_aclfcontroller_V_lie_derivatives1(self):
        """
        Description:
            Tests the inner components of the method V_lie_derivatives of the aclf_controller object.
            Doesn't instantiate the object, but uses the logic with some dummy values.
        """

        # Constants
        # =========

        dynamics_model, scenario0 = self.get_scalar_system()
        dynamics_model.P = torch.Tensor([[1.0, 0.0], [0.0, 1.0]])

        # Create the data
        batch_size = 4
        x = torch.zeros((batch_size, 1))
        x[0, 0] = 1.2
        x[1, 0] = 0.6
        x[2, 0] = 0.7

        theta_hat = torch.zeros((batch_size, 1))
        theta_hat[0, 0] = 0.68552
        theta_hat[1, 0] = 0.6
        theta_hat[2, 0] = 0.5

        # Algorithm
        # =========
        print("test_aclfcontroller_V_lie_derivatives1")

        scenarios = [scenario0]
        n_scenarios = len(scenarios)

        # Get the Jacobian of V for each entry in the batch
        x_theta = torch.cat([x, theta_hat], dim=1)

        # First, get the Lyapunov function value and gradient at this state
        P = dynamics_model.P.type_as(x_theta)
        # Reshape to use pytorch's bilinear function
        P = P.reshape(
            1,
            dynamics_model.n_dims + dynamics_model.n_params,
            dynamics_model.n_dims + dynamics_model.n_params
        )
        Va = F.bilinear(x_theta, x_theta, P).squeeze()
        Va = Va.reshape(x_theta.shape[0])

        # Reshape again for the gradient calculation
        P = P.reshape(
            dynamics_model.n_dims + dynamics_model.n_params,
            dynamics_model.n_dims + dynamics_model.n_params
        )
        JxV = 2 * F.linear(x, P[:dynamics_model.n_dims, :dynamics_model.n_dims]) + \
              2 * F.linear(theta_hat, P[dynamics_model.n_dims:dynamics_model.n_dims + dynamics_model.n_params, :dynamics_model.n_dims])
        JxV = JxV.reshape(x.shape[0], 1, dynamics_model.n_dims)

        JthV = 2 * F.linear(theta_hat, P[dynamics_model.n_dims:dynamics_model.n_dims + dynamics_model.n_params, dynamics_model.n_dims:dynamics_model.n_dims + dynamics_model.n_params]) + \
            2 * F.linear(x, P[:dynamics_model.n_dims, dynamics_model.n_dims:dynamics_model.n_dims + dynamics_model.n_params])
        JxV = JxV.reshape(x.shape[0], 1, dynamics_model.n_params)

        JthV = JthV.reshape(x_theta.shape[0], 1, dynamics_model.n_params)

        # We need to compute Lie derivatives for each scenario
        batch_size = x.shape[0]
        Lf_V = torch.zeros(batch_size, n_scenarios, 1)
        LF_V = torch.zeros(batch_size, n_scenarios, dynamics_model.n_params)
        Lg_V = torch.zeros(batch_size, n_scenarios, dynamics_model.n_controls)
        LG_V = torch.zeros(batch_size, n_scenarios, dynamics_model.n_controls, dynamics_model.n_params)

        Lf_V = Lf_V.type_as(x)
        LF_V = LF_V.type_as(x)
        Lg_V = Lg_V.type_as(x)
        LG_V = LG_V.type_as(x)

        for i in range(n_scenarios):
            # Get the dynamics f and g for this scenario
            s = scenarios[i]
            f = dynamics_model._f(x, s)
            Fterm = dynamics_model._F(x, s)
            g = dynamics_model._g(x, s)
            G = dynamics_model._G(x, s)

            # Multiply these with the Jacobian to get the Lie derivatives
            # print("gradV: ", JxV.shape)
            # print(torch.bmm(JxV, f))
            Lf_V[:, i, :] = torch.bmm(JxV, f).squeeze(1)
            LF_V[:, i, :] = torch.bmm(JxV, Fterm).squeeze(1)
            Lg_V[:, i, :] = torch.bmm(JxV, g).squeeze(1)
            for mode_index in range(dynamics_model.n_params):
                G_i = G[:, :, :, mode_index].reshape(
                    (batch_size, dynamics_model.n_dims, dynamics_model.n_controls))
                LG_V[:, i, :, mode_index] = torch.bmm(JxV, G_i).squeeze(1)

        # Check the first elements
        assert torch.isclose(Lf_V[0, 0, 0], torch.Tensor([2.8800]))
        assert torch.isclose(LF_V[0, 0, 0], torch.Tensor([2.8800]))
        assert torch.isclose(Lg_V[0, 0, 0], torch.Tensor([2.4000]))
        assert torch.isclose(LG_V[0, 0, 0], torch.Tensor([2.4000]))

    def test_aclfcontroller_init1(self):
        """
        Description:
            Tests the initialization of the ACLF controller object.
        """

        # Constants
        scenario0 = {
            "wall_position": -2.0,
        }

        th_dim = 1
        lb = [0.5]
        ub = [0.8]
        Theta = pc.box2poly(np.array([lb, ub]).T)

        scalar_system = ScalarCAPA2Demo(scenario0, Theta)

        # Create ExperimentSuite
        experiment0 = AdaptiveCLFContourExperiment(
            "experiment0",
            [(-2.0, 2.0), (0.4, 0.9)],
        )

        suite0 = ExperimentSuite([experiment0])

        # Define Controller
        controller = aCLFController2(scalar_system, [scenario0], suite0)

    # def test_aclfcontroller_gradient1():
    #     """
    #     Description:
    #         Attempts to create the gradient descent condition that the aclf_controller object will use.
    #         This should satisfy the aCLF gradient condition and not just the normal robust aCLF condition.
    #     """
    #
    #     1

    def test_aclfcontroller_V_oracle1(self):
        """
        Description:
            Tests the V_oracle method of the ACLF controller.
        """

        print("test_aclfcontroller_V_oracle1")

        # Constants
        # =========

        scalar_system, _ = self.get_scalar_system()
        scalar_system.P = torch.eye(scalar_system.n_dims)

        # Create the data
        batch_size = 4
        x = torch.zeros((batch_size, 1))
        x[0, 0] = 0.5
        x[1, 0] = 0.6
        x[2, 0] = 0.7
        x[3, 0] = 0.5

        theta_hat = torch.zeros((batch_size, 1))
        theta_hat[0, 0] = 0.5
        theta_hat[1, 0] = 0.6
        theta_hat[2, 0] = 0.7
        theta_hat[3, 0] = 0.8

        # Create true thetas
        theta = torch.zeros((batch_size, 1))
        theta[0, 0] = 0.5
        theta[1, 0] = 0.6
        theta[2, 0] = 0.7
        theta[3, 0] = 0.5

        # Create the ACLF controller
        scenario0 = {
            "wall_position": -2.0,
        }
        experiment0 = AdaptiveCLFContourExperiment(
            "experiment0",
            [(-2.0, 2.0), (0.4, 0.9)],
        )
        suite0 = ExperimentSuite([experiment0])
        controller = aCLFController2(scalar_system, [scenario0], suite0)

        # Test the V_oracle method
        Vo = controller.V_oracle(x, theta_hat, theta)
        Va = controller.V(x, theta_hat)

        assert torch.isclose(Vo[0], Va[0])
        assert torch.isclose(Vo[1], Va[1])
        assert torch.isclose(Vo[2], Va[2])
        assert not torch.isclose(Vo[3], Va[0])


        print("Vo", Vo)
        print("Va", Va)

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


    def get_lsm_with_aclfcontroller1(self)->(ControlAffineParameterAffineSystem, aCLFController2):
        """
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
        controller = aCLFController2(
            lsm0,
            lsm0.scenarios,
            experiment_suite,
        )

        return lsm0, controller


    def test_aclfcontroller2__solve_CLF_QP_cvxpylayers1(self):
        """
        Description:
            Tests that cvxpylayers is solving the right problem.
        """

        # Constants
        lsm0, controller0 = self.get_lsm_with_aclfcontroller1()

        relaxation_penalty = 1e3

        # Define initial state and parameter estimate
        x0 = 0.5*torch.ones((1, lsm0.n_dims))
        theta_hat0 = torch.Tensor([0.2, 0.5, 0.25]).reshape(
            (1, lsm0.n_params)
        )

        V0 = controller0.V(x0, theta_hat0)

        # Run solve
        u_qp, r_qp = controller0._solve_CLF_QP_cvxpylayers(
            x0,
            lsm0.u_nominal(x0, theta_hat0),
            V0,
            relaxation_penalty,
        )

        print(u_qp)
        print("r = ", r_qp)

        self.assertAlmostEqual(float(r_qp), 0.0, places=5)
        # self.assertGreaterEqual(r_qp, 0.0)

    def test_aclfcontroller2__solve_CLF_QP_cvxpylayers2(self):
        """
        Description:
            Tests that cvxpylayers is solving the right problem. Is it close to Gurobi's solution?
        """

        # Constants
        lsm0, controller0 = self.get_lsm_with_aclfcontroller1()

        relaxation_penalty = 1e3

        # Define initial state and parameter estimate
        x0 = 0.5*torch.ones((1, lsm0.n_dims))
        theta_hat0 = torch.Tensor([0.2, 0.5, 0.25]).reshape(
            (1, lsm0.n_params)
        )

        V0 = controller0.V(x0, theta_hat0)

        # Run solve
        u_qp_cvxpylayers, r_qp_cvxpylayers = controller0._solve_CLF_QP_cvxpylayers(
            x0,
            lsm0.u_nominal(x0, theta_hat0),
            V0,
            relaxation_penalty,
        )

        self.assertAlmostEqual(float(r_qp_cvxpylayers), 0.0, places=5)

        u_qp_gurobi, r_qp_gurobi = controller0.solve_CLF_QP(
            x0, theta_hat0,
            u_ref=lsm0.u_nominal(x0, theta_hat0),
            relaxation_penalty=relaxation_penalty,
        )

        # self.assertGreaterEqual(r_qp, 0.0)

        print("u_qp_cvxpylayers = ", u_qp_cvxpylayers)
        print("u_qp_gurobi = ", u_qp_gurobi)

        self.assertLessEqual(torch.norm(u_qp_cvxpylayers - u_qp_gurobi), 1e-5)
        self.assertAlmostEqual(float(r_qp_gurobi), float(r_qp_cvxpylayers), places=5)

    def test_aclfcontroller2_solve_CLF_QP_qpth1(self):
        """
        test_aclfcontroller2_solve_CLF_QP_qpth1
        Description:
            Testing the ability of QPTH to solve the CLFQP that we're interested in solving.
        """

        # Create constants
        lsm0, controller0 = self.get_lsm_with_aclfcontroller1()
        relaxation_penalty = 1e3
        x0 = 0.5 * torch.ones((1, lsm0.n_dims))  # initial state
        x0[0, LoadSharingManipulator.P_Z] = 0.1
        # x0[0, LoadSharingManipulator.V_Z] = 0.5
        # x0[0, LoadSharingManipulator.V_X] = 0.5
        # x0[0, LoadSharingManipulator.V_Y] = 0.5
        theta_hat0 = torch.Tensor([0.2, 0.5, 0.25]).reshape(  # initial parameter estimate
            (1, lsm0.n_params)
        )
        batch_size = 1

        # Create objective
        Q_u = torch.eye(lsm0.n_controls)
        Q_r = torch.ones((1, 1)) * 1e-4

        Q = torch.zeros((lsm0.n_controls + 1, lsm0.n_controls + 1))
        Q[:lsm0.n_controls, :lsm0.n_controls] = Q_u
        Q[-1, -1] = Q_r
        Q = Q.repeat(batch_size, 1, 1)

        q_u = -2.*torch.bmm(Q[:, :lsm0.n_controls, :lsm0.n_controls], lsm0.u_nominal(x0, theta_hat0).unsqueeze(2)).squeeze(2)
        q_r = torch.zeros((1))
        q_r[0] = relaxation_penalty

        q = torch.zeros((batch_size, lsm0.n_controls + 1))
        q[:, :lsm0.n_controls] = q_u
        q[:, -1] = q_r
        # q = q.repeat(batch_size, 1)

        # Create constraints
        V0 = controller0.V(x0, theta_hat0)
        Lf_V, LF_V, LFGammadV_V, Lg_V, list_LGi_V, LGammadVG_V = controller0.V_lie_derivatives(x0, theta_hat0)
        # print("Lie derivatives")
        # print("Lf_V = ", Lf_V)
        # print("LF_V = ", LF_V)
        # print("LFGammadV_V = ", LFGammadV_V)
        # print("Lg_V = ", Lg_V)
        # print("list_LGi_V = ", list_LGi_V)
        # print("LGammadVG_V = ", LGammadVG_V)
        n_scenarios = len(controller0.scenarios)
        sum_LG_V = torch.zeros((batch_size, n_scenarios, lsm0.n_controls), device=x0.device)
        for theta_dim in range(lsm0.n_params):
            sum_LG_V = sum_LG_V + \
                       torch.bmm(theta_hat0[:, theta_dim].reshape((batch_size, 1, 1)), list_LGi_V[theta_dim]).to(x0.device)

        #G_u = torch.tensor(lsm0.U.A).repeat(batch_size, 1, 1)
        dhdt_lhs = Lg_V[:, 0, :].unsqueeze(1) + sum_LG_V + LGammadVG_V
        G_u = torch.zeros(
            (batch_size, lsm0.U.A.shape[0]+dhdt_lhs.shape[1], lsm0.U.A.shape[1]),
        )
        G_u[:, :lsm0.U.A.shape[0], :lsm0.U.A.shape[1]] = torch.tensor(lsm0.U.A)
        G_u[:, lsm0.U.A.shape[0]:, :] = dhdt_lhs
        print("dhdt_lhs = ", dhdt_lhs)

        G_r = torch.tensor([[-1.0]]).repeat(batch_size, 1)
        G = torch.zeros(
            (batch_size, G_u.shape[1]+1, G_u.shape[2]+1)
        )
        G[:, :G_u.shape[1], :G_u.shape[2]] = G_u
        # Define lie derivative columns in G_u
        lie_deriv_rows = [G_u.shape[1]-1]
        G[:, lie_deriv_rows, G_u.shape[2]] = -1.0
        G[:, G_u.shape[1]:, G_u.shape[2]:] = G_r

        h_u = torch.zeros(
            (batch_size, G_u.shape[1])
        )
        h_u[:, :lsm0.U.b.shape[0]] = torch.tensor(lsm0.U.b).repeat(batch_size, 1)
        h_u[:, lsm0.U.b.shape[0]:] = -Lf_V[:, 0, :] - torch.bmm(
            LF_V[:, 0, :].unsqueeze(1),
            theta_hat0.reshape((batch_size, theta_hat0.shape[1], 1)),
        ) - LFGammadV_V[:, 0, :].unsqueeze(1) - controller0.clf_lambda * V0

        h_r = torch.zeros((batch_size, 1))
        h_r[:, 0] = 0.0
        h = torch.zeros(
            (batch_size, h_u.shape[1] + h_r.shape[1])
        )
        h[:, :h_u.shape[1]] = h_u
        h[:, -1] = h_r

        A = torch.zeros((batch_size, 0, G.shape[2]))
        b = torch.zeros((batch_size, 0))

        z = QPFunction()(Q, q, G, h, A, b)

        print("z = ", z)

        self.assertAlmostEqual(
            float(z[0,-1]), 0,
            5, # places
        )


    def test_aclfcontroller2_solve_CLF_QP_qpth2(self):
        """
        Description:
            Testing the ability of QPTH to solve the CLFQP that we're interested in solving.
        """

        # Create constants
        lsm0, controller0 = self.get_lsm_with_aclfcontroller1()
        relaxation_penalty = 1e3
        x0 = 0.5 * torch.ones((1, lsm0.n_dims))  # initial state
        x0[0, LoadSharingManipulator.P_Z] = 0.1
        # x0[0, LoadSharingManipulator.V_Z] = 0.5
        # x0[0, LoadSharingManipulator.V_X] = 0.5
        # x0[0, LoadSharingManipulator.V_Y] = 0.5
        theta_hat0 = torch.Tensor([0.2, 0.5, 0.25]).reshape(  # initial parameter estimate
            (1, lsm0.n_params)
        )
        batch_size = 1

        # Create objective
        Q_u = torch.eye(lsm0.n_controls)
        Q_r = torch.ones((1, 1)) * 1e-4

        Q = torch.zeros((lsm0.n_controls + 1, lsm0.n_controls + 1))
        Q[:lsm0.n_controls, :lsm0.n_controls] = Q_u
        Q[-1, -1] = Q_r
        Q = Q.repeat(batch_size, 1, 1)

        q_u = -2.*torch.bmm(Q[:, :lsm0.n_controls, :lsm0.n_controls], lsm0.u_nominal(x0, theta_hat0).unsqueeze(2)).squeeze(2)
        q_r = torch.zeros((1))
        q_r[0] = relaxation_penalty

        q = torch.zeros((batch_size, lsm0.n_controls + 1))
        q[:, :lsm0.n_controls] = q_u
        q[:, -1] = q_r
        # q = q.repeat(batch_size, 1)

        # Create constraints

        V_Theta = pc.extreme(lsm0.Theta)
        n_V_Theta = V_Theta.shape[0]
        n_scenarios = len(controller0.scenarios)

        G_u = torch.zeros(
            (batch_size, lsm0.U.A.shape[0] + 1*n_V_Theta, lsm0.U.A.shape[1]),
        )
        G_u[:, :lsm0.U.A.shape[0], :lsm0.U.A.shape[1]] = torch.tensor(lsm0.U.A)
        for theta_index in range(n_V_Theta):
            print("theta_index = ", theta_index)
            theta_i = torch.tensor(V_Theta[theta_index, :]).reshape((batch_size, lsm0.n_params)).type_as(x0)

            V_i = controller0.V(x0, theta_i)
            Lf_V_i, LF_V_i, LFGammadV_V_i, Lg_V_i, list_LGi_V_i, LGammadVG_V_i = controller0.V_lie_derivatives(x0, theta_i)

            sum_LG_V_i = torch.zeros((batch_size, n_scenarios, lsm0.n_controls), device=x0.device)
            for theta_dim in range(lsm0.n_params):
                sum_LG_V_i = sum_LG_V_i + \
                           torch.bmm(
                               theta_i[:, theta_dim].reshape((batch_size, 1, 1)), list_LGi_V_i[theta_dim]
                           ).to(x0.device)

            #G_u = torch.tensor(lsm0.U.A).repeat(batch_size, 1, 1)
            dhdt_lhs_i = Lg_V_i[:, 0, :].unsqueeze(1) + sum_LG_V_i + LGammadVG_V_i
            G_u[:, lsm0.U.A.shape[0]+theta_index, :] = dhdt_lhs_i

            print("dhdt_lhs_i = ", dhdt_lhs_i)

        G_r = torch.tensor([[-1.0]]).repeat(batch_size, 1)

        G = torch.zeros(
            (batch_size, G_u.shape[1]+1, G_u.shape[2]+1)
        )
        G[:, :G_u.shape[1], :G_u.shape[2]] = G_u

        # Define lie derivative columns in G_u
        lie_deriv_rows = range(G_u.shape[1]-n_V_Theta, G_u.shape[1])
        G[:, lie_deriv_rows, G_u.shape[2]] = -1.0
        G[:, G_u.shape[1]:, G_u.shape[2]:] = G_r

        h_u = torch.zeros(
            (batch_size, G_u.shape[1])
        )
        h_u[:, :lsm0.U.b.shape[0]] = torch.tensor(lsm0.U.b).repeat(batch_size, 1)
        for theta_index in range(n_V_Theta):
            theta_i = torch.tensor(V_Theta[theta_index, :]).reshape(
                (batch_size, lsm0.n_params)
            ).type_as(x0)

            V_i = controller0.V(x0, theta_i)
            Lf_V_i, LF_V_i, LFGammadV_V_i, Lg_V_i, list_LGi_V_i, LGammadVG_V_i = controller0.V_lie_derivatives(
                x0, theta_i,
            )
            h_u[:, lsm0.U.b.shape[0]+theta_index] = -Lf_V_i[:, 0, :] - torch.bmm(
                LF_V_i[:, 0, :].unsqueeze(1),
                theta_i.reshape((batch_size, theta_i.shape[1], 1)),
            ) - LFGammadV_V_i[:, 0, :].unsqueeze(1) - controller0.clf_lambda * V_i

        h_r = torch.zeros((batch_size, 1))
        h_r[:, 0] = 0.0
        h = torch.zeros(
            (batch_size, h_u.shape[1] + h_r.shape[1])
        )
        h[:, :h_u.shape[1]] = h_u
        h[:, -1] = h_r

        A = torch.zeros((batch_size, 0, G.shape[2]))
        b = torch.zeros((batch_size, 0))

        z = QPFunction()(Q, q, G, h, A, b)

        print("z = ", z)

        self.assertGreaterEqual(z[0, -1], 0)

    def test_aclfcontroller2_solve_CLF_QP_qpth3(self):
        """
        Description:
            Testing the ability of QPTH to solve the CLFQP that we're interested in solving.
            Using batches of 10.
        """

        # Create constants
        lsm0, controller0 = self.get_lsm_with_aclfcontroller1()
        relaxation_penalty = 1e3
        batch_size = 10
        x0 = torch.rand(
            (batch_size, lsm0.n_dims)
        )
        theta_hat0 = lsm0.sample_Theta_space(batch_size)

        # Create objective
        Q_u = torch.eye(lsm0.n_controls)
        Q_r = torch.ones((1, 1)) * 1e-4

        Q = torch.zeros((lsm0.n_controls + 1, lsm0.n_controls + 1))
        Q[:lsm0.n_controls, :lsm0.n_controls] = Q_u
        Q[-1, -1] = Q_r
        Q = Q.repeat(batch_size, 1, 1)

        q_u = -2.*torch.bmm(Q[:, :lsm0.n_controls, :lsm0.n_controls], lsm0.u_nominal(x0, theta_hat0).unsqueeze(2)).squeeze(2)
        q_r = torch.zeros((1))
        q_r[0] = relaxation_penalty

        q = torch.zeros((batch_size, lsm0.n_controls + 1))
        q[:, :lsm0.n_controls] = q_u
        q[:, -1] = q_r
        # q = q.repeat(batch_size, 1)

        # Create constraints

        V_Theta = pc.extreme(lsm0.Theta)
        n_V_Theta = V_Theta.shape[0]
        n_scenarios = len(controller0.scenarios)

        G_u = torch.zeros(
            (batch_size, lsm0.U.A.shape[0] + 1*n_V_Theta, lsm0.U.A.shape[1]),
        )
        G_u[:, :lsm0.U.A.shape[0], :lsm0.U.A.shape[1]] = torch.tensor(lsm0.U.A)
        for theta_index in range(n_V_Theta):
            print("theta_index = ", theta_index)
            theta_i = torch.tensor(V_Theta[theta_index, :]).repeat(
                (batch_size, 1)
            ).type_as(x0)

            V_i = controller0.V(x0, theta_i)
            Lf_V_i, LF_V_i, LFGammadV_V_i, Lg_V_i, list_LGi_V_i, LGammadVG_V_i = controller0.V_lie_derivatives(x0, theta_i)

            sum_LG_V_i = torch.zeros((batch_size, n_scenarios, lsm0.n_controls), device=x0.device)
            for theta_dim in range(lsm0.n_params):
                sum_LG_V_i = sum_LG_V_i + \
                           torch.bmm(
                               theta_i[:, theta_dim].reshape((batch_size, 1, 1)), list_LGi_V_i[theta_dim]
                           ).to(x0.device)

            #G_u = torch.tensor(lsm0.U.A).repeat(batch_size, 1, 1)
            dhdt_lhs_i = Lg_V_i[:, 0, :].unsqueeze(1) + sum_LG_V_i + LGammadVG_V_i
            G_u[:, lsm0.U.A.shape[0]+theta_index, :] = dhdt_lhs_i[:, 0, :]

            print("dhdt_lhs_i = ", dhdt_lhs_i)

        G_r = torch.tensor([[-1.0]]).repeat(batch_size, 1, 1)

        G = torch.zeros(
            (batch_size, G_u.shape[1]+1, G_u.shape[2]+1)
        )
        G[:, :G_u.shape[1], :G_u.shape[2]] = G_u

        # Define lie derivative columns in G_u
        lie_deriv_rows = range(G_u.shape[1]-n_V_Theta, G_u.shape[1])
        G[:, lie_deriv_rows, G_u.shape[2]] = -1.0
        G[:, G_u.shape[1]:, G_u.shape[2]:] = G_r

        h_u = torch.zeros(
            (batch_size, G_u.shape[1])
        )
        h_u[:, :lsm0.U.b.shape[0]] = torch.tensor(lsm0.U.b).repeat(batch_size, 1)
        for theta_index in range(n_V_Theta):
            theta_i = torch.tensor(V_Theta[theta_index, :]).repeat(
                (batch_size, 1)
            ).type_as(x0)

            V_i = controller0.V(x0, theta_i)
            Lf_V_i, LF_V_i, LFGammadV_V_i, Lg_V_i, list_LGi_V_i, LGammadVG_V_i = controller0.V_lie_derivatives(
                x0, theta_i,
            )
            print("test1 = ",  torch.bmm(
                    LF_V_i[:, 0, :].unsqueeze(1),
                    theta_i.reshape((batch_size, theta_i.shape[1], 1)),
                ).shape)
            print("test2 = ", LFGammadV_V_i[:, 0, :].unsqueeze(1).shape)
            print("test3 = ", V_i.shape)
            print("test4 = ", Lf_V_i[:, 0, :].shape)
            h_u[:, lsm0.U.b.shape[0]+theta_index:lsm0.U.b.shape[0]+theta_index+1] = \
                -Lf_V_i[:, 0, :] - \
                torch.bmm(
                    LF_V_i[:, 0, :].unsqueeze(1),
                    theta_i.reshape((batch_size, theta_i.shape[1], 1)),
                ).squeeze(2) - \
                LFGammadV_V_i[:, 0, :] \
                - controller0.clf_lambda * V_i.reshape((batch_size, 1))

        h_r = torch.zeros((batch_size, 1))
        h_r[:, 0] = 0.0
        h = torch.zeros(
            (batch_size, h_u.shape[1] + h_r.shape[1])
        )
        h[:, :h_u.shape[1]] = h_u
        h[:, -1] = h_r.squeeze(1)

        A = torch.zeros((batch_size, 0, G.shape[2]))
        b = torch.zeros((batch_size, 0))

        z = QPFunction()(Q, q, G, h, A, b)

        print("z = ", z)

        self.assertGreaterEqual(z[0, -1], 0)

    def test_aclfcontroller2_solve_CLF_QP_qpth4(self):
        """
        Description:
            Testing the ability of QPTH to solve the CLFQP that we're interested in solving.
            Using batches of 10.
            Compares output of this with function output.
        """

        # Create constants
        lsm0, controller0 = self.get_lsm_with_aclfcontroller1()
        relaxation_penalty = 1e3
        batch_size = 10
        x0 = torch.rand(
            (batch_size, lsm0.n_dims)
        )
        theta_hat0 = lsm0.sample_Theta_space(batch_size)

        V0 = controller0.V(x0, theta_hat0)

        # Create objective
        Q_u = torch.eye(lsm0.n_controls)
        Q_r = torch.ones((1, 1)) * 1e-4

        Q = torch.zeros((lsm0.n_controls + 1, lsm0.n_controls + 1))
        Q[:lsm0.n_controls, :lsm0.n_controls] = Q_u
        Q[-1, -1] = Q_r
        Q = Q.repeat(batch_size, 1, 1)

        q_u = -2.*torch.bmm(Q[:, :lsm0.n_controls, :lsm0.n_controls], lsm0.u_nominal(x0, theta_hat0).unsqueeze(2)).squeeze(2)
        q_r = torch.zeros((1))
        q_r[0] = relaxation_penalty

        q = torch.zeros((batch_size, lsm0.n_controls + 1))
        q[:, :lsm0.n_controls] = q_u
        q[:, -1] = q_r
        # q = q.repeat(batch_size, 1)

        # Create constraints

        V_Theta = pc.extreme(lsm0.Theta)
        n_V_Theta = V_Theta.shape[0]
        n_scenarios = len(controller0.scenarios)

        G_u = torch.zeros(
            (batch_size, lsm0.U.A.shape[0] + 1*n_V_Theta, lsm0.U.A.shape[1]),
        )
        G_u[:, :lsm0.U.A.shape[0], :lsm0.U.A.shape[1]] = torch.tensor(lsm0.U.A)
        for theta_index in range(n_V_Theta):
            print("theta_index = ", theta_index)
            theta_i = torch.tensor(V_Theta[theta_index, :]).repeat(
                (batch_size, 1)
            ).type_as(x0)

            V_i = controller0.V(x0, theta_i)
            Lf_V_i, LF_V_i, LFGammadV_V_i, Lg_V_i, list_LGi_V_i, LGammadVG_V_i = controller0.V_lie_derivatives(x0, theta_i)

            sum_LG_V_i = torch.zeros((batch_size, n_scenarios, lsm0.n_controls), device=x0.device)
            for theta_dim in range(lsm0.n_params):
                sum_LG_V_i = sum_LG_V_i + \
                           torch.bmm(
                               theta_i[:, theta_dim].reshape((batch_size, 1, 1)), list_LGi_V_i[theta_dim]
                           ).to(x0.device)

            #G_u = torch.tensor(lsm0.U.A).repeat(batch_size, 1, 1)
            dhdt_lhs_i = Lg_V_i[:, 0, :].unsqueeze(1) + sum_LG_V_i + LGammadVG_V_i
            G_u[:, lsm0.U.A.shape[0]+theta_index, :] = dhdt_lhs_i[:, 0, :]

            print("dhdt_lhs_i = ", dhdt_lhs_i)

        G_r = torch.tensor([[-1.0]]).repeat(batch_size, 1, 1)

        G = torch.zeros(
            (batch_size, G_u.shape[1]+1, G_u.shape[2]+1)
        )
        G[:, :G_u.shape[1], :G_u.shape[2]] = G_u

        # Define lie derivative columns in G_u
        lie_deriv_rows = range(G_u.shape[1]-n_V_Theta, G_u.shape[1])
        G[:, lie_deriv_rows, G_u.shape[2]] = -1.0
        G[:, G_u.shape[1]:, G_u.shape[2]:] = G_r

        h_u = torch.zeros(
            (batch_size, G_u.shape[1])
        )
        h_u[:, :lsm0.U.b.shape[0]] = torch.tensor(lsm0.U.b).repeat(batch_size, 1)
        for theta_index in range(n_V_Theta):
            theta_i = torch.tensor(V_Theta[theta_index, :]).repeat(
                (batch_size, 1)
            ).type_as(x0)

            V_i = controller0.V(x0, theta_i)
            Lf_V_i, LF_V_i, LFGammadV_V_i, Lg_V_i, list_LGi_V_i, LGammadVG_V_i = controller0.V_lie_derivatives(
                x0, theta_i,
            )
            print("test1 = ",  torch.bmm(
                    LF_V_i[:, 0, :].unsqueeze(1),
                    theta_i.reshape((batch_size, theta_i.shape[1], 1)),
                ).shape)
            print("test2 = ", LFGammadV_V_i[:, 0, :].unsqueeze(1).shape)
            print("test3 = ", V_i.shape)
            print("test4 = ", Lf_V_i[:, 0, :].shape)
            h_u[:, lsm0.U.b.shape[0]+theta_index:lsm0.U.b.shape[0]+theta_index+1] = \
                -Lf_V_i[:, 0, :] - \
                torch.bmm(
                    LF_V_i[:, 0, :].unsqueeze(1),
                    theta_i.reshape((batch_size, theta_i.shape[1], 1)),
                ).squeeze(2) - \
                LFGammadV_V_i[:, 0, :] \
                - controller0.clf_lambda * V_i.reshape((batch_size, 1))

        h_r = torch.zeros((batch_size, 1))
        h_r[:, 0] = 0.0
        h = torch.zeros(
            (batch_size, h_u.shape[1] + h_r.shape[1])
        )
        h[:, :h_u.shape[1]] = h_u
        h[:, -1] = h_r.squeeze(1)

        A = torch.zeros((batch_size, 0, G.shape[2]))
        b = torch.zeros((batch_size, 0))

        z = QPFunction()(Q, q, G, h, A, b)

        # Use Function
        u_qpth, r_qpth = controller0._solve_CLF_QP_qpth(
            x0, theta_hat0, V0, relaxation_penalty=relaxation_penalty,
        )
        print(u_qpth)
        print(z)

        self.assertAlmostEqual(
            torch.linalg.norm(u_qpth - z[:, :lsm0.n_controls], dim=0).max().item(),
            0.0,
        )

if __name__ == "__main__":
    unittest.main()
