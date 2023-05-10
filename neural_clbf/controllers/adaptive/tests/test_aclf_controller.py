"""
test_aclf_controller.py
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

from neural_clbf.controllers.adaptive import aCLFController
from neural_clbf.systems.adaptive import (
    ScalarCAPA2Demo, ControlAffineParameterAffineSystem
)
from neural_clbf.systems.utils import Scenario
from neural_clbf.experiments import (
    AdaptiveCLFContourExperiment, ExperimentSuite
)

import unittest

class TestaCLFController(unittest.TestCase):
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
        dynamics_model.P = torch.Tensor([[1.0, 0.0], [0.0, 1.0]])

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
        P = dynamics_model.P.type_as(x_theta)
        # Reshape to use pytorch's bilinear function
        P = P.reshape(
            1,
            dynamics_model.n_dims + dynamics_model.n_params,
            dynamics_model.n_dims + dynamics_model.n_params
        )
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
        controller = aCLFController(scalar_system, [scenario0], suite0)

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
        scalar_system.P = torch.Tensor([[1.0, 0.0], [0.0, 1.0]])

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
        controller = aCLFController(scalar_system, [scenario0], suite0)

        # Test the V_oracle method
        Vo = controller.V_oracle(x, theta_hat, theta)
        Va = controller.V(x, theta_hat)

        assert torch.isclose(Vo[0], Va[0])
        assert torch.isclose(Vo[1], Va[1])
        assert torch.isclose(Vo[2], Va[2])
        assert not torch.isclose(Vo[3], Va[0])


        print("Vo", Vo)
        print("Va", Va)

if __name__ == "__main__":
    unittest.main()
