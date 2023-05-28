"""
test_aclf_controller5.py
Description:
    This test suite verifies that aCLFController 5 is working with the given definitions.
"""
import unittest
import polytope as pc
import torch

from neural_clbf.experiments import (
    ExperimentSuite,
)
from neural_clbf.experiments.adaptive_w_observed_parameters import (
    AdaptiveCLFContourExperiment3,
)
from neural_clbf.controllers.adaptive_with_observed_parameters import (
    aCLFController5,
)
from neural_clbf.systems.adaptive_w_scenarios import (
    AdaptivePusherSliderStickingForceInput_NObstacles,
)

class TestACLFController5(unittest.TestCase):
    def get_ps1(self):
        """
        get_ps1
        Description:
            Creates a simple pusher slider for testing.
        """

        # Constants
        nominal_scenario = {}
        for ind in range(2):
            nominal_scenario[f"obstacle_{ind}_center_x"] = 0.0
            nominal_scenario[f"obstacle_{ind}_center_y"] = 0.0
            nominal_scenario[f"obstacle_{ind}_radius"] = 0.1

        nominal_scenario["goal_x"] = 0.5
        nominal_scenario["goal_y"] = 0.5

        Theta = pc.box2poly([(0.0, 0.1), (0.0, 0.1)])

        # Create Object
        aps = AdaptivePusherSliderStickingForceInput_NObstacles(
            nominal_scenario, Theta,
        )

        return aps

    def test_init1(self):
        """
        test_init1
        Description:
            Tests that we can correctly initialize such a controller.
        """
        # Constants
        aps = self.get_ps1()

        # Create experiments
        exp1 = AdaptiveCLFContourExperiment3(
            "test-aclf_experiment",
        )
        suite = ExperimentSuite([exp1])

        # Create controller
        controller0 = aCLFController5(
            aps, suite,
        )

        self.assertEqual(
            controller0.clf_lambda, 1.0,
        )
        self.assertEqual(controller0.controller_period, 0.01)


    def test_V_with_jacobian1(self):
        """
        test_V_with_jacobian1
        Description:
            This function tests that the proper compuation is done for V with jacobian in this new aCLFcontroller.
        """
        # Constants
        aps = self.get_ps1()

        # Create experiments
        exp1 = AdaptiveCLFContourExperiment3(
            "test-aclf_experiment",
        )
        suite = ExperimentSuite([exp1])

        # Create controller
        controller0 = aCLFController5(
            aps, suite,
        )

        # Algorithm
        x = torch.tensor([
            [0.1, -0.4, torch.pi/2],
        ])
        theta_hat = aps.sample_Theta_space(1)
        scen = aps.sample_scenario_space(1)

        V, JxV, JthV = controller0.V_with_jacobian(x, theta_hat, scen)

        self.assertGreaterEqual(
            V[0], 0
        )

    def test_V1(self):
        """
        test_V1
        Description:
            This function tests that V works properly.
        """
        # Constants
        aps = self.get_ps1()

        # Create experiments
        exp1 = AdaptiveCLFContourExperiment3(
            "test-aclf_experiment",
        )
        suite = ExperimentSuite([exp1])

        # Create controller
        controller0 = aCLFController5(
            aps, suite,
        )

        # Algorithm
        x = torch.tensor([
            [0.1, -0.4, torch.pi / 2],
        ])
        theta_hat = aps.sample_Theta_space(1)
        scen = aps.sample_scenario_space(1)

        V = controller0.V(x, theta_hat, scen)

        self.assertGreaterEqual(
            V[0], 0
        )

    def test_V_lie_derivatives1(self):
        """
        test_V_lie_derivatives1
        Description:
            Tests that the lie derivative computation works for the aclf controller.
        """
        # Constants
        aps = self.get_ps1()

        # Create experiments
        exp1 = AdaptiveCLFContourExperiment3(
            "test-aclf_experiment",
        )
        suite = ExperimentSuite([exp1])

        # Create controller
        controller0 = aCLFController5(
            aps, suite,
        )

        # Algorithm
        x = torch.tensor([
            [0.1, -0.4, torch.pi / 2],
            [0.1, -0.4, torch.pi / 2],
        ])
        theta_hat = aps.sample_Theta_space(2)
        scen = aps.sample_scenario_space(2)
        scen[:, -1] = -0.4
        scen[0, -2] = 0.0
        scen[1, -2] = 0.2

        Lf_V, LF_V, LFGammadV_V, Lg_V, list_LGi_V, LGammadVG_V = controller0.V_lie_derivatives(x, theta_hat, scen)

        self.assertTrue(
            torch.all(Lf_V == 0.0)
        )
        self.assertFalse(
            torch.all(Lg_V == 0.0),
        )
        for list_idx in range(len(list_LGi_V)):
            self.assertFalse(
                torch.all(list_LGi_V[list_idx] == 0.0),
            )

    def test_Vdot_for_scenario1(self):
        """
        test_Vdot_for_scenario1
        Description:
            Tests whether or not this shortcut function for computing "Vdot" (for barrier condition) is valid.
        """
        # Constants
        aps = self.get_ps1()

        # Create experiments
        exp1 = AdaptiveCLFContourExperiment3(
            "test-aclf_experiment",
        )
        suite = ExperimentSuite([exp1])

        # Create controller
        controller0 = aCLFController5(
            aps, suite,
        )

        # Algorithm
        x = torch.tensor([
            [0.1, -0.4, 0.0],
            [0.1, -0.4, 0.0],
        ])
        theta_hat = aps.sample_Theta_space(2)
        scen = aps.sample_scenario_space(2)
        scen[:, -1] = -0.4
        scen[0, -2] = 0.0
        scen[1, -2] = 0.2

        u = torch.tensor([
            [0.1, 0.05],
            [0.1, 0.05],
        ])

        Vdot = controller0.Vdot_for_scenario(
            x, theta_hat, scen, u,
        )

        self.assertGreaterEqual(Vdot[0], 0)
        self.assertLessEqual(Vdot[1], 0)


    def test__solve_CLF_QP_gurobi1(self):
        """
        test__solve_CLF_QP_gurobi1
        Description
            Tests that the gurobi solver works for the CLF QP.
            Batch size is 2.
        """

        # Constants
        aps = self.get_ps1()

        # Create experiments
        exp1 = AdaptiveCLFContourExperiment3(
            "test-aclf_experiment",
        )
        suite = ExperimentSuite([exp1])

        # Create controller
        controller0 = aCLFController5(
            aps, suite,
        )

        # Create test data
        x = torch.tensor([
            [0.1, -0.4, 0.0],
            [0.1, -0.4, 0.0],
        ])
        theta_hat = aps.sample_Theta_space(2)
        scen = aps.sample_scenario_space(2)
        scen[:, -1] = -0.4
        scen[0, -2] = 0.0
        scen[1, -2] = 0.2

        # Solve aCLF QP
        u, relaxation = controller0._solve_CLF_QP_gurobi(
            x, theta_hat, scen,
            controller0.u_reference(x, theta_hat, scen),
            controller0.V(x, theta_hat, scen),
            1e-3,
        )

        # Check that the solution is valid
        for batch_idx in range(2):
            self.assertTrue(
                aps.U.__contains__(u[batch_idx, :].numpy()),
            )

    def test__solve_CLF_QP_cvxpylayers1(self):
        """
        test__solve_CLF_QP_cvxpylayers1
        Description
            Tests that the CvxPyLayers solver works for the CLF QP.
            Batch size is 2.
        """

        # Constants
        aps = self.get_ps1()

        # Create experiments
        exp1 = AdaptiveCLFContourExperiment3(
            "test-aclf_experiment",
        )
        suite = ExperimentSuite([exp1])

        # Create controller
        controller0 = aCLFController5(
            aps, suite,
        )

        # Create test data
        x = torch.tensor([
            [0.1, -0.4, 0.0],
            [0.1, -0.4, 0.0],
        ])
        theta_hat = aps.sample_Theta_space(2)
        scen = aps.sample_scenario_space(2)
        scen[:, -1] = -0.4
        scen[0, -2] = 0.0
        scen[1, -2] = 0.2

        # Solve aCLF QP
        u_cvxpylayers, relaxation_cvxpylayers = controller0._solve_CLF_QP_cvxpylayers(
            x, theta_hat, scen,
            controller0.u_reference(x, theta_hat, scen),
            controller0.V(x, theta_hat, scen),
            1e-3,
        )

        u_gurobi, relaxation_gurobi = controller0._solve_CLF_QP_gurobi(
            x, theta_hat, scen,
            controller0.u_reference(x, theta_hat, scen),
            controller0.V(x, theta_hat, scen),
            1e-3,
        )

        # Check that the solution is valid
        for batch_idx in range(2):
            self.assertTrue(
                aps.U.__contains__(u_cvxpylayers[batch_idx, :].numpy()),
            )

        print(u_gurobi)
        print(u_cvxpylayers)

        self.assertTrue(
            torch.allclose(u_cvxpylayers, u_gurobi, atol=1e-3),
        )

if __name__ == '__main__':
    unittest.main()