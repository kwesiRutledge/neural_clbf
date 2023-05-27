"""
test_apfsi_no_g.py
Description:
    Tests the new AdaptivePusherSliderForceInput_NoObstacles but with parametric goal position.
"""

import torch
import numpy as np

import unittest

import polytope as pc

from neural_clbf.systems.adaptive_w_scenarios import AdaptivePusherSliderStickingForceInput_NObstacles

class TestAPSFI_NO_G(unittest.TestCase):
    def test_init1(self):
        """
        test_init1()
        Description:
            Attempts to initialize one of the objects.
        """

        # Constants
        nominal_scenario = {}
        for ind in range(2):
            nominal_scenario[f"obstacle_{ind}_center_x"] = 0.0
            nominal_scenario[f"obstacle_{ind}_center_y"] = 0.0
            nominal_scenario[f"obstacle_{ind}_radius"] = 0.1

        nominal_scenario["goal_x"] = 0.5
        nominal_scenario["goal_y"] = 0.5

        Theta = pc.box2poly([[0.0, 0.0], [1.0, 1.0]])

        # Create Object
        aps = AdaptivePusherSliderStickingForceInput_NObstacles(
            nominal_scenario, Theta,
        )

        self.assertEqual(aps.N_SCENARIO, 2*aps.N_OBSTACLES+2)

    def test_scenario_set1(self):
        """
        test_scenario_set1()
        Description:
            Tests the scenario set function.
            Verifies that the dimension is correct.
        """
        # Constants
        nominal_scenario = {}
        for ind in range(2):
            nominal_scenario[f"obstacle_{ind}_center_x"] = 0.0
            nominal_scenario[f"obstacle_{ind}_center_y"] = 0.0
            nominal_scenario[f"obstacle_{ind}_radius"] = 0.1

        nominal_scenario["goal_x"] = 0.5
        nominal_scenario["goal_y"] = 0.5

        Theta = pc.box2poly([[0.0, 0.0], [1.0, 1.0]])

        # Create Object
        aps = AdaptivePusherSliderStickingForceInput_NObstacles(
            nominal_scenario, Theta,
        )

        P_scenario = aps.scenario_set

        # Test
        self.assertEqual(P_scenario.dim, aps.N_OBSTACLES*3+2)

    def test_safe_mask1(self):
        """
        test_safe_mask1
        Description:
            Tests whether or not safe_mask correctly understands when a safe point is safe.
        """
        # Constants
        nominal_scenario = {}
        for ind in range(2):
            nominal_scenario[f"obstacle_{ind}_center_x"] = 0.0
            nominal_scenario[f"obstacle_{ind}_center_y"] = 0.0
            nominal_scenario[f"obstacle_{ind}_radius"] = 0.1

        nominal_scenario["goal_x"] = 0.5
        nominal_scenario["goal_y"] = 0.5

        Theta = pc.box2poly([[0.0, 0.0], [1.0, 1.0]])

        # Create Object
        aps = AdaptivePusherSliderStickingForceInput_NObstacles(
            nominal_scenario, Theta,
        )

        # Test safe mask
        test_state = torch.tensor([[0.5, 0.0, 0.0]])
        test_theta = torch.tensor([[0.01, 0.03]])
        test_s_vec = torch.tensor(aps.scenario_to_list(nominal_scenario)).unsqueeze(0)

        self.assertTrue(
            torch.all(aps.safe_mask(test_state, test_theta, test_s_vec))
        )

    def test_safe_mask2(self):
        """
        test_safe_mask2
        Description:
            Tests whether or not safe_mask correctly understands when an unsafe point is unsafe.
        """
        # Constants
        nominal_scenario = {}
        nominal_scenario[f"obstacle_0_center_x"] = 0.0
        nominal_scenario[f"obstacle_0_center_y"] = 0.0
        nominal_scenario[f"obstacle_0_radius"] = 0.1

        nominal_scenario[f"obstacle_1_center_x"] = 1.0
        nominal_scenario[f"obstacle_1_center_y"] = 0.0
        nominal_scenario[f"obstacle_1_radius"] = 0.1

        nominal_scenario["goal_x"] = 0.5
        nominal_scenario["goal_y"] = 0.5

        Theta = pc.box2poly([[0.0, 0.0], [1.0, 1.0]])

        # Create Object
        aps = AdaptivePusherSliderStickingForceInput_NObstacles(
            nominal_scenario, Theta,
        )

        # Test safe mask
        test_state = torch.tensor([[0.95, 0.0, 0.0]])
        test_theta = torch.tensor([[0.01, 0.03]])
        test_s_vec = torch.tensor(aps.scenario_to_list(nominal_scenario)).unsqueeze(0)

        self.assertTrue(
            not torch.all(aps.safe_mask(test_state, test_theta, test_s_vec))
        )

    def test_unsafe_mask1(self):
        """
        test_unsafe_mask1
        Description:
            Tests whether or not unsafe_mask correctly understands when a safe point is safe.
        """
        # Constants
        nominal_scenario = {}
        for ind in range(2):
            nominal_scenario[f"obstacle_{ind}_center_x"] = 0.0
            nominal_scenario[f"obstacle_{ind}_center_y"] = 0.0
            nominal_scenario[f"obstacle_{ind}_radius"] = 0.1

        nominal_scenario["goal_x"] = 0.5
        nominal_scenario["goal_y"] = 0.5

        Theta = pc.box2poly([[0.0, 0.0], [1.0, 1.0]])

        # Create Object
        aps = AdaptivePusherSliderStickingForceInput_NObstacles(
            nominal_scenario, Theta,
        )

        # Test safe mask
        test_state = torch.tensor([[0.5, 0.0, 0.0]])
        test_theta = torch.tensor([[0.01, 0.03]])
        test_s_vec = torch.tensor(aps.scenario_to_list(nominal_scenario)).unsqueeze(0)

        self.assertTrue(
            not torch.all(aps.unsafe_mask(test_state, test_theta, test_s_vec))
        )

    def test_unsafe_mask2(self):
        """
        test_unsafe_mask2
        Description:
            Tests whether or not unsafe_mask correctly understands when an unsafe point is unsafe.
        """
        # Constants
        nominal_scenario = {}
        nominal_scenario[f"obstacle_0_center_x"] = 0.0
        nominal_scenario[f"obstacle_0_center_y"] = 0.0

        nominal_scenario[f"obstacle_1_center_x"] = 1.0
        nominal_scenario[f"obstacle_1_center_y"] = 0.0

        nominal_scenario["goal_x"] = 0.5
        nominal_scenario["goal_y"] = 0.5

        Theta = pc.box2poly([[0.0, 0.0], [1.0, 1.0]])

        # Create Object
        aps = AdaptivePusherSliderStickingForceInput_NObstacles(
            nominal_scenario, Theta,
        )

        # Test safe mask
        test_state = torch.tensor([[0.95, 0.0, 0.0]])
        test_theta = torch.tensor([[0.01, 0.03]])
        test_s_vec = torch.tensor(aps.scenario_to_list(nominal_scenario)).unsqueeze(0)

        self.assertTrue(
            torch.all(aps.unsafe_mask(test_state, test_theta, test_s_vec))
        )

    def test_goal_mask1(self):
        """
        test_goal_mask1
        Description:
            Verifies that the goal_point function correctly detects that a point is not close to the goal (for a given parameter)!
        """

        # Constants
        nominal_scenario = {}
        for ind in range(2):
            nominal_scenario[f"obstacle_{ind}_center_x"] = 0.0
            nominal_scenario[f"obstacle_{ind}_center_y"] = 0.0
            nominal_scenario[f"obstacle_{ind}_radius"] = 0.1

        nominal_scenario["goal_x"] = 0.5
        nominal_scenario["goal_y"] = 0.5

        Theta = pc.box2poly([[0.0, 0.0], [1.0, 1.0]])

        # Create Object
        aps = AdaptivePusherSliderStickingForceInput_NObstacles(
            nominal_scenario, Theta,
        )

        # Test safe mask
        test_state = torch.tensor([[0.5, 0.45, 0.0]])
        test_theta = torch.tensor([[0.01, 0.03]])
        test_s_vec = torch.tensor(aps.scenario_to_list(nominal_scenario)).unsqueeze(0)

        self.assertTrue(
            torch.all(aps.goal_mask(test_state, test_theta, test_s_vec))
        )

    def test_goal_point1(self):
        """
        test_goal_point1
        Description:
            Verifies that the goal_point function correctly detects that a point is not close to the goal (for a given parameter)!
        """

        # Constants
        nominal_scenario = {}
        for ind in range(2):
            nominal_scenario[f"obstacle_{ind}_center_x"] = 0.0
            nominal_scenario[f"obstacle_{ind}_center_y"] = 0.0
            nominal_scenario[f"obstacle_{ind}_radius"] = 0.1

        nominal_scenario["goal_x"] = 0.5
        nominal_scenario["goal_y"] = 0.5

        Theta = pc.box2poly([[0.0, 0.0], [1.0, 1.0]])

        # Create Object
        aps = AdaptivePusherSliderStickingForceInput_NObstacles(
            nominal_scenario, Theta,
        )

        # Test goal point
        test_state = torch.tensor([[0.5, 0.45, 0.0]])
        test_theta = torch.tensor([[0.01, 0.03]])
        test_s_vec = torch.tensor(aps.scenario_to_list(nominal_scenario)).unsqueeze(0)

        self.assertTrue(
            torch.equal(aps.goal_point(test_theta, test_s_vec), torch.tensor([[0.5, 0.5, 0.0]]))
        )

    def test_goal_point2(self):
        """
        test_goal_point2
        Description:
            Verifies that the goal_point function correctly detects that a point is not close to the goal (for a given parameter)!
            Uses multiple scenarios
        """

        # Constants
        nominal_scenario = {}
        for ind in range(2):
            nominal_scenario[f"obstacle_{ind}_center_x"] = 0.0
            nominal_scenario[f"obstacle_{ind}_center_y"] = 0.0
            nominal_scenario[f"obstacle_{ind}_radius"] = 0.1

        nominal_scenario["goal_x"] = 0.5
        nominal_scenario["goal_y"] = 0.5

        Theta = pc.box2poly([[0.0, 0.0], [1.0, 1.0]])

        # Create Object
        aps = AdaptivePusherSliderStickingForceInput_NObstacles(
            nominal_scenario, Theta,
        )

        # Test goal point
        test_state = torch.tensor([
            [0.5, 0.45, 0.0],
            [0.5, 0.65, 0.0]
        ])
        test_theta = torch.tensor([
            [0.01, 0.03],
            [0.04, 0.02],
        ])
        test_s_vec = torch.tensor([
            aps.scenario_to_list(nominal_scenario),
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.6],
        ])

        self.assertTrue(torch.equal(
                aps.goal_point(test_theta, test_s_vec),
                torch.tensor([[0.5, 0.5, 0.0], [0.5, 0.6, 0.0]])
        ))

    def test_sample_unsafe1(self):
        """
        test_sample_unsafe1
        Description:
            Verifies that the sample_unsafe function correctly detects that a point is not close to the goal (for a given parameter)!
            Uses multiple scenarios
        """

        # Constants
        nominal_scenario = {}
        for ind in range(2):
            nominal_scenario[f"obstacle_{ind}_center_x"] = 0.0
            nominal_scenario[f"obstacle_{ind}_center_y"] = 0.0
            nominal_scenario[f"obstacle_{ind}_radius"] = 0.1

        nominal_scenario["goal_x"] = 0.5
        nominal_scenario["goal_y"] = 0.5

        Theta = pc.box2poly([(0.0, 1.0), (0.0, 1.0)])

        # Create Object
        aps = AdaptivePusherSliderStickingForceInput_NObstacles(
            nominal_scenario, Theta,
        )

        # Test goal point
        n_samples = 100
        xth_samples, x_samples, th_samples, s_samples = aps.sample_unsafe(n_samples)

        self.assertEqual(xth_samples.shape, (100, aps.n_dims+aps.n_params))
        self.assertEqual(x_samples.shape, (100, aps.n_dims))

        self.assertEqual(
            sum(aps.unsafe_mask(x_samples, th_samples, s_samples)),
            n_samples,
        )

if __name__ == '__main__':
    unittest.main()
