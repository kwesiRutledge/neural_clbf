"""
test_apfsi3.py
Description:
    Tests the new AdaptivePusherSliderForceInput3 with parametric goal position and NO polytopes embedded inside.
"""

import torch
import numpy as np

import unittest
import os

import polytope as pc
import matplotlib.pyplot as plt

from neural_clbf.systems.adaptive_w_scenarios import AdaptivePusherSliderStickingForceInput3

class TestAPSFI3(unittest.TestCase):
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

        V_Theta = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])

        # Create Object
        aps = AdaptivePusherSliderStickingForceInput3(
            nominal_scenario, V_Theta,
        )

        self.assertEqual(aps.n_scenario, 2*aps.n_obstacles+2)

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

        V_Theta = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])

        # Create Object
        aps = AdaptivePusherSliderStickingForceInput3(
            nominal_scenario, V_Theta,
        )

        V_scenario = aps.scenario_set_vertices

        # Test
        self.assertEqual(V_scenario.shape[1], aps.n_obstacles*2+2)

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

        V_Theta = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])

        # Create Object
        aps = AdaptivePusherSliderStickingForceInput3(
            nominal_scenario, V_Theta,
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

        V_Theta = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])

        # Create Object
        aps = AdaptivePusherSliderStickingForceInput3(
            nominal_scenario, V_Theta,
            n_obstacles=2,
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

        V_Theta = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])

        # Create Object
        aps = AdaptivePusherSliderStickingForceInput3(
            nominal_scenario, V_Theta,
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

        V_Theta = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])

        # Create Object
        aps = AdaptivePusherSliderStickingForceInput3(
            nominal_scenario, V_Theta,
            n_obstacles=2,
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

        V_Theta = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])

        # Create Object
        aps = AdaptivePusherSliderStickingForceInput3(
            nominal_scenario, V_Theta,
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

        V_Theta = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])

        # Create Object
        aps = AdaptivePusherSliderStickingForceInput3(
            nominal_scenario, V_Theta,
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

        V_Theta = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])

        # Create Object
        aps = AdaptivePusherSliderStickingForceInput3(
            nominal_scenario, V_Theta,
            n_obstacles=2,
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

        V_Theta = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])

        # Create Object
        aps = AdaptivePusherSliderStickingForceInput3(
            nominal_scenario, V_Theta,
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

    def test_plot10(self):
        """
        test_plot10
        Description:
            Verifies that we can successfully plot:
            - the obstacle AND
            - the goal
            along with the current state.
        Notes:
            Used on an update meeting on April 3, 2023.
        """

        # Constants
        nominal_scenario = {
            "obstacle_0_center_x": 0.1,
            "obstacle_0_center_y": 0.3,
            "goal_x": 0.5,
            "goal_y": 0.5,
        }
        s_length = 0.09
        s_width = 0.09
        Theta1 = pc.box2poly(
            np.array([
                [-0.01, 0.01],  # CoM_x
                [-0.01 + (s_length / 2.0), 0.01 + (s_length / 2.0)]  # ub
            ])
        )
        V_Theta = torch.tensor(pc.extreme(Theta1))

        ps = AdaptivePusherSliderStickingForceInput3(
            nominal_scenario, V_Theta,
            n_obstacles=1,
        )

        # Get Initial Conditions and Parameter
        batch_size = 1
        x = torch.zeros((batch_size, ps.n_dims))

        x[0, :] = torch.tensor([-0.8, -0.8, torch.pi/4])
        # x[1, :] = torch.Tensor([-0.1, 0.1, 0.0])
        # x[2, :] = torch.Tensor([-0.1, -0.1, 0.0])
        # x[3, :] = torch.Tensor([0.1, -0.1, 0.0])

        th = torch.zeros((batch_size, ps.n_params))
        f = torch.zeros((batch_size, ps.n_controls))

        th[:, :] = ps.sample_Theta_space(batch_size)
        # print(th)

        f = torch.tensor([[-0.01, 0.1] for idx in range(batch_size)])

        # Algorithm
        # limits = [[-0.3, 0.7], [-0.3, 0.3]]

        p9 = plt.figure()
        ax = p9.add_subplot(111)
        ps.plot(x, th, torch.tensor(ps.scenario_to_list(nominal_scenario)).reshape(1, -1),
                ax=ax, hide_axes=False, current_force=f,
                show_obstacle=True,
                show_friction_cone_vectors=False,
                limits=[[-1.0, 1.0], [-1.0, 1.0]],
                )

        goal_point = torch.tensor([0.0, 0.0])
        plt.scatter(
            goal_point[0], goal_point[1],
            marker="s",
        )

        if "/neural_clbf/systems/adaptive_w_scenarios/tests" in os.getcwd():
            # Only save if we are running from inside tests directory
            p9.savefig("figures/pusher-slider-test_plot10.png", dpi=300)

if __name__ == '__main__':
    unittest.main()
