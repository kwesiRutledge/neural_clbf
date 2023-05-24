"""
test_adaptive_pusher_slider_force_input_custom_goal.py
Description:
    Tests some of the unique methods that belong to the AdaptivePusherSliderForceInputCustomGoal class.
"""

import unittest

import numpy as np
import gurobipy as gp
import torch

import polytope as pc

from neural_clbf.systems.adaptive.adaptive_pusher_slider_force_input_custom_goal import (
    AdaptivePusherSliderStickingForceInput_CustomGoal,
)

class TestAPSFI_CustomGoal(unittest.TestCase):
    def test_estimate_goal1(self):
        # Constants

        # Create the model
        ps = self.retrieve_apsfi_cg1()

        # Call estimation script
        x0 = torch.tensor([
            [0.1, 0.1, torch.pi/4.0],
        ])
        theta_hat0 = torch.tensor([
            [0.0+0.09/2.0, 0.01, 1.0, -0.5],
        ])
        u_nom0 = torch.tensor([
            [1.0, 0.0],
        ])
        xg = ps.estimate_goal(
            x0, theta_hat0, u_nom0,
        )

        # Test that goal is in the Theta set
        theta_hat1 = theta_hat0.clone()
        theta_hat1[:, 2:] = xg

        self.assertTrue(ps.Theta.__contains__(theta_hat1[0, :].detach().numpy()))

    def retrieve_apsfi_cg1(self):

        # Define the scenarios
        nominal_scenario = {
            "obstacle_center_x": 0.0,
            "obstacle_center_y": 0.0,
            "obstacle_radius": 0.1,
        }

        scenarios = [
            nominal_scenario,
            # {"m": 1.25, "L": 1.0, "b": 0.01},  # uncomment to add robustness
            # {"m": 1.0, "L": 1.25, "b": 0.01},
            # {"m": 1.25, "L": 1.25, "b": 0.01},
        ]

        # Define the range of possible goal region centers
        s_width = 0.09
        eps0 = 0.05
        lb = [-0.03 + s_width/2.0, -0.03, 1.0-eps0, -1.0]
        ub = [ 0.03 + s_width/2.0,  0.03, 1.0+eps0, +1.0]
        Theta = pc.box2poly(np.array([lb, ub]).T)

        # Define the dynamics model
        dynamics_model = AdaptivePusherSliderStickingForceInput_CustomGoal(
            nominal_scenario,
            Theta,
            dt=0.01,
            controller_dt=0.01,
            scenarios=scenarios,
        )

        return dynamics_model

if __name__ == "__main__":
    unittest.main()