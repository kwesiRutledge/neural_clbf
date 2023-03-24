"""Test some controller utilities"""
import unittest

import torch

from neural_clbf.controllers.controller_utils import normalize_with_angles, normalize_theta, normalize_theta_with_angles
from neural_clbf.systems.tests.mock_system import MockSystem
from neural_clbf.systems.adaptive.tests import MockCAPA2System

import polytope as pc

class utils_tester(unittest.TestCase):
    """Test some controller utilities"""
    def test_normalize_x(self):
        """Test the ability to normalize states"""
        # Define the model system
        params = {}
        system = MockSystem(params)

        # Define states on which to test.
        # Start with the upper and lower state limits
        x_upper, x_lower = system.state_limits
        x_upper = x_upper.unsqueeze(0)
        x_lower = x_lower.unsqueeze(0)

        # These should be normalized so that the first dimension becomes 1 and -1 (resp)
        # The second dimension is an angle and should be replaced with its sine and cosine
        x_upper_norm = normalize_with_angles(system, x_upper)
        assert torch.allclose(x_upper_norm[0, 0], torch.ones(1))
        assert torch.allclose(
            x_upper_norm[0, 1:],
            torch.tensor([torch.sin(x_upper[0, 1]), torch.cos(x_upper[0, 1])]),
        )
        x_lower_norm = normalize_with_angles(system, x_lower)
        assert torch.allclose(x_lower_norm[0, 0], -torch.ones(1))
        assert torch.allclose(
            x_lower_norm[0, 1:],
            torch.tensor([torch.sin(x_lower[0, 1]), torch.cos(x_lower[0, 1])]),
        )

        # Also test that the center of the range is normalized to zero
        x_center = 0.5 * (x_upper + x_lower)
        x_center_norm = normalize_with_angles(system, x_center)
        assert torch.allclose(x_center_norm[0, 0], torch.zeros(1))
        assert torch.allclose(
            x_center_norm[0, 1:],
            torch.tensor([torch.sin(x_center[0, 1]), torch.cos(x_center[0, 1])]),
        )

    def test_unittest_normalize_x2(self):
        """
        Description:
            Test the ability to normalize the parameters of the system using normalize_x.
            Tested on a ControlAffineParameterAffineSystem object.
        """
        # Constants
        params = {}
        Theta = pc.box2poly([(0, 1), (0, 1)])
        system_under_test = MockCAPA2System(params, Theta)

        # Define states on which to test.
        # Start with the upper and lower state limits
        x_upper, x_lower = system_under_test.state_limits
        x_upper = x_upper.unsqueeze(0)
        x_lower = x_lower.unsqueeze(0)

        # These should be normalized so that the first dimension becomes 1 and -1 (resp)
        # The second dimension is an angle and should be replaced with its sine and cosine
        x_upper_norm = normalize_with_angles(system_under_test, x_upper)
        assert torch.allclose(x_upper_norm[0, 0], torch.ones(1))
        assert torch.allclose(
            x_upper_norm[0, 1:],
            torch.tensor([torch.sin(x_upper[0, 1]), torch.cos(x_upper[0, 1])]),
        )
        x_lower_norm = normalize_with_angles(system_under_test, x_lower)
        assert torch.allclose(x_lower_norm[0, 0], -torch.ones(1))
        assert torch.allclose(
            x_lower_norm[0, 1:],
            torch.tensor([torch.sin(x_lower[0, 1]), torch.cos(x_lower[0, 1])]),
        )
    def test_unittest_normalize_theta1(self):
        """
        Description:
            Test the ability to normalize the parameters of the system using normalize_theta
        """
        # Constants
        params = {}
        Theta = pc.box2poly([(-2, 1), (-10, 1)])
        system_under_test = MockCAPA2System(params, Theta)

        # Define parameters on which to test.
        # Start with the vertices of the theta set
        V_Theta = pc.extreme(Theta)
        V_Theta_torch = torch.from_numpy(V_Theta).type_as(torch.ones(1))

        # ======================
        # Test vertex 1 [1, -10]

        # These should be normalized so that the first dimension becomes 1 and -1 (resp)
        # The second dimension is an angle and should be replaced with its sine and cosine
        v0_norm = normalize_theta_with_angles(system_under_test, V_Theta_torch[0, :].unsqueeze(0))

        assert torch.allclose(v0_norm[0, 1], -torch.ones(1)), "v0_norm[0, 1] ({}) != -1.0".format(v0_norm[0, 1])
        assert torch.allclose(
            v0_norm[0, (0, 2)],
            torch.tensor([torch.sin(V_Theta_torch[0, 0]), torch.cos(V_Theta_torch[0, 0])]),
        ), "v0_norm[0, (0, 2)] ({}) != [sin({}), cos({})]".format(v0_norm[0, (0, 2)], V_Theta[0][0], V_Theta[0][0])

        # =======================
        # Test Vertex 2 [1, 1]
        v1_norm = normalize_theta_with_angles(system_under_test, V_Theta_torch[1, :].unsqueeze(0))

        assert torch.allclose(v1_norm[0, 1], torch.ones(1)), "v1_norm[0, 1] ({}) != 1.0".format(v1_norm[0, 1])
        assert torch.allclose(
            v1_norm[0, (0, 2)],
            torch.tensor([torch.sin(V_Theta_torch[1, 0]), torch.cos(V_Theta_torch[1, 0])]),
        ), "v1_norm[0, (0, 2)] ({}) != [sin({}), cos({})]".format(v1_norm[0, (0, 2)], V_Theta[1][0], V_Theta[1][0])


        # =======================
        # Test Vertex 3 [-2, 1]
        v2_norm = normalize_theta_with_angles(system_under_test, V_Theta_torch[2, :].unsqueeze(0))

        assert torch.allclose(v2_norm[0, 1], torch.ones(1)), "v2_norm[0, 1] ({}) != 1.0".format(v2_norm[0, 1])
        assert torch.allclose(
            v2_norm[0, (0, 2)],
            torch.tensor([torch.sin(V_Theta_torch[2, 0]), torch.cos(V_Theta_torch[2, 0])]),
        ), "v2_norm[0, (0, 2)] ({}) != [sin({}), cos({})]".format(v2_norm[0, (0, 2)], V_Theta[2][0], V_Theta[2][0])

        # =======================
        # Test Vertex 4 [-2, -10]
        v3_norm = normalize_theta_with_angles(system_under_test, V_Theta_torch[3, :].unsqueeze(0))
        assert torch.allclose(v3_norm[0, 1], -torch.ones(1)), "v3_norm[0, 1] ({}) != -1.0".format(v3_norm[0, 1])
        assert torch.allclose(
            v3_norm[0, (0, 2)],
            torch.tensor([torch.sin(V_Theta_torch[3, 0]), torch.cos(V_Theta_torch[3, 0])]),
        ), "v3_norm[0, (0, 2)] ({}) != [sin({}), cos({})]".format(v3_norm[0, (0, 2)], V_Theta[3][0], V_Theta[3][0])

if __name__ == "__main__":
    unittest.main()
