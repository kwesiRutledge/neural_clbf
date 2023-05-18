"""
test_adaptive_control_utils.py
"""

import torch

from neural_clbf.controllers.adaptive.adaptive_control_utils import (
    center_and_radius_to_vertices,
)

import unittest

class TestAdaptiveControlUtils(unittest.TestCase):

    def test_center_and_radius_to_vertices1(self):
        # Constants
        center = torch.tensor([0, 0], dtype=torch.get_default_dtype()).unsqueeze(0)
        radius = torch.tensor([1, 1], dtype=torch.get_default_dtype()).unsqueeze(0)

        # Expected
        expected_vertices = [
            torch.tensor([-1, -1], dtype=torch.get_default_dtype()).unsqueeze(0),
            torch.tensor([-1, 1], dtype=torch.get_default_dtype()).unsqueeze(0),
            torch.tensor([1, -1], dtype=torch.get_default_dtype()).unsqueeze(0),
            torch.tensor([1, 1], dtype=torch.get_default_dtype()).unsqueeze(0),
        ]

        # Actual
        actual_vertices = center_and_radius_to_vertices(center, radius)

        for vertex_index in range(len(expected_vertices)):
            # print("Vertex index: ", vertex_index)
            # print("Expected: ", expected_vertices[vertex_index])
            # print("Actual: ", actual_vertices[vertex_index])
            self.assertTrue(torch.allclose(expected_vertices[vertex_index], actual_vertices[vertex_index]))




if __name__ == '__main__':
    unittest.main()