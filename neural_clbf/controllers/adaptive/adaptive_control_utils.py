"""
adaptive_control_utils.py
Description:
    Contains some utility functions for adaptive controllers in this directory.
"""

import torch

from typing import List

def center_and_radius_to_vertices(
    center: torch.tensor,
    radius: torch.tensor,
) -> List[torch.tensor]:
    """
    center_and_radius_to_vertices
    Description:
        Converts a center and radius (defining a hyperrectangle) into the vertices defining that hyperrectangle.
    Args:
        center (torch.tensor): (bs, dim) tensor defining center of hyperrectangle
        radius (torch.tensor): (bs, dim) tensor defining "radii" (in each direction) of hyperrectangle in infinity norm sense
    """

    # Constnats
    batch_size = center.shape[0]
    dim = center.shape[1]
    n_Vertices = 2 ** dim

    # Create list
    vertices = []
    for vertex_index in range(n_Vertices):
        # Create binary representation of vertex index
        binnum_as_str = bin(vertex_index)[2:].zfill(dim)  # Convert number to binary string
        binnum = [float(digit) for digit in binnum_as_str]  # Convert to list of digits

        binnum_t = torch.tensor(binnum, dtype=torch.get_default_dtype())

        v_Theta = center.unsqueeze(2) + \
                  torch.bmm(torch.diag(binnum_t).repeat(batch_size, 1, 1), radius.unsqueeze(2)) - \
                  torch.bmm(torch.diag(1 - binnum_t).repeat(batch_size, 1, 1), radius.unsqueeze(2))

        vertices.append(v_Theta.squeeze(2))

    return vertices