"""
adaptive_control_utils.py
Description:
    Contains some utility functions for adaptive controllers in this directory.
"""

import torch

import cvxpy as cp
import numpy as np
from cvxpylayers.torch import CvxpyLayer

from typing import List

from neural_clbf.systems.adaptive import ControlAffineParameterAffineSystem

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

def define_set_valued_estimator_cvxpylayer1(
    dynamics: ControlAffineParameterAffineSystem,

):
    """
    define_set_valued_estimator_cvxpylayer
    Description:
        Defines a set-valued estimator using cvxpylayer.
    """

    # Constants
    n_params = dynamics.n_params
    n_dims = dynamics.n_dims

    # Define variables
    theta_ub = cp.Variable((n_params,))
    theta_lb = cp.Variable((n_params,))

    F_plus_Gu = cp.Parameter((n_dims, n_params))
    D_minus_extras1 = cp.Parameter((n_dims,))
    D_minus_extras2 = cp.Parameter((n_dims,))
    theta = cp.Parameter((n_params,))

    # Create the constraints
    constraints = []
    constraints.append(
        - F_plus_Gu @ theta_ub <= D_minus_extras1
    )
    constraints.append(
        F_plus_Gu @ theta_ub <= D_minus_extras2
    )
    constraints.append(
        dynamics.Theta.A @ theta_ub <= dynamics.Theta.b,
    )

    constraints.append(
        - F_plus_Gu @ theta_lb <= D_minus_extras1
    )
    constraints.append(
        F_plus_Gu @ theta_lb <= D_minus_extras2
    )
    constraints.append(
        dynamics.Theta.A @ theta_lb <= dynamics.Theta.b,
    )

    # Create the objective
    objective_expression = np.ones((1, n_params)) @ theta_ub - np.ones((1, n_params)) @ (theta_lb)
    objective = cp.Maximize(objective_expression)

    # Create the problem
    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp(), f"Expected problem to follow the Disciplined Parametric Programming rules, but it doesn't."

    variables = [theta_ub, theta_lb]
    parameters = [F_plus_Gu, D_minus_extras1, D_minus_extras2]

    # Create the cvxpylayer
    return CvxpyLayer(problem, variables=variables, parameters=parameters)
