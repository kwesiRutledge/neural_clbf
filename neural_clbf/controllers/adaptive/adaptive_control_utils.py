"""
adaptive_control_utils.py
Description:
    Contains some utility functions for adaptive controllers in this directory.
"""

import torch
import torch.nn.functional as F

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

        binnum_t = torch.tensor(binnum, dtype=torch.get_default_dtype()).to(center.device)

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

def linearized_clf_value_with_jacobians_uncertain(
    dynamics: ControlAffineParameterAffineSystem,
    x: torch.Tensor,
    theta_hat: torch.Tensor,
    theta_err_hat: torch.Tensor,
)->torch.Tensor:
    """
    linearized_clf_value_with_jacobians_unc
    Description:
        Computes the linearized Control Lyapunov Function value at the current (x,theta_hat,theta_err_hat)
        for the uncertainty aware CLF.
    """
    # constants
    batch_size = x.shape[0]
    n_dims = dynamics.n_dims
    n_params = dynamics.n_params

    # Compute the linearized CLF value

    # Create batches of x-theta pairs
    x_theta_therr = torch.cat([x, theta_hat, theta_err_hat], dim=1).to(x.device)
    x_theta0 = torch.zeros(
        x_theta_therr.shape,
        device=x.device,
    ).type_as(x_theta_therr)
    x0 = dynamics.goal_point(theta_hat).type_as(x_theta_therr)
    theta_hat0 = torch.tensor(
        dynamics.sample_polytope_center(dynamics.Theta),
        dtype=torch.get_default_dtype(),
    ).to(x.device).type_as(x_theta_therr).repeat(batch_size, 1)
    theta_err_hat0 = torch.zeros(
        theta_err_hat.shape,
        device=x.device,
    ).type_as(x_theta_therr)

    x_theta0[:, :dynamics.n_dims] = x0
    x_theta0[:, dynamics.n_dims:dynamics.n_dims + dynamics.n_params] = theta_hat0
    x_theta0[:, dynamics.n_dims + dynamics.n_params:] = theta_err_hat0

    # First, get the Lyapunov function value and gradient at this state
    Px = dynamics.P.type_as(x_theta_therr)
    Ptherr = torch.eye(dynamics.n_params, device=x.device).type_as(x_theta_therr)
    # Reshape to use pytorch's bilinear function
    P = torch.zeros(
        1, n_dims + 2 * n_params, n_dims + 2 * n_params,
    ).to(x.device)
    P[0, :dynamics.n_dims, :dynamics.n_dims] = Px
    P[0, n_dims:n_dims + n_params, n_dims:n_dims + n_params] = torch.zeros(
        (dynamics.n_params, dynamics.n_params),
        device=x.device,
    )
    P[0, n_dims + n_params:, n_dims + n_params:] = Ptherr

    Va = 0.5 * F.bilinear(x_theta_therr - x_theta0, x_theta_therr - x_theta0, P).squeeze()
    Va = Va.reshape(batch_size)

    # Reshape again for the gradient calculation
    P = P.reshape(n_dims + 2 * n_params, n_dims + 2 * n_params)
    JxV = F.linear(x - x0, P[:n_dims, :n_dims].T) + \
          F.linear(theta_hat - theta_hat0, P[n_dims:n_dims + n_params, :n_dims].T)
    JxV = JxV.reshape(batch_size, 1, dynamics.n_dims)

    JthV = F.linear(theta_hat - theta_hat0, P[n_dims:n_dims + n_params, n_dims:n_dims + n_params].T) + \
           F.linear(x - x0, P[:n_dims, n_dims:n_dims + n_params].T)
    JthV = JthV.reshape(batch_size, 1, n_params)

    JtherrV = F.linear(theta_err_hat - theta_err_hat0, P[n_dims+n_params:, n_dims + n_params:].T) + \
           F.linear(x - x0, P[:n_dims, n_dims + n_params:].T)

    return Va, JxV, JthV, JtherrV

def linearized_clf_value_uncertain(
    dynamics: ControlAffineParameterAffineSystem,
    x: torch.Tensor,
    theta_hat: torch.Tensor,
    theta_err_hat: torch.Tensor,
)->torch.Tensor:
    # Constants

    # Algorithm
    Va, _, _, _ = linearized_clf_value_with_jacobians_uncertain(
        dynamics,
        x, theta_hat, theta_err_hat,
    )

    return Va

def violation_weighting_function1(
    dynamics: ControlAffineParameterAffineSystem,
    x: torch.Tensor,
    theta_hat: torch.Tensor,
    scale_factor: float = 1e2,
) -> (torch.Tensor):
    # Constants

    # Algorithm 1: Weight according to distance to goal
    x0 = dynamics.goal_point(theta_hat).type_as(x)
    delta0 = torch.norm(x - x0, dim=1)

    # Higher weight to points closer to the goal
    w = scale_factor * torch.exp(-delta0)

    return w

