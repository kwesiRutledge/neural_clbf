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
from neural_clbf.systems.adaptive_w_scenarios import ControlAffineParameterAffineSystem2

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
    """
    violation_weighting_function1
    """
    # Constants

    # Algorithm 1: Weight according to distance to goal
    x0 = dynamics.goal_point(theta_hat, scen).type_as(x)
    delta0 = torch.norm(x - x0, dim=1)

    # Higher weight to points closer to the goal
    w = scale_factor * torch.exp(-delta0)

    return w

def violation_weighting_function2(
    dynamics: ControlAffineParameterAffineSystem2,
    x: torch.Tensor,
    theta_hat: torch.Tensor,
    scen: torch.Tensor,
    scale_factor: float = 1e2,
) -> (torch.Tensor):
    """
    violation_weighting_function2
    Args:
        scen: bs x self.dynamics_model.n_scenario the observed scenario of the current environment
    """
    # Constants

    # Algorithm 1: Weight according to distance to goal
    x0 = dynamics.goal_point(theta_hat, scen).type_as(x)
    delta0 = torch.norm(x - x0, dim=1)

    # Higher weight to points closer to the goal
    w = scale_factor * torch.exp(-delta0)

    return w

def nominal_lyapunov_function(
    dynamics: ControlAffineParameterAffineSystem2,
    x: torch.Tensor,
    theta_hat: torch.Tensor,
    scen: torch.Tensor,
):
    """
    Description:
        Computes the nominal Lyapunov function for the given dynamics model.
    """
    # Constants
    batch_size = x.shape[0]

    n_dims = dynamics.n_dims
    n_params = dynamics.n_params

    # Create batches of x-theta pairs
    x_theta = torch.cat([x, theta_hat], dim=1).to(x.device)
    x_theta0 = torch.zeros(x_theta.shape).type_as(x_theta).to(x.device)
    x0 = dynamics.goal_point(theta_hat, scen).type_as(x_theta).to(x.device)
    theta_hat0 = dynamics.sample_Theta_space(1).type_as(x_theta).repeat(batch_size, 1)

    x_theta0[:, :n_dims] = x0
    x_theta0[:, n_dims:] = theta_hat0

    # First, get the Lyapunov function value and gradient at this state
    Px = dynamics.P.type_as(x_theta)
    # Reshape to use pytorch's bilinear function
    P = torch.zeros(1, n_dims + n_params, n_dims + n_params).to(x.device)
    P[0, :n_dims, :n_dims] = Px
    P[0, n_dims:, n_dims:] = torch.zeros((n_params, n_params))

    Va = 0.5 * F.bilinear(x_theta - x_theta0, x_theta - x_theta0, P).squeeze()
    Va = Va.reshape(batch_size)

    return Va

def create_grid_of_feasible_convex_combinations_of_V(n_V: int, n_steps: int = 10)->torch.Tensor:
    """
    create_grid_of_feasible_convex_combinations_of_V
    Description:
        Creates a matrix where each vector represents a convex combination vector (i.e. each row sums to 1, or is on the simplex).
        The convex combination of n_V vertices can be used to quickly sample over the entire space of the polytope defined by
        the vertices in V (not used).
    Returns:
        grid_as_matrix: (n_V - 1, n_steps ** 2) matrix where each row is a convex combination vector
    """
    # Constants

    # Grid input space
    ranges = []
    for vertex_index in range(n_V - 1):
        ranges.append(
            torch.linspace(0, 1.0, n_steps),
        )

    grid_tuple = torch.meshgrid(
        *ranges,
        indexing='xy',
    )

    # Create grid as a single matrix
    grid_as_matrix = torch.zeros(
        (n_V - 1, n_steps ** 2),
    )
    for vertex_index in range(n_V - 1):
        grid_as_matrix[vertex_index, :] = grid_tuple[vertex_index].flatten().squeeze()

    sum = torch.sum(grid_as_matrix, dim=0, keepdim=True)
    grid_as_matrix = torch.vstack(
        (grid_as_matrix, float(n_V) - sum),
    )

    grid_as_matrix = grid_as_matrix / float(n_V)

    return grid_as_matrix

def create_uniform_samples_across_polytope(
    V: torch.Tensor,
    n_grid: int,
)->torch.Tensor:
    """
    samples = controller.create_uniform_samples_across_polytope(V, n_samples)
    Args:
        V: torch.Tensor
            A n_vertices x n tensor of vertices of a polytope.
        n_grid: int
            The number of samples to create along each direction of the n_vertices grid.
            (in other words, the samples will be created by gridding up a hypercube in
             n_vertices-dimensional space. Then each of those points will become a convex
             combination vector that combines the vertices in V.)

    """
    # Constants
    n_vertices = V.shape[0]

    grid_as_matrix = create_grid_of_feasible_convex_combinations_of_V(
        n_vertices,
        n_grid,
    )

    return V.T @ grid_as_matrix