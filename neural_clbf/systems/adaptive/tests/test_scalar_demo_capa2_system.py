"""
test_scalar_demo_capa2_system.py
Description:
    Test the simple load sharing example's object.
"""
from copy import copy

import matplotlib.pyplot as plt
import tqdm
import numpy as np
import torch

from neural_clbf.systems.adaptive import ScalarCAPA2Demo

import polytope as pc

import cvxpy as cp

def test_scalardemocapa2system_init():
    """Test initialization of AutoRally model"""
    # Test instantiation with valid parameters
    valid_params = {
        "wall_position": -2.0,
    }
    th_dim = 1

    lb = [0.5]
    ub = [0.8]
    Theta = pc.box2poly(np.array([lb, ub]).T)
    # print(Theta)

    sys0 = ScalarCAPA2Demo(valid_params, Theta)

    assert sys0.n_dims == 1
    assert sys0.n_controls == 1
    assert sys0.n_params == th_dim

def test_scalardemocapa2system_simulate_and_plot1():
    """
    Description:
        Simulates a few initial conditions of the LoadSharingManipulator using
        the same inputs. Then plots everything to an image.
    """

    # Create Pusher Slider
    scenario0 = {
        "wall_position": -2.0,
    }

    th_dim = 1
    lb = [0.5]
    ub = [0.8]
    Theta = pc.box2poly(np.array([lb, ub]).T)
    # print(Theta)

    sys0 = ScalarCAPA2Demo(scenario0, Theta)

    # Create initial condition options
    batch_size = 10
    x = torch.zeros((batch_size, sys0.n_dims))

    x[0, 0] = 0.25  # Control
    x[1, 0] = 0.5
    x[2, 0] = 0.75
    x[3, 0] = 1.0
    x[4, 0] = 1.5

    print("Initial conditions:")
    print(x)

    theta = sys0.get_N_samples_from_polytope(Theta, batch_size).T
    theta = torch.Tensor(theta)
    print("Parameter values:")
    print(theta)

    N_sim = 1000

    # Create silly controller
    def silly_controller(x_in: torch.Tensor):
        # Constants
        n_batch = x_in.shape[0]
        n_controls = 1

        # Algorithm
        u = torch.zeros((n_batch, n_controls))
        u[:, 0] = 0.05

        return u

    # Simulate using the built-in function
    x_sim, th_sim, th_h_sim = sys0.simulate(x, theta, N_sim, silly_controller, 0.01)

    # Plot 1 (Projection onto 2d)
    fig, ax = plt.subplots(1, 1)
    list_of_lines = []
    list_of_line_labels = []
    for batch_i in range(batch_size):
        temp_line, = plt.plot(
            x_sim[batch_i, :, 0]
        )

        # print(x_sim[batch_i, :, :])
        list_of_lines.append(temp_line)
        list_of_line_labels.append('Line #' + str(batch_i) )

    # Add legend
    ax.legend(list_of_lines, list_of_line_labels)

    fig.savefig("demo-capa2-s_and_plot1.png")

def test_scalardemocapa2system_basic_mpc1_1():
    """
    test_scalardemocapa2system_basic_mpc1_1
    Description
        Tests the method basic_mpc1.
        Uses the funciton logic here for easy access/verification.
    """

    print("Starting test_scalardemocapa2system_basic_mpc1_1()...")

    # Create System
    scenario0 = {
        "wall_position": -2.0,
    }

    th_dim = 1
    lb = [0.5]
    ub = [0.8]
    Theta = pc.box2poly(np.array([lb, ub]).T)
    # print(Theta)

    sys0 = ScalarCAPA2Demo(scenario0, Theta)

    # Constants
    x = torch.Tensor([
        [0.5], [0.8], [1.0]
    ])
    dt = 0.01
    U = pc.box2poly([[-1.0, 1.0]])
    N_mpc: int = 5

    # Input Processing

    # Constants
    n_controls = sys0.n_controls
    n_dims = sys0.n_dims
    S_w, S_u, S_x0 = sys0.get_mpc_matrices(N_mpc)

    U_T_A = np.kron(U.A, np.eye(N_mpc))
    U_T_b = np.kron(U.b, np.ones(N_mpc))
    U_T = pc.Polytope(U_T_A, U_T_b)

    assert U_T.dim == n_controls * N_mpc

    # Solve the MPC problem for each element of the batch
    batch_size = x.shape[0]
    u = torch.zeros(batch_size, n_controls).type_as(x)
    goal_x = np.array([[0.0]])
    for batch_idx in range(batch_size):
        batch_x = x[batch_idx, :n_dims].cpu().detach().numpy()

        # Create input for this batch
        u_T = cp.Variable((n_controls * N_mpc,))

        # Define objective as being the distance from state at t_0 + N_MPC to
        # the goal.
        M_T = np.zeros((n_dims, n_dims * (N_mpc)))
        M_T[:, -n_dims:] = np.eye(n_dims)
        obj = cp.norm(M_T @ (S_u @ u_T + np.dot(S_x0, batch_x)) - goal_x)
        obj += cp.norm(u_T)

        constraints = [U_T.A @ u_T <= U_T.b]

        # Solve for the P with largest volume
        prob = cp.Problem(
            cp.Minimize(obj), constraints
        )
        prob.solve()
        # Skip if no solution
        assert prob.status == "optimal"
        # if prob.status != "optimal":
        #     continue

        print(u_T.value)


if __name__ == "__main__":
    # Test initialization
    test_scalardemocapa2system_init()

    # Test simulate()
    test_scalardemocapa2system_simulate_and_plot1()

    # Test MPC1
    test_scalardemocapa2system_basic_mpc1_1()