"""
Test the simple load sharing example's object.
"""
from copy import copy

import matplotlib.pyplot as plt
import tqdm
import numpy as np
import torch

from neural_clbf.systems.adaptive import LoadSharingManipulator

import polytope as pc

def test_loadsharingmanipulator_init():
    """Test initialization of AutoRally model"""
    # Test instantiation with valid parameters
    valid_params = {
        "obstacle_center_x": 1.0,
        "obstacle_center_y": 1.0,
        "obstacle_center_z": 0.3,
        "obstacle_width": 1.0,
    }
    th_dim = 3
    A = np.vstack((np.eye(th_dim), -np.eye(th_dim)))
    b = np.ones(th_dim*2)
    Theta = pc.Polytope(A, b)
    # print(Theta)

    sys0 = LoadSharingManipulator(valid_params, Theta)

    assert sys0.n_dims == 6
    assert sys0.n_controls == 3
    assert sys0.n_params == th_dim

def test_loadsharingmanipulator_simulate_and_plot1():
    """
    Description:
        Simulates a few initial conditions of the LoadSharingManipulator using
        the same inputs. Then plots everything to an image.
    """

    # Create Pusher Slider
    scenario0 = {
        "obstacle_center_x": 1.0,
        "obstacle_center_y": 1.0,
        "obstacle_center_z": 0.3,
        "obstacle_width": 1.0,
    }
    th_dim = 3
    A = np.vstack((np.eye(th_dim), -np.eye(th_dim)))
    b = np.ones(th_dim * 2)
    Theta = pc.Polytope(A, b)
    sys0 = LoadSharingManipulator(scenario0, Theta)

    # Create initial condition options
    batch_size = 10
    x = torch.zeros((batch_size, sys0.n_dims))

    x[0, 0] = 0.5  # Control
    x[1, 1] = 0.5
    x[2, 2] = 0.5
    x[3, 0] = 1.0
    x[4, 1] = 1.0

    print("Initial conditions:")
    print(x)

    theta = sys0.get_N_samples_from_polytope(Theta, batch_size).T
    theta = torch.Tensor(theta)
    print("Parameter values:")
    print(theta)

    N_sim = 1000

    # Create silly controller
    def silly_controller(x_in: torch.Tensor, theta_in: torch.Tensor):
        # Constants
        n_batch = x_in.shape[0]
        n_controls = 3

        # Algorithm
        u = torch.zeros((n_batch, n_controls))
        u[:, 0] = 0.05
        u[:, 1] = 10*10

        return u

    # Simulate using the built-in function
    x_sim, th_sim, th_h_sim = sys0.simulate(x, theta, N_sim, silly_controller, 0.01)

    # Plot 1 (Projection onto 2d)
    fig, ax = plt.subplots(1, 1)
    list_of_lines = []
    list_of_line_labels = []
    for batch_i in range(batch_size):
        print(batch_i)
        print("x_sim = ", x_sim)
        temp_line = plt.plot(
            x_sim[batch_i, :, 0],
            x_sim[batch_i, :, 1]
        )

        # print(x_sim[batch_i, :, :])
        list_of_lines.append(temp_line)
        list_of_line_labels.append('Line #' + str(batch_i) )

    # Add legend
    ax.legend(list_of_lines, list_of_line_labels)

    fig.savefig("lsm-s_and_plot1.png")

    # Plot 3d
    #fig, ax = plt.subplots(1, 1)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(projection='3d')
    list_of_lines = []
    list_of_line_labels = []
    for batch_i in range(batch_size):
        ax2.scatter(
            np.array(x_sim[batch_i, :, 0].flatten()),
            np.array(x_sim[batch_i, :, 1].flatten()),
            np.array(x_sim[batch_i, :, 2].flatten()),
            marker='o'
        )

        # print(x_sim[batch_i, :, :])
        # list_of_lines.append(temp_line)
        list_of_line_labels.append('Line #' + str(batch_i))

    # Add legend
    #ax2.legend(list_of_lines, list_of_line_labels)

    fig2.savefig("lsm-s_and_plot2.png")

if __name__ == "__main__":
    # Test initialization
    test_loadsharingmanipulator_init()

    # Test simulate()
    test_loadsharingmanipulator_simulate_and_plot1()