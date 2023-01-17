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
    x_sim = sys0.simulate(x, theta, N_sim, silly_controller, 0.01)

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

if __name__ == "__main__":
    # Test initialization
    test_scalardemocapa2system_init()

    # Test simulate()
    test_scalardemocapa2system_simulate_and_plot1()