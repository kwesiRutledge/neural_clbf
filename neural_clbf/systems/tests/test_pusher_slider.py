"""
test_pusher_slider.py
Description:
    Tests the hybrid pusher-slider dynamics written in the form of the new
    HybridControlAffineSystem.
"""
from copy import copy

import matplotlib.pyplot as plt
import tqdm
import numpy as np
import torch

from neural_clbf.systems import PusherSlider


def test_PusherSlider_init():
    """Test initialization of StickingPusherSlider model"""
    # Test instantiation with valid parameters
    scenario = {
        "s_x_ref": 1.0,
        "s_y_ref": 1.0,
        "bar_radius": 0.15,
    }
    ps0 = PusherSlider(scenario)
    assert ps0 is not None

def test_PusherSlider_f_all1():
    """
    Description:
        Test the method f_all() for the PusherSlider object.
        Gives a nonzero input but should produce a zero output.
    """

    # Constants
    valid_params = {
        "s_x_ref": 1.0,
        "s_y_ref": 1.0,
    }
    ps0 = PusherSlider(valid_params)

    # Create a set of data
    batch_size = 2
    x = torch.zeros((batch_size, ps0.n_dims))
    x[0, 2] = 0.2

    u = torch.ones( (batch_size, ps0.n_controls) )


    # Compute _f_all()
    fa_out1 = ps0._f_all(x, u, valid_params)

    assert torch.all(torch.isclose(fa_out1, torch.zeros((batch_size, ps0.n_dims, ps0.n_modes))))

def test_PusherSlider_g_all1():
    """
    Description:
        Test the method f_all() for the PusherSlider object.
        Gives a nonzero input but should produce a zero output.
    """

    # Constants
    valid_params = {
        "s_x_ref": 1.0,
        "s_y_ref": 1.0,
    }
    ps0 = PusherSlider(valid_params)

    # Create a set of data
    batch_size = 4
    x = torch.zeros((batch_size, ps0.n_dims))
    x[0, 2] = 0.2

    u = torch.ones((batch_size, ps0.n_controls))

    # Compute _g_all()
    ga_out1 = ps0._g_all(x, u, valid_params)

    assert not torch.all(torch.isclose(ga_out1, torch.zeros((batch_size, ps0.n_dims, ps0.n_controls, ps0.n_modes))))

def test_PusherSlider_simulate_and_plot1():
    """
    Description:
        Simulates a few initial conditions of the PusherSlider using
        the same inputs. Then plots everything to an image.
    """

    # Create Pusher Slider
    valid_params = {
        "s_x_ref": 1.0,
        "s_y_ref": 1.0,
    }
    ps0 = PusherSlider(valid_params)

    # Create initial condition options
    batch_size = 10
    x = torch.zeros((batch_size, ps0.n_dims))

    x[0, 3] = 0.0  # Control
    x[1, 3] = 0.005
    x[2, 3] = -0.005
    x[3, 3] = 0.01
    x[4, 3] = -0.01

    print("Initial conditions:")
    print(x)

    # Create silly controller
    def silly_controller(x_in: torch.Tensor):
        # Constants
        n_batch = x_in.shape[0]

        # Algorithm
        u = torch.zeros((n_batch, 2))
        u[:, 0] = 0.05

        return u

    # Simulate using the built-in function
    x_sim = ps0.simulate(x, 100, silly_controller, 0.01)

    # Plot
    fig, ax = plt.subplots(1, 1)
    list_of_lines = []
    list_of_line_labels = []
    for batch_i in range(5): #range(batch_size):
        temp_line, = plt.plot(
            x_sim[batch_i, :, 0],
            x_sim[batch_i, :, 1]
        )

        # print(x_sim[batch_i, :, :])
        list_of_lines.append(temp_line)
        list_of_line_labels.append('Line #' + str(batch_i) )

    # Add legend
    ax.legend(list_of_lines, list_of_line_labels)

    fig.savefig("s_and_plot1.png")

if __name__ == "__main__":
    # Test initialization
    test_PusherSlider_init()
    # Test _f_all()
    test_PusherSlider_f_all1()
    # Test _g_all()
    test_PusherSlider_g_all1()

    # Test simulate()
    test_PusherSlider_simulate_and_plot1()

