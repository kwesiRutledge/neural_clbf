"""
test_pusher_slider_force_input.py
Description:
    A collection of tests for the Pusher-Slider system that we've developed
    with a force input.
"""

import unittest
import torch
import numpy as np
import polytope as pc
import matplotlib.pyplot as plt

from neural_clbf.systems.adaptive.pusher_slider_force_input import PusherSliderStickingForceInput


class TestStringMethods(unittest.TestCase):

    """
    test_plot1
    Description:
        This method tests that the plot method returns a plot handle type of object
    """
    def test_plot1(self):
        nominal_scenario = {
            "obstacle_center_x": 0.0,
            "obstacle_center_y": 0.0,
            "obstacle_radius": 0.2,
        }
        Theta1 = pc.box2poly(
            np.array([
                [-0.3, -0.3],   # lb
                [0.3, 0.3]      # ub
            ])
        )

        ps = PusherSliderStickingForceInput(
            nominal_scenario,
            Theta1,
        )
        x = np.array([[0.1, 0.1, torch.pi/3]])

        th = ps.sample_Theta_space(1)
        th = torch.Tensor(th)

        p1, _ = ps.plot(x, th)

        self.assertEqual(type(p1), type(plt.figure()))

    """
    test_plot2
    Description:
        This method tests that the plot method returns a plot handle that we can save to a file.
        Show contact point
    """
    def test_plot2(self):
        # Constants
        nominal_scenario = {
            "obstacle_center_x": 0.0,
            "obstacle_center_y": 0.0,
            "obstacle_radius": 0.2,
        }
        s_length = 0.09
        s_width = 0.09
        Theta1 = pc.box2poly(
            np.array([
                [-0.03, 0.03],  # CoM_x
                [-0.03+(s_length/2), 0.03+(s_length/2)]  # CoM_y
            ])
        )
        ps = PusherSliderStickingForceInput(
            nominal_scenario,
            Theta1,
        )

        # Get Initial Condition and Parameter
        x = torch.Tensor([[0.1, 0.1, torch.pi / 6]])
        th = ps.sample_Theta_space(1)
        th = torch.Tensor(th)

        print(th)

        # Algorithm
        p2, _ = ps.plot(x, th, hide_axes=False)

        # Show contact point
        cp1 = ps.contact_point(x)
        plt.scatter(cp1[0, 0], cp1[0, 1], color="orange", s=10)

        p2.savefig("pusherslider-test_plot2.png", dpi=300)

    """
    test_plot3
    Description:
        Verifies that we can successfully plot a random force vector [fx, fy] in the plot.
    """
    def test_plot3(self):
        # Constants
        nominal_scenario = {
            "obstacle_center_x": 0.0,
            "obstacle_center_y": 0.0,
            "obstacle_radius": 0.2,
        }
        s_length = 0.09
        s_width = 0.09
        Theta1 = pc.box2poly(
            np.array([
                [-0.01, 0.01],  # CoM_x
                [-0.01+(s_length/2.0), 0.01+(s_length/2.0)]  # ub
            ])
        )
        ps = PusherSliderStickingForceInput(
            nominal_scenario,
            Theta1,
        )

        # Get Initial Condition and Parameter
        x = torch.Tensor([[0.1, 0.1, torch.pi / 6]])
        th = ps.sample_Theta_space(1)
        th = torch.Tensor(th)
        print(th)

        f = torch.Tensor([0.1, 0.1])

        # Algorithm
        p3, _ = ps.plot(x, th, hide_axes=False, current_force=f)

        p3.savefig("pusherslider-test_plot3.png", dpi=300)

    def test_animate1(self):
        # Constants
        nominal_scenario = {
            "obstacle_center_x": 0.0,
            "obstacle_center_y": 0.0,
            "obstacle_radius": 0.2,
        }
        s_length = 0.09
        s_width = 0.09
        Theta1 = pc.box2poly(
            np.array([
                [-0.01, 0.01],  # CoM_x
                [-0.01 + (s_length / 2.0), 0.01 + (s_length / 2.0)]  # ub
            ])
        )
        ps = PusherSliderStickingForceInput(
            nominal_scenario,
            Theta1,
        )

        # Create synthetic trajectory
        x0 = torch.Tensor([[0.1, 0.1, torch.pi / 6]])
        x_trajectory = torch.Tensor(
            [t, t, (torch.pi/6)*t] for t in np.linspace(0, 1, 100)
        )

        # Animate

        # This function will modify each of the values of the functions above.
        def update(frame_number):
            t = (frame_number / num_frames) * (max_t - min_t) + min_t
            # print(t)
            for i in range(num_agents):
                plan_i = pwl_plans[i]
                # print(plan_i)
                x_t = get_state_at_t(t, plan_i)
                team_positions_at_t[:, i] = x_t
                agent_circles[i].set(
                    center=x_t,
                )

            # If we want to show the team circle moving, then update it here
            if show_moving_team_radius:
                team_center_t = get_state_at_t(t, team_plan)
                team_circle.set(
                    center=team_center_t
                )

        # Construct the animation, using the update function as the animation
        # director.
        animation = manimation.FuncAnimation(fig, update, np.arange(1, num_frames), interval=25)
        animation.save(filename=filename, fps=15)

        # Algorithm
        for t in np.linspace(min_t,max_t,num_frames):
            fig_t = plot_single_frame_of_team_plan(
                t, pwl_plans=pwl_plans, team_plan=team_plan, plot_tuples=plot_tuples, team_radius=team_radius, size_list=size_list, equal_aspect=equal_aspect, limits=limits, show_team_plan=True        )

            writer.saving(fig_t,filename,20)
            writer.grab_frame()


    # def test_isupper(self):
    #     self.assertTrue('FOO'.isupper())
    #     self.assertFalse('Foo'.isupper())
    #
    # def test_split(self):
    #     s = 'hello world'
    #     self.assertEqual(s.split(), ['hello', 'world'])
    #     # check that s.split fails when the separator is not a string
    #     with self.assertRaises(TypeError):
    #         s.split(2)

if __name__ == '__main__':
    unittest.main()
