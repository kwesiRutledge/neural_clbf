"""
test_pusher_slider_force_input.py
Description:
    A collection of tests for the Pusher-Slider system that we've developed
    with a force input.
"""

import unittest, os
import torch
import numpy as np
import polytope as pc

import matplotlib.pyplot as plt
import matplotlib.animation as manimation

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
        x = torch.Tensor([0.1, 0.1, torch.pi/3])

        th = ps.sample_Theta_space(1)
        th = torch.Tensor(th).flatten()

        p1 = plt.figure()  # This used to be created by plot, so we needed to check its type.
        ps.plot(x, th)

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
        x = torch.Tensor([0.1, 0.1, torch.pi / 6])
        th = ps.sample_Theta_space(1)
        th = torch.Tensor(th).flatten()

        print(th)

        # Algorithm
        p2 = plt.figure()
        ps.plot(x, th, hide_axes=False)

        # Show contact point
        cp1 = ps.contact_point(x)
        plt.scatter(cp1[0, 0], cp1[0, 1], color="orange", s=10)

        if "/neural_clbf/systems/adaptive/tests" in os.getcwd():
            # Only save if we are running from inside tests directory
            p2.savefig("figures/pusherslider-test_plot2.png", dpi=300)

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
        x = torch.Tensor([0.1, 0.1, torch.pi / 6])
        th = ps.sample_Theta_space(1)
        th = torch.Tensor(th).flatten()
        print(th)

        f = torch.Tensor([0.1, 0.1])

        # Algorithm
        p3 = plt.figure()
        ps.plot(x, th, hide_axes=False, current_force=f)

        if "/neural_clbf/systems/adaptive/tests" in os.getcwd():
            # Only save if we are running from inside tests directory
            p3.savefig("figures/pusherslider-test_plot3.png", dpi=300)

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
        N_traj = 100
        T_sim = 0.5
        x_trajectory = torch.Tensor(
            [[t, t, (torch.pi/6)*t*3*2] for t in np.linspace(0, T_sim, N_traj+1)]
        )
        th = ps.sample_Theta_space(1)
        th = torch.Tensor(th).flatten()
        f0 = torch.Tensor([0.1, 0.0])

        # Animate
        fig = plt.figure()
        filename = "figures/pusherslider-test_animate1.mp4"
        num_frames = N_traj
        max_t = T_sim
        min_t = 0.0

        print("th: ", th)

        # Plot the initial state.
        _, plot_objects = ps.plot(x0.flatten(), th, hide_axes=False, current_force=f0)

        # This function will modify each of the values of the functions above.
        def update(frame_index):
            # print(t)
            x_t = x_trajectory[frame_index, :]
            f_t = f0

            ps.update_plot_objects(
                plot_objects,
                x_t.flatten(), th.flatten(),
                current_force=f0,
            )

        # Construct the animation, using the update function as the animation
        # director.
        animation = manimation.FuncAnimation(
            fig, update,
            np.arange(0, num_frames))#, interval=25)

        # Save as mp4. This requires mplayer or ffmpeg to be installed
        # plot.show()
        if "/neural_clbf/systems/adaptive/tests" in os.getcwd(): # Only plot if we're running this from inside tests directory.
            animation.save(filename=filename, fps=15)

        # Algorithm
        # for t in np.linspace(min_t,max_t,num_frames):
        #     fig_t = plot_single_frame_of_team_plan(
        #         t, pwl_plans=pwl_plans, team_plan=team_plan, plot_tuples=plot_tuples, team_radius=team_radius, size_list=size_list, equal_aspect=equal_aspect, limits=limits, show_team_plan=True        )
        #
        #     writer.saving(fig_t,filename,20)
        #     writer.grab_frame()

    """
    test_animate2
    Description:
        Creates a member function that animates a given state trajectory and
        force trajectory.
    """
    def test_animate2(self):
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

        # Create synthetic state, force trajectories
        x0 = torch.Tensor([[0.1, 0.1, torch.pi / 6]])
        N_traj = 100
        T_sim = 0.5
        x_trajectory = torch.Tensor(
            [[t, t, (torch.pi / 6) * t * 3 * 2] for t in np.linspace(0, T_sim, N_traj + 1)]
        )
        th = ps.sample_Theta_space(1)
        th = torch.Tensor(th).flatten()
        f0 = torch.Tensor([0.1, 0.0])
        f_trajectory = torch.kron(torch.ones((N_traj+1, 1)), f0)

        # Animate with function
        if "/neural_clbf/systems/adaptive/tests" in os.getcwd():
            ps.save_animated_trajectory(
                x_trajectory=x_trajectory,
                th=th,
                f_trajectory=f_trajectory,
                filename="figures/pusherslider-test_animate2.mp4",
            )

    def test_u_nominal1(self):
        """
        test_u_nominal1
        Description:
            Tests how well u_nominal works when the direction to the target point is outside
            of the friction cone.
        """
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

        # Create nominal input
        x0 = torch.Tensor([0.1, 0.1, torch.pi/2])
        th = ps.sample_Theta_space(1)
        th = torch.Tensor(th)

        u_nom = ps.u_nominal(
            x0.reshape(1, PusherSliderStickingForceInput.N_DIMS),
            th,
        )

        # Plot it
        fig = plt.figure()
        ps.plot(
            x0, th.flatten(),
            limits=[[0.0, 0.6], [0.0, 0.6]],
            hide_axes=False,
            current_force=u_nom.flatten(),
        )
        goal = ps.goal_point(th)
        print("goal = ", goal)
        print("goal[0, 0] = ", goal[0, 0])
        plt.scatter(
            goal[0, 0], goal[0, 1],
            color="magenta",
        )

        if "/neural_clbf/systems/adaptive/tests" in os.getcwd():
            fig.savefig("figures/pusherslider-test_u_nominal1.png", dpi=300)
            # green is the current force

    def test_u_nominal2(self):
        """
        test_u_nominal2
        Description:
            Tests how well u_nominal works when the direction to the target point is inside
            of the friction cone.
            Nominal should choose the force input f_l
        """
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

        # Create nominal input
        x0 = torch.Tensor([0.6, 0.4, torch.pi / 2])
        th = ps.sample_Theta_space(1)
        th = torch.Tensor(th)

        u_nom = ps.u_nominal(
            x0.reshape(1, PusherSliderStickingForceInput.N_DIMS),
            th,
        )

        # Plot it
        fig = plt.figure()
        ps.plot(
            x0, th.flatten(),
            limits=[[0.4, 0.7], [0.3, 0.6]],
            hide_axes=False,
            current_force=u_nom.flatten(),
        )
        goal = ps.goal_point(th)
        print("goal = ", goal)
        print("goal[0, 0] = ", goal[0, 0])
        plt.scatter(
            goal[0, 0], goal[0, 1],
            color="magenta",
        )

        if "/neural_clbf/systems/adaptive/tests" in os.getcwd():
            fig.savefig("figures/pusherslider-test_u_nominal2.png", dpi=300)
            # green is the current force

    def test_u_nominal3(self):
        """
        test_u_nominal3
        Description:
            Tests how well u_nominal works when the direction to the target point is inside
            of the friction cone.
            Nominal should choose the force vector that points from CoM to goal.
        """
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

        # Create nominal input
        x0 = torch.Tensor([0.52, 0.4, torch.pi / 2])
        th = ps.sample_Theta_space(1)
        th = torch.Tensor(th)

        u_nom = ps.u_nominal(
            x0.reshape(1, PusherSliderStickingForceInput.N_DIMS),
            th,
        )

        # Plot it
        fig = plt.figure()
        ps.plot(
            x0, th.flatten(),
            limits=[[0.4, 0.7], [0.3, 0.6]],
            hide_axes=False,
            current_force=u_nom.flatten(),
        )
        goal = ps.goal_point(th)
        print("goal = ", goal)
        print("goal[0, 0] = ", goal[0, 0])
        plt.scatter(
            goal[0, 0], goal[0, 1],
            color="magenta",
        )

        if "/neural_clbf/systems/adaptive/tests" in os.getcwd():
            fig.savefig("figures/pusherslider-test_u_nominal3.png", dpi=300)
            # green is the current force


if __name__ == '__main__':
    unittest.main()
