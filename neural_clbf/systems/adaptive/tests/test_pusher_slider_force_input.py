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

from torch.distributions.uniform import Uniform

import matplotlib.pyplot as plt
import matplotlib.animation as manimation

from neural_clbf.systems.adaptive.adaptive_pusher_slider_force_input import AdaptivePusherSliderStickingForceInput


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

        ps = AdaptivePusherSliderStickingForceInput(
            nominal_scenario,
            Theta1,
        )
        x = torch.Tensor([0.1, 0.1, np.pi/3])

        th = ps.sample_Theta_space(1)
        th = torch.Tensor(th).flatten()

        p1 = plt.figure()  # This used to be created by plot, so we needed to check its type.
        ax = p1.add_subplot(111)
        ps.plot_single(x, th, ax, show_obstacle=False, show_goal=False)

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
        ps = AdaptivePusherSliderStickingForceInput(
            nominal_scenario,
            Theta1,
        )

        # Get Initial Condition and Parameter
        x = torch.Tensor([0.1, 0.1, np.pi / 6])
        th = ps.sample_Theta_space(1)
        th = torch.Tensor(th).flatten()

        # print(th)

        # Algorithm
        p2 = plt.figure()
        ax = p2.add_subplot(111)
        ps.plot_single(x, th, ax, hide_axes=False, show_obstacle=False, show_goal=False)

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
        ps = AdaptivePusherSliderStickingForceInput(
            nominal_scenario,
            Theta1,
        )

        # Get Initial Condition and Parameter
        x = torch.Tensor([0.1, 0.1, np.pi / 6])
        th = ps.sample_Theta_space(1)
        th = torch.Tensor(th).flatten()
        # print(th)

        f = torch.Tensor([0.1, 0.1])

        # Algorithm
        p3 = plt.figure()
        ax = p3.add_subplot(111)
        ps.plot_single(x, th, ax, hide_axes=False, current_force=f, show_obstacle=False, show_goal=False)

        if "/neural_clbf/systems/adaptive/tests" in os.getcwd():
            # Only save if we are running from inside tests directory
            p3.savefig("figures/pusherslider-test_plot3.png", dpi=300)

    def test_plot4(self):
        """
        test_plot4
        Description:
            Verifies that we can successfully plot a bunch of pusher-slider's simultaneously.
        Notes:
            This code was used to create images in Group Slides for Mon. March 20, 2023.
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
        ps = AdaptivePusherSliderStickingForceInput(
            nominal_scenario,
            Theta1,
        )

        # Get Initial Conditions and Parameter
        batch_size = 4
        x = torch.zeros((batch_size, ps.n_dims))

        x[0, :] = torch.Tensor([0.1, 0.1, 0.0])
        x[1, :] = torch.Tensor([-0.1, 0.1, 0.0])
        x[2, :] = torch.Tensor([-0.1, -0.1, 0.0])
        x[3, :] = torch.Tensor([0.1, -0.1, 0.0])

        th = torch.zeros((batch_size, ps.n_params))
        f = torch.zeros((batch_size, ps.n_controls))

        th[:, :] = ps.sample_Theta_space(batch_size)
        # print(th)

        f = torch.tensor([[-0.01, 0.1] for idx in range(batch_size)])

        # Algorithm
        limits = [[-0.3, 0.7], [-0.3, 0.3]]

        p4 = plt.figure()
        ax = p4.add_subplot(111)
        ps.plot(x, th,
                limits=limits,
                ax=ax, hide_axes=False, current_force=f,
                show_friction_cone_vectors=False,
                show_obstacle=False, show_goal=False,
                )

        goal_point = torch.tensor([0.5, 0.0])
        plt.scatter(
            goal_point[0], goal_point[1],
            marker="s",
        )

        if "/neural_clbf/systems/adaptive/tests" in os.getcwd():
            # Only save if we are running from inside tests directory
            p4.savefig("figures/pusherslider-test_plot4.png", dpi=300)

    def test_plot5(self):
        """
        test_plot5
        Description:
            Verifies that we can successfully plot a bunch of pusher-slider's
             simultaneously. We also show a "goal" point at the origin.
        Notes:
            This code was used to create images in Group Slides for Mon. March 20, 2023.
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
        ps = AdaptivePusherSliderStickingForceInput(
            nominal_scenario,
            Theta1,
        )

        # Get Initial Conditions and Parameter
        batch_size = 4
        x = torch.zeros((batch_size, ps.n_dims))

        x[0, :] = torch.Tensor([0.1, 0.1, 0.0])
        x[1, :] = torch.Tensor([-0.1, 0.1, 0.0])
        x[2, :] = torch.Tensor([-0.1, -0.1, 0.0])
        x[3, :] = torch.Tensor([0.1, -0.1, 0.0])

        th = torch.zeros((batch_size, ps.n_params))
        f = torch.zeros((batch_size, ps.n_controls))

        th[:, :] = ps.sample_Theta_space(batch_size)
        # print(th)

        f = torch.tensor([[-0.01, 0.1] for idx in range(batch_size)])

        # Algorithm
        limits = [[-0.3, 0.7], [-0.3, 0.3]]

        p4 = plt.figure()
        ax = p4.add_subplot(111)
        ps.plot(x, th,
                limits=limits,
                ax=ax, hide_axes=False, current_force=f,
                show_friction_cone_vectors=False,
                show_obstacle=False, show_goal=False,
                )

        goal_point = torch.tensor([0.0, 0.0])
        plt.scatter(
            goal_point[0], goal_point[1],
            marker="s",
        )

        if "/neural_clbf/systems/adaptive/tests" in os.getcwd():
            # Only save if we are running from inside tests directory
            p4.savefig("figures/pusherslider-test_plot5.png", dpi=300)

    def test_plot6(self):
        """
        test_plot5
        Description:
            Verifies that we can successfully plot a bunch of
            pusher-sliders simultaneously. We
        Notes:
            This code was used to create images in Group Slides for Mon. March 20, 2023.
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
        ps = AdaptivePusherSliderStickingForceInput(
            nominal_scenario,
            Theta1,
        )

        # Get Initial Conditions and Parameter
        batch_size = 1
        x = torch.zeros((batch_size, ps.n_dims))

        x[0, :] = torch.Tensor([-0.05, 0.06, 0.0])
        # x[1, :] = torch.Tensor([-0.1, 0.1, 0.0])
        # x[2, :] = torch.Tensor([-0.1, -0.1, 0.0])
        # x[3, :] = torch.Tensor([0.1, -0.1, 0.0])

        th = torch.zeros((batch_size, ps.n_params))
        f = torch.zeros((batch_size, ps.n_controls))

        th[:, :] = ps.sample_Theta_space(batch_size)
        # print(th)

        f = torch.tensor([[-0.01, 0.1] for idx in range(batch_size)])

        # Algorithm
        limits = [[-0.3, 0.7], [-0.3, 0.3]]

        p6 = plt.figure()
        ax = p6.add_subplot(111)
        ps.plot(x, th,
                limits=limits,
                ax=ax, hide_axes=False, current_force=f,
                show_friction_cone_vectors=False,
                show_obstacle=False, show_goal=False,
                )

        goal_point = torch.tensor([0.5, 0.5])
        plt.scatter(
            goal_point[0], goal_point[1],
            marker="s",
        )

        if "/neural_clbf/systems/adaptive/tests" in os.getcwd():
            # Only save if we are running from inside tests directory
            p6.savefig("figures/pusherslider-test_plot6.png", dpi=300)


    def test_plot7(self):
        """
        test_plot7
        Description:
            Verifies that we can successfully plot the obstacle along with the current state.

        """

        # Constants
        nominal_scenario = {
            "obstacle_center_x": 0.0,
            "obstacle_center_y": 0.0,
            "obstacle_radius": 0.05,
        }
        s_length = 0.09
        s_width = 0.09
        Theta1 = pc.box2poly(
            np.array([
                [-0.01, 0.01],  # CoM_x
                [-0.01 + (s_length / 2.0), 0.01 + (s_length / 2.0)]  # ub
            ])
        )
        ps = AdaptivePusherSliderStickingForceInput(
            nominal_scenario,
            Theta1,
        )

        # Get Initial Conditions and Parameter
        batch_size = 1
        x = torch.zeros((batch_size, ps.n_dims))

        x[0, :] = torch.Tensor([-0.1, 0.1, 0.0])
        # x[1, :] = torch.Tensor([-0.1, 0.1, 0.0])
        # x[2, :] = torch.Tensor([-0.1, -0.1, 0.0])
        # x[3, :] = torch.Tensor([0.1, -0.1, 0.0])

        th = torch.zeros((batch_size, ps.n_params))
        f = torch.zeros((batch_size, ps.n_controls))

        th[:, :] = ps.sample_Theta_space(batch_size)
        # print(th)

        f = torch.tensor([[-0.01, 0.1] for idx in range(batch_size)])

        # Algorithm
        # limits = [[-0.3, 0.7], [-0.3, 0.3]]

        p6 = plt.figure()
        ax = p6.add_subplot(111)
        ps.plot(x, th,
                ax=ax, hide_axes=False, current_force=f,
                show_friction_cone_vectors=False, show_goal=False,
                )

        goal_point = torch.tensor([0.0, 0.0])
        plt.scatter(
            goal_point[0], goal_point[1],
            marker="s",
        )

        if "/neural_clbf/systems/adaptive/tests" in os.getcwd():
            # Only save if we are running from inside tests directory
            p6.savefig("figures/pusherslider-test_plot7.png", dpi=300)

    def test_plot8(self):
        """
        test_plot8
        Description:
            Verifies that we can successfully plot:
            - the obstacle AND
            - the goal
            along with the current state.

        """

        # Constants
        nominal_scenario = {
            "obstacle_center_x": 0.0,
            "obstacle_center_y": 0.0,
            "obstacle_radius": 0.05,
        }
        s_length = 0.09
        s_width = 0.09
        Theta1 = pc.box2poly(
            np.array([
                [-0.01, 0.01],  # CoM_x
                [-0.01 + (s_length / 2.0), 0.01 + (s_length / 2.0)]  # ub
            ])
        )
        ps = AdaptivePusherSliderStickingForceInput(
            nominal_scenario,
            Theta1,
        )

        # Get Initial Conditions and Parameter
        batch_size = 1
        x = torch.zeros((batch_size, ps.n_dims))

        x[0, :] = torch.Tensor([-0.1, 0.1, 0.0])
        # x[1, :] = torch.Tensor([-0.1, 0.1, 0.0])
        # x[2, :] = torch.Tensor([-0.1, -0.1, 0.0])
        # x[3, :] = torch.Tensor([0.1, -0.1, 0.0])

        th = torch.zeros((batch_size, ps.n_params))
        f = torch.zeros((batch_size, ps.n_controls))

        th[:, :] = ps.sample_Theta_space(batch_size)
        # print(th)

        f = torch.tensor([[-0.01, 0.1] for idx in range(batch_size)])

        # Algorithm
        # limits = [[-0.3, 0.7], [-0.3, 0.3]]

        p6 = plt.figure()
        ax = p6.add_subplot(111)
        ps.plot(x, th,
                ax=ax, hide_axes=False, current_force=f,
                show_friction_cone_vectors=False,
                )

        goal_point = torch.tensor([0.0, 0.0])
        plt.scatter(
            goal_point[0], goal_point[1],
            marker="s",
        )

        if "/neural_clbf/systems/adaptive/tests" in os.getcwd():
            # Only save if we are running from inside tests directory
            p6.savefig("figures/pusherslider-test_plot8.png", dpi=300)

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
        ps = AdaptivePusherSliderStickingForceInput(
            nominal_scenario,
            Theta1,
        )

        # Create synthetic trajectory
        x0 = torch.Tensor([[0.1, 0.1, np.pi / 6]])
        N_traj = 100
        T_sim = 0.5
        x_trajectory = torch.Tensor(
            [[t, t, (np.pi/6)*t*3*2] for t in np.linspace(0, T_sim, N_traj+1)]
        )
        th = ps.sample_Theta_space(1)
        th = torch.Tensor(th).flatten()
        f0 = torch.Tensor([0.1, 0.0])

        # Animate
        fig = plt.figure()
        ax = fig.add_subplot(111)
        filename = "figures/pusherslider-test_animate1.mp4"
        num_frames = N_traj
        max_t = T_sim
        min_t = 0.0

        # print("th: ", th)

        # Plot the initial state.
        plot_objects = ps.plot_single(
            x0.flatten(), th, ax,
            hide_axes=False, current_force=f0,
            show_obstacle=False, show_goal=False,
        )

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
        ps = AdaptivePusherSliderStickingForceInput(
            nominal_scenario,
            Theta1,
        )

        # Create synthetic state, force trajectories
        x0 = torch.Tensor([[0.1, 0.1, np.pi / 6]])
        N_traj = 100
        T_sim = 0.5
        x_trajectory = torch.Tensor(
            [[t, t, (np.pi / 6) * t * 3 * 2] for t in np.linspace(0, T_sim, N_traj + 1)]
        ).T
        x_trajectory = x_trajectory.reshape(1, x_trajectory.shape[0], x_trajectory.shape[1])
        th = ps.sample_Theta_space(1)
        th = torch.Tensor(th).flatten()
        th = th.unsqueeze(0)

        f0 = torch.Tensor([0.1, 0.0])
        f_trajectory = torch.kron(torch.ones((N_traj+1, 1)), f0).T
        f_trajectory = f_trajectory.unsqueeze(0)

        # Animate with function
        if "/neural_clbf/systems/adaptive/tests" in os.getcwd():
            ps.save_animated_trajectory(
                x_trajectory=x_trajectory,
                th=th,
                f_trajectory=f_trajectory,
                hide_axes=False,
                filename="figures/pusherslider-test_animate2.mp4",
                show_obstacle=False, show_goal=False,
            )

    def test_animate3(self):
        """
        test_animate3
        Description:
            Tests the ability to simulate a trajectory using the closed-loop functions of pusher-slider.
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
        ps = AdaptivePusherSliderStickingForceInput(
            nominal_scenario,
            Theta1,
        )

        # Create synthetic state, force trajectories
        x0 = torch.Tensor([[0.1, 0.1, np.pi / 6]])
        N_traj = 100
        T_sim = 2.5
        dt = T_sim / N_traj
        x_trajectory = x0.T

        f_trajectory = torch.zeros((ps.n_controls, 0))

        U_dist = Uniform(ps.control_limits[1][0], ps.control_limits[0][0])
        theta = ps.sample_Theta_space(1)
        theta = torch.Tensor(theta).flatten()
        theta = theta.unsqueeze(0)
        for k in range(N_traj):
            x_k = x_trajectory[:, k].reshape(1, 3)
            u_k = torch.tensor(U_dist.sample((1, 2)))
            u_k[0, 1] = 7.0

            x_kp1 = x_k + ps.dt * ps.closed_loop_dynamics(
                x_k, u_k, theta.reshape((1, ps.n_params)))

            # Save data
            f_trajectory = torch.cat((f_trajectory, u_k.T), dim=1)
            x_trajectory = torch.cat((x_trajectory, x_kp1.T), dim=1)

        # Reshape
        x_trajectory = x_trajectory.reshape(1, x_trajectory.shape[0], x_trajectory.shape[1])
        f_trajectory = f_trajectory.reshape(1, f_trajectory.shape[0], f_trajectory.shape[1])

        # Animate with function
        if "/neural_clbf/systems/adaptive/tests" in os.getcwd():
            ps.save_animated_trajectory(
                x_trajectory=x_trajectory,
                th=theta,
                f_trajectory=f_trajectory,
                hide_axes=False,
                filename="figures/pusherslider-test_animate3.mp4",
                show_obstacle=False, show_goal=False,
            )

    def test_save_animated_trajectory1(self):
        """
        test_save_animated_trajectory1
        Description:
            Tests whether or not this function will properly save video of two pusher-slider systems at once.
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
        ps = AdaptivePusherSliderStickingForceInput(
            nominal_scenario,
            Theta1,
        )

        # Create synthetic state, force trajectories
        x0 = torch.Tensor([[0.1, 0.1, np.pi / 6]])
        N_traj = 100
        T_sim = 0.5
        batch_size = 2
        x_trajectory = torch.zeros((batch_size, ps.n_dims, N_traj+1))
        x_trajectory[0, :, :] = torch.Tensor(
            [[t, t, (np.pi / 6) * t * 3 * 2] for t in np.linspace(0, T_sim, N_traj + 1)]
        ).T
        x_trajectory[1, :, :] = torch.Tensor(
            [[t+0.5, t+0.5, (np.pi / 6) * t * 3 * 2] for t in np.linspace(0, T_sim, N_traj + 1)]
        ).T
        th = ps.sample_Theta_space(2)
        th = torch.Tensor(th).reshape(2, 2)

        f0 = torch.Tensor([0.1, 0.0])
        f_trajectory = torch.zeros((batch_size, ps.n_controls, N_traj+1))
        for batch_index in range(batch_size):
            f_trajectory[batch_index, :, :] = torch.kron(torch.ones((N_traj + 1, 1)), f0).T

        # Animate with function
        if "/neural_clbf/systems/adaptive/tests" in os.getcwd():
            ps.save_animated_trajectory(
                x_trajectory=x_trajectory,
                th=th,
                f_trajectory=f_trajectory,
                hide_axes=False,
                filename="figures/pusherslider-test_save_animated_trajectory1.mp4",
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
        ps = AdaptivePusherSliderStickingForceInput(
            nominal_scenario,
            Theta1,
        )

        # Create nominal input
        x0 = torch.Tensor([0.1, 0.1, np.pi/2])
        th = ps.sample_Theta_space(1)
        th = torch.Tensor(th)

        u_nom = ps.u_nominal(
            x0.reshape(1, AdaptivePusherSliderStickingForceInput.N_DIMS),
            th,
        )

        # Plot it
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ps.plot_single(
            x0, th.flatten(),
            ax,
            limits=[[0.0, 0.6], [0.0, 0.6]],
            hide_axes=False,
            current_force=u_nom.flatten(),
            show_obstacle=False,
            show_goal=False,
        )
        goal = ps.goal_point(th)
        # print("goal = ", goal)
        # print("goal[0, 0] = ", goal[0, 0])
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
        ps = AdaptivePusherSliderStickingForceInput(
            nominal_scenario,
            Theta1,
        )

        # Create nominal input
        x0 = torch.Tensor([0.6, 0.4, np.pi / 2])
        th = ps.sample_Theta_space(1)
        th = torch.Tensor(th)

        u_nom = ps.u_nominal(
            x0.reshape(1, AdaptivePusherSliderStickingForceInput.N_DIMS),
            th,
        )

        # Plot it
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ps.plot_single(
            x0, th.flatten(),
            ax,
            limits=[[0.4, 0.7], [0.3, 0.6]],
            hide_axes=False,
            current_force=u_nom.flatten(),
            show_goal=False,
        )
        goal = ps.goal_point(th)
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
        ps = AdaptivePusherSliderStickingForceInput(
            nominal_scenario,
            Theta1,
        )

        # Create nominal input
        x0 = torch.Tensor([0.52, 0.4, np.pi / 2])
        th = ps.sample_Theta_space(1)
        th = torch.Tensor(th)

        u_nom = ps.u_nominal(
            x0.reshape(1, AdaptivePusherSliderStickingForceInput.N_DIMS),
            th,
        )

        # Plot it
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ps.plot_single(
            x0, th.flatten(),
            ax,
            limits=[[0.4, 0.7], [0.3, 0.6]],
            hide_axes=False,
            current_force=u_nom.flatten(),
            show_obstacle=False,
            show_goal=False,
        )
        goal = ps.goal_point(th)
        plt.scatter(
            goal[0, 0], goal[0, 1],
            color="magenta",
        )

        if "/neural_clbf/systems/adaptive/tests" in os.getcwd():
            fig.savefig("figures/pusherslider-test_u_nominal3.png", dpi=300)
            # green is the current force

    def test_U1(self):
        """
        test_U1
        Description:
            Displays an example friction cone.

        """

        # Constants
        nominal_scenario = {
            "obstacle_center_x": 0.0,
            "obstacle_center_y": 0.0,
            "obstacle_radius": 0.05,
        }
        s_length = 0.09
        s_width = 0.09
        Theta1 = pc.box2poly(
            np.array([
                [-0.01, 0.01],  # CoM_x
                [-0.01 + (s_length / 2.0), 0.01 + (s_length / 2.0)]  # ub
            ])
        )
        ps = AdaptivePusherSliderStickingForceInput(
            nominal_scenario,
            Theta1,
        )

        # Create Friction Cone
        f_l, f_u = ps.friction_cone_extremes()

        # Algorithm
        H = np.array([
            [1.0, -ps.ps_cof],
            [-1.0, -ps.ps_cof],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ])
        # H[2, 0] = 0.0
        # H[2, 1] = 0.0

        h = np.zeros((5, 1))
        h[2, 0] = 2.0
        h[3, 0] = 2.0

        U1 = pc.Polytope(
            H, h,
        )

        p1 = plt.figure()
        ax = p1.add_subplot(111)

        # Plot Friction Cone Vectors
        plt.arrow(0.0, 0.0, f_l[0], f_l[1], color="red", width=0.001)
        plt.arrow(0.0, 0.0, f_u[0], f_u[1], color="red", width=0.001)

        U1.plot(ax, color="blue", alpha=0.5)

        assert U1.dim == 2, f"U1.dim = {U1.dim}; expected dimesnion 2"

        print("U1.bounding_box = ", U1.bounding_box)

        if "/neural_clbf/systems/adaptive/tests" in os.getcwd():
            # Only save if we are running from inside tests directory
            p1.savefig("figures/pusherslider-test_U1.png", dpi=300)

    def test_U2(self):
        """
        test_U2
        Description:
            Displays an example friction cone using the build in U() method.

        """

        # Constants
        nominal_scenario = {
            "obstacle_center_x": 0.0,
            "obstacle_center_y": 0.0,
            "obstacle_radius": 0.05,
        }
        s_length = 0.09
        s_width = 0.09
        Theta1 = pc.box2poly(
            np.array([
                [-0.01, 0.01],  # CoM_x
                [-0.01 + (s_length / 2.0), 0.01 + (s_length / 2.0)]  # ub
            ])
        )
        ps = AdaptivePusherSliderStickingForceInput(
            nominal_scenario,
            Theta1,
        )

        # Create Friction Cone
        f_l, f_u = ps.friction_cone_extremes()

        # Algorithm
        U2 = ps.U

        p1 = plt.figure()
        ax = p1.add_subplot(111)

        # Plot Friction Cone Vectors
        plt.arrow(0.0, 0.0, f_l[0], f_l[1], color="red", width=0.001)
        plt.arrow(0.0, 0.0, f_u[0], f_u[1], color="red", width=0.001)

        U2.plot(ax, color="blue", alpha=0.5)

        assert U2.dim == 2, f"U1.dim = {U2.dim}; expected dimesnion 2"

        print("U2.bounding_box = ", U2.bounding_box)
        print("ps.control_limits = ", ps.control_limits)

        if "/neural_clbf/systems/adaptive/tests" in os.getcwd():
            # Only save if we are running from inside tests directory
            p1.savefig("figures/pusherslider-test_U2.png", dpi=300)


if __name__ == '__main__':
    unittest.main()
