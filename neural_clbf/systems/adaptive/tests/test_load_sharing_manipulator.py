"""
Test the simple load sharing example's object.
"""
from copy import copy

import matplotlib.pyplot as plt
import tqdm
import numpy as np
import torch, os
import osqp
import scipy as sp
from scipy import sparse

from neural_clbf.systems.adaptive import LoadSharingManipulator

import polytope as pc

import unittest


class TestLoadSharingManipulator(unittest.TestCase):
    def test_loadsharingmanipulator_init(self):
        """
        Description:
            Test initialization of LoadSharingManipulator model object.
            Should correctly be initialized.
        """
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

        self.assertEqual(sys0.n_dims, 6)
        self.assertEqual(sys0.n_controls, 3)
        self.assertEqual(sys0.n_params, th_dim)

    def test_loadsharingmanipulator_simulate_and_plot1(self):
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

        # print("Initial conditions:")
        # print(x)

        theta = sys0.sample_Theta_space(batch_size)
        # print("Parameter values:")
        # print(theta)

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
            # print(batch_i)
            # print("x_sim = ", x_sim)
            temp_line = plt.plot(
                x_sim[batch_i, :, 0],
                x_sim[batch_i, :, 1]
            )

            # print(x_sim[batch_i, :, :])
            list_of_lines.append(temp_line)
            list_of_line_labels.append('Line #' + str(batch_i) )

        # Add legend
        ax.legend(list_of_lines, list_of_line_labels)

        if "/neural_clbf/systems/adaptive/tests" in os.getcwd():
            # Only save if we are running from inside tests directory
            fig.savefig("figures/lsm-s_and_plot1.png")

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

        if "/neural_clbf/systems/adaptive/tests" in os.getcwd():
            # Only save if we are running from inside tests directory
            fig2.savefig("figures/lsm-s_and_plot2.png")

    def test_loadsharingmanipulator_u_nominal1(self):
        """
        Description:
            Simulates a single initial condition of the LoadSharingManipulator using
            and uses u_nominal. Should lead to convergence to a target point.
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
        sys0 = LoadSharingManipulator(
            scenario0, Theta,
        )

        # Create initial condition options
        batch_size = 1
        x = torch.zeros((batch_size, sys0.n_dims))

        x[0, 0] = 0.5  # Control

        theta = sys0.sample_Theta_space(batch_size)

        N_sim = 1000

        # Simulate using the built-in function
        x_sim, th_sim, th_h_sim = sys0.simulate(
            x, theta,
            N_sim,
            sys0.u_nominal,
        )

        # Plot 3d
        #fig, ax = plt.subplots(1, 1)
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(1, 2, 1, projection='3d')
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

        ax2.scatter(
            theta[0, 0], theta[0, 1], theta[0, 2],
            marker='x', color='red', s=100,
        )

        ax3 = fig2.add_subplot(1, 2, 2)
        ax3.plot(
            torch.arange(0, N_sim*sys0.dt, sys0.dt),
            torch.norm(x_sim[:, :, :3] - theta, dim=2).flatten(),
        )
        ax3.set_title("Norm of error from target")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Norm of error")

        if "/neural_clbf/systems/adaptive/tests" in os.getcwd():
            # Only save if we are running from inside tests directory
            fig2.savefig("figures/lsm-u_nominal1.1.png")

        fig3, axs = plt.subplots(1, 3)
        for u_index in range(sys0.n_controls):
            axs[u_index].plot(
                torch.arange(0, N_sim*sys0.dt, sys0.dt),
                sys0.u_nominal(
                    x_sim[0, :, :].reshape(N_sim, sys0.n_dims),
                    theta[0, :].repeat((N_sim, 1)).reshape(N_sim, sys0.n_params),
                )[:, u_index].flatten(),
            )

        # Add legend
        #ax2.legend(list_of_lines, list_of_line_labels)

        if "/neural_clbf/systems/adaptive/tests" in os.getcwd():
            # Only save if we are running from inside tests directory
            fig3.savefig("figures/lsm-u_nominal1.2.png")


        # Verify that simulated trajectory is close to target
        x_target = theta
        self.assertLess(
            torch.min(
                torch.norm(x_sim[:, :, :3] - theta, dim=2)
            ),
            sys0.goal_tolerance,
        )

    def test_loadsharingmanipulator_plot_environment1(self):
        """
        Description:
            Tests that we can plot a proper empty environment.
        """

        # Create Pusher Slider
        scenario0 = {
            "obstacle_center_x": 0.2,
            "obstacle_center_y": 0.1,
            "obstacle_center_z": 0.3,
            "obstacle_width": 0.2,
        }
        th_dim = 3

        Theta = pc.box2poly(
            np.array([[-0.15, 0.15], [0.4, 0.45], [0.1, 0.3]])
        )
        sys0 = LoadSharingManipulator(
            scenario0, Theta,
        )

        # Create initial condition options
        batch_size = 1
        x = torch.zeros((batch_size, sys0.n_dims))

        x[0, 0] = 0.5  # Control

        theta = sys0.sample_Theta_space(batch_size)
        theta = theta.reshape((1, sys0.n_params))

        # Plot environment
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        sys0.plot_environment(ax, theta, show_goal=False)

        ax.scatter(
            0.3, -0.3, 0.4,
            marker='o', color='blue', s=100,
        )

        ax.view_init(azim=45, elev=15)

        # plt.show()

        if "/neural_clbf/systems/adaptive/tests" in os.getcwd():
            # Only save if we are running from inside tests directory
            fig.savefig("figures/lsm-plot_environment1.png")


    def test_loadsharingmanipulator_basic_mpc1_1(self):
        """
        Description:
            Tests that we can do MPC on a simple system.
        """
        # Discrete time model of a quadcopter
        Ad = sparse.csc_matrix([
            [1., 0., 0., 0., 0., 0., 0.1, 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0., 0., 0.1, 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0., 0., 0., 0.1, 0., 0., 0.],
            [0.0488, 0., 0., 1., 0., 0., 0.0016, 0., 0., 0.0992, 0., 0.],
            [0., -0.0488, 0., 0., 1., 0., 0., -0.0016, 0., 0., 0.0992, 0.],
            [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.0992],
            [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
            [0.9734, 0., 0., 0., 0., 0., 0.0488, 0., 0., 0.9846, 0., 0.],
            [0., -0.9734, 0., 0., 0., 0., 0., -0.0488, 0., 0., 0.9846, 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.9846]
        ])
        Bd = sparse.csc_matrix([
            [0., -0.0726, 0., 0.0726],
            [-0.0726, 0., 0.0726, 0.],
            [-0.0152, 0.0152, -0.0152, 0.0152],
            [-0., -0.0006, -0., 0.0006],
            [0.0006, 0., -0.0006, 0.0000],
            [0.0106, 0.0106, 0.0106, 0.0106],
            [0, -1.4512, 0., 1.4512],
            [-1.4512, 0., 1.4512, 0.],
            [-0.3049, 0.3049, -0.3049, 0.3049],
            [-0., -0.0236, 0., 0.0236],
            [0.0236, 0., -0.0236, 0.],
            [0.2107, 0.2107, 0.2107, 0.2107]])
        [nx, nu] = Bd.shape

        # Constraints
        u0 = 10.5916
        umin = np.array([9.6, 9.6, 9.6, 9.6]) - u0
        umax = np.array([13., 13., 13., 13.]) - u0
        xmin = np.array([-np.pi / 6, -np.pi / 6, -np.inf, -np.inf, -np.inf, -1.,
                         -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
        xmax = np.array([np.pi / 6, np.pi / 6, np.inf, np.inf, np.inf, np.inf,
                         np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])

        # Objective function
        Q = sparse.diags([0., 0., 10., 10., 10., 10., 0., 0., 0., 5., 5., 5.])
        QN = Q
        R = 0.1 * sparse.eye(4)

        # Initial and reference states
        x0 = np.zeros(12)
        xr = np.array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

        # Prediction horizon
        N = 10

        # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
        # - quadratic objective
        P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN,
                               sparse.kron(sparse.eye(N), R)], format='csc')
        # - linear objective
        q = np.hstack([np.kron(np.ones(N), -Q.dot(xr)), -QN.dot(xr),
                       np.zeros(N * nu)])
        # - linear dynamics
        Ax = sparse.kron(sparse.eye(N + 1), -sparse.eye(nx)) + sparse.kron(sparse.eye(N + 1, k=-1), Ad)
        Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd)
        Aeq = sparse.hstack([Ax, Bu])
        leq = np.hstack([-x0, np.zeros(N * nx)])
        ueq = leq
        # - input and state constraints
        Aineq = sparse.eye((N + 1) * nx + N * nu)
        lineq = np.hstack([np.kron(np.ones(N + 1), xmin), np.kron(np.ones(N), umin)])
        uineq = np.hstack([np.kron(np.ones(N + 1), xmax), np.kron(np.ones(N), umax)])
        # - OSQP constraints
        A = sparse.vstack([Aeq, Aineq], format='csc')
        l = np.hstack([leq, lineq])
        u = np.hstack([ueq, uineq])

        # Create an OSQP object
        prob = osqp.OSQP()

        # Setup workspace
        prob.setup(P, q, A, l, u, warm_start=True)

        # Simulate in closed loop
        nsim = 15
        for i in range(nsim):
            # Solve
            res = prob.solve()

            # Check solver status
            if res.info.status != 'solved':
                raise ValueError('OSQP did not solve the problem!')

            self.assertEqual(res.info.status, 'solved')

            # Apply first control input to the plant
            ctrl = res.x[-N * nu:-(N - 1) * nu]
            x0 = Ad.dot(x0) + Bd.dot(ctrl)

            # Update initial state
            l[:nx] = -x0
            u[:nx] = -x0
            prob.update(l=l, u=u)

    def test_loadsharingmanipulator_basic_mpc1_2(self):
        """
        test_loadsharingmanipulator_basic_mpc1_2
        Description:
            Tests that we can do MPC on a simple system.
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
        sys0 = LoadSharingManipulator(scenario0, Theta, dt=0.025)

        # Discrete time model of a quadcopter
        Ad, Bd = sys0.linearized_dt_dynamics_matrices()
        [nx, nu] = Bd.shape

        x0 = np.zeros(nx)
        x0_t = torch.tensor(x0).unsqueeze(0)
        theta_hat_t = sys0.sample_Theta_space(1)
        theta_hat = theta_hat_t.reshape((th_dim,)).numpy()

        fd_t = sys0._f(
            x0_t,
            sys0.u_nominal(x0_t, theta_hat_t),
        ) * sys0.dt
        fd = fd_t.reshape((nx,)).numpy()
        print("fd =", fd)

        # Constraints
        u0 = 0.5916
        # umin = 200*np.array([-1.3, -1.3, -1.6]) - u0
        # umax = 200*np.array([1.3, 1.3, 1.3]) - u0
        umax_t, umin_t = sys0.control_limits
        umax, umin = umax_t.numpy(), umin_t.numpy()
        xmax_t, xmin_t = sys0.state_limits
        xmax, xmin = xmax_t.numpy(), xmin_t.numpy()

        # Objective function
        Q = sparse.diags([10., 10., 10., 1., 1., 1.])
        QN = Q
        R = 0.1 * sparse.eye(nu)

        # Initial and reference states
        x0 = np.zeros((nx,))
        xr = np.array([0., 0.5, 0.5, 0., 0., 0.])

        # Prediction horizon
        N = 10

        # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
        # - quadratic objective
        P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN,
                               sparse.kron(sparse.eye(N), R)], format='csc')
        # - linear objective
        q = np.hstack([np.kron(np.ones(N), -Q.dot(xr)), -QN.dot(xr),
                       np.zeros(N * nu)])
        # - linear dynamics
        Ax = sparse.kron(sparse.eye(N + 1), -sparse.eye(nx)) + sparse.kron(sparse.eye(N + 1, k=-1), Ad)
        Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd)
        Aeq = sparse.hstack([Ax, Bu])
        # leq = np.hstack([-x0, np.zeros(N * nx)])
        leq = np.hstack([-x0, np.kron(np.ones(N), -fd)])
        ueq = leq
        # - input and state constraints
        Aineq = sparse.eye((N + 1) * nx + N * nu)
        lineq = np.hstack([np.kron(np.ones(N + 1), xmin), np.kron(np.ones(N), umin)])
        uineq = np.hstack([np.kron(np.ones(N + 1), xmax), np.kron(np.ones(N), umax)])
        # - OSQP constraints
        A = sparse.vstack([Aeq, Aineq], format='csc')
        l = np.hstack([leq, lineq])
        u = np.hstack([ueq, uineq])

        # Create an OSQP object
        prob = osqp.OSQP()

        # Setup workspace
        prob.setup(P, q, A, l, u, warm_start=True)

        # Simulate in closed loop
        nsim = 15
        for i in range(nsim):
            # Solve
            res = prob.solve()

            print(i)

            print(res.info)
            print(res.info.status)

            print("xmax = ", xmax, "xmin = ", xmin)
            print("Aeq", Aeq)

            # Check solver status
            if res.info.status != 'solved':
                raise ValueError('OSQP did not solve the problem!')

            self.assertEqual(res.info.status, 'solved')

            # Apply first control input to the plant
            ctrl = res.x[-N * nu:-(N - 1) * nu]
            x0 = Ad.dot(x0) + Bd.dot(ctrl) + fd

            # Update initial state
            l[:nx] = -x0
            u[:nx] = -x0
            prob.update(l=l, u=u)

    def test_loadsharingmanipulator_sample_polytope_center1(self):
        """
        test_loadsharingmanipulator_sample_polytope_center1
        Description:
            This test verifies that the center of the polytope (as sampled) is in the right place)
        """

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
        sys0 = LoadSharingManipulator(scenario0, Theta, dt=0.025)

        center = sys0.sample_polytope_center(Theta)

        self.assertAlmostEqual(
            np.linalg.norm(center - np.array([0.0, 0.0, 0.0])),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
