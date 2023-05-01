"""
test_traj_opt2_safety_experiment.py
Description:
    Test some of the basic methods of the new class CaseStudySafetyExperimentTrajOpt2.
"""

import unittest

import numpy as np
import torch
import polytope as pc

import control as ct
import control.optimal as opt
import scipy.optimize as optimize

import matplotlib.pyplot as plt

import os

from neural_clbf.experiments.adaptive.safety_case_study import (
    CaseStudySafetyExperimentTrajOpt2,
)
from neural_clbf.systems.adaptive import LoadSharingManipulator

class TestTrajOpt2SafetyExperiment(unittest.TestCase):
    """
    TestTrajOpt2SafetyExperiment
    Description:
        Test the CaseStudySafetyExperimentTrajOpt2 class.
    """
    def test_synthesize_trajectories1(self):
        """
        test_synthesize_trajectories1
        Descripiton:
            Run the synthesis functions.
        """

        N_timepts = 10

        # Create Load Sharing Manipulator
        scenario0 = {
            "obstacle_center_x": 0.0,
            "obstacle_center_y": 0.0,
            "obstacle_center_z": 0.3,
            "obstacle_width": 0.1,
        }
        th_dim = 3
        A = np.vstack((np.eye(th_dim), -np.eye(th_dim)))
        b = np.ones(th_dim * 2)
        Theta = pc.Polytope(A, b)
        lsm0 = LoadSharingManipulator(scenario0, Theta)

        theta_hat0 = lsm0.sample_Theta_space(1).reshape((th_dim,)).numpy()

        # Create Case Study Safety Experiment
        x_start = torch.tensor(
            [
                [0.3, -0.3, 0.4, 0.0, 0.0, 0.0],
                # [0.3, -0.3, 0.4, 0.1, 0.0, 0.0],
                # [0.3, -0.3, 0.3, 0.0, 0.0, 0.0],
            ]
        )

        experiment0 = CaseStudySafetyExperimentTrajOpt2(
            "CS-TestTrajOpt2", x_start,
            n_sims_per_start=1,
            plot_x_indices=[LoadSharingManipulator.P_X, LoadSharingManipulator.P_Y, LoadSharingManipulator.P_Z],
            plot_x_labels=["$p_x$", "$p_y$", "$p_z$"],
        )

        # Set Up Inputs to run_trajopt_with_synthesis
        dynamics = lsm0
        Q = None
        R = None
        P = None
        u0 = None
        uf = None
        Tf = 10.0

        obs_center = np.array([
            lsm0.nominal_scenario["obstacle_center_x"],
            lsm0.nominal_scenario["obstacle_center_y"],
            lsm0.nominal_scenario["obstacle_center_z"]
        ])

        constraints = []
        constraints.append(
            (
                optimize.NonlinearConstraint,
                lambda x, u: np.linalg.norm(x[0:3] - obs_center),
                lsm0.nominal_scenario["obstacle_width"] / 2.0,
                float('Inf'),
            )
        )

        def dynamics_update(t, x, u, params):
            """
            dxdt = dynamics_update(t,x,u, params)
            Description:
                This function defines the dynamics of the system.
            """
            # Constants
            m = lsm0.m
            gravity = 9.81

            theta = params.get("theta", theta_hat0)

            # Unpack the state
            p = x[0:3]
            v = x[3:6]

            # Algorithm
            f = np.zeros((6,))
            f[0:3] = v
            f[3:6] = (1.0 / m) * np.diag([lsm0.K_x, lsm0.K_y, lsm0.K_z]) @ p
            f[-1] = f[-1] - gravity

            F = (1.0 / m) * np.vstack(
                    (np.zeros((3, lsm0.n_controls)), np.diag([lsm0.K_x, lsm0.K_y, lsm0.K_z]))
                )

            g = (1.0 / m) * np.vstack(
                (np.zeros((3, lsm0.n_controls)), np.eye(lsm0.n_controls))
            )

            return f + F @ theta + g @ u

        control_sequences, theta_samples, traj_opt_times, t, state_sequences = experiment0.synthesize_trajectories(
            dynamics, dynamics_update,
            Tf = 5.0,
            u0=u0, uf=uf,
            constraints=constraints,
            Q=Q, R=R, P=P,
            N_timepts=N_timepts,
        )

        self.assertEqual(control_sequences.shape, (1, N_timepts, lsm0.n_controls))
        self.assertEqual(theta_samples.shape, (1, th_dim))
        self.assertEqual(state_sequences.shape, (1, N_timepts, lsm0.n_dims))

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)

        ax1.plot(
            state_sequences[0, :, LoadSharingManipulator.P_X].numpy(),
            state_sequences[0, :, LoadSharingManipulator.P_Y].numpy(),
        )

        goal = lsm0.goal_point(
            torch.tensor(theta_samples)
        ).type_as(state_sequences)
        ax1.scatter(
            goal[0, LoadSharingManipulator.P_X].numpy(),
            goal[0, LoadSharingManipulator.P_Y].numpy(),
            color="red",
            marker="x",
        )

        if "/neural_clbf/experiments/adaptive/tests" in os.getcwd():
            # Only save if we are running from inside tests directory
            fig1.savefig("figures/trajopt2-test-synth.png")

    def test_synthesize_trajectories2(self):
        """
        test_synthesize_trajectories2
        Descripiton:
            Run the synthesis functions around an obstacle.
        """

        N_timepts = 10

        # Create Load Sharing Manipulator
        scenario0 = {
            "obstacle_center_x": 0.0,
            "obstacle_center_y": 0.0,
            "obstacle_center_z": 0.3,
            "obstacle_width": 0.1,
        }
        th_dim = 3
        A = np.vstack((np.eye(th_dim), -np.eye(th_dim)))
        b = np.ones(th_dim * 2)
        Theta = pc.Polytope(A, b)
        lsm0 = LoadSharingManipulator(scenario0, Theta)

        theta_hat0 = lsm0.sample_Theta_space(1).reshape((th_dim,)).numpy()

        # Create Case Study Safety Experiment
        x_start = torch.tensor(
            [
                [-0.3, -0.3, 0.4, 0.0, 0.0, 0.0],
                # [0.3, -0.3, 0.4, 0.1, 0.0, 0.0],
                # [0.3, -0.3, 0.3, 0.0, 0.0, 0.0],
            ]
        )

        experiment0 = CaseStudySafetyExperimentTrajOpt2(
            "CS-TestTrajOpt2", x_start,
            n_sims_per_start=1,
            plot_x_indices=[LoadSharingManipulator.P_X, LoadSharingManipulator.P_Y, LoadSharingManipulator.P_Z],
            plot_x_labels=["$p_x$", "$p_y$", "$p_z$"],
        )

        # Set Up Inputs to run_trajopt_with_synthesis
        dynamics = lsm0
        Q = None
        R = None
        P = None
        u0 = None
        uf = None
        Tf = 10.0

        obs_center = np.array([
            lsm0.nominal_scenario["obstacle_center_x"],
            lsm0.nominal_scenario["obstacle_center_y"],
            lsm0.nominal_scenario["obstacle_center_z"]
        ])

        constraints = []
        constraints.append(
            (
                optimize.NonlinearConstraint,
                lambda x, u: np.linalg.norm(x[0:3] - obs_center),
                lsm0.nominal_scenario["obstacle_width"] / 2.0,
                float('Inf'),
            )
        )

        def dynamics_update(t, x, u, params):
            """
            dxdt = dynamics_update(t,x,u, params)
            Description:
                This function defines the dynamics of the system.
            """
            # Constants
            m = lsm0.m
            gravity = 9.81

            theta = params.get("theta", theta_hat0)

            # Unpack the state
            p = x[0:3]
            v = x[3:6]

            # Algorithm
            f = np.zeros((6,))
            f[0:3] = v
            f[3:6] = (1.0 / m) * np.diag([lsm0.K_x, lsm0.K_y, lsm0.K_z]) @ p
            f[-1] = f[-1] - gravity

            F = (1.0 / m) * np.vstack(
                (np.zeros((3, lsm0.n_controls)), np.diag([lsm0.K_x, lsm0.K_y, lsm0.K_z]))
            )

            g = (1.0 / m) * np.vstack(
                (np.zeros((3, lsm0.n_controls)), np.eye(lsm0.n_controls))
            )

            return f + F @ theta + g @ u

        control_sequences, theta_samples, traj_opt_times, t, state_sequences = experiment0.synthesize_trajectories(
            dynamics, dynamics_update,
            Tf=5.0,
            u0=u0, uf=uf,
            constraints=constraints,
            Q=Q, R=R, P=P,
            N_timepts=N_timepts,
        )

        self.assertEqual(control_sequences.shape, (1, N_timepts, lsm0.n_controls))
        self.assertEqual(theta_samples.shape, (1, th_dim))
        self.assertEqual(state_sequences.shape, (1, N_timepts, lsm0.n_dims))

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)

        ax1.plot(
            state_sequences[0, :, LoadSharingManipulator.P_X].numpy(),
            state_sequences[0, :, LoadSharingManipulator.P_Y].numpy(),
        )

        goal = lsm0.goal_point(
            torch.tensor(theta_samples)
        ).type_as(state_sequences)
        ax1.scatter(
            goal[0, LoadSharingManipulator.P_X].numpy(),
            goal[0, LoadSharingManipulator.P_Y].numpy(),
            color="red",
            marker="x",
        )

        if "/neural_clbf/experiments/adaptive/tests" in os.getcwd():
            # Only save if we are running from inside tests directory
            fig1.savefig("figures/trajopt2-test-synth2.png")

    def test_run1(self):
        """
        test_run1
        Description:
            Run the synthesis function AND test for a single
            initial condition.
        """

        N_timepts = 10

        # Create Load Sharing Manipulator
        scenario0 = {
            "obstacle_center_x": 0.0,
            "obstacle_center_y": 0.0,
            "obstacle_center_z": 0.3,
            "obstacle_width": 0.1,
        }
        th_dim = 3
        A = np.vstack((np.eye(th_dim), -np.eye(th_dim)))
        b = np.ones(th_dim * 2)
        Theta = pc.Polytope(A, b)
        lsm0 = LoadSharingManipulator(scenario0, Theta)

        theta_hat0 = lsm0.sample_Theta_space(1).reshape((th_dim,)).numpy()

        # Create Case Study Safety Experiment
        x_start = torch.tensor(
            [
                [0.3, -0.3, 0.4, 0.0, 0.0, 0.0],
                # [0.3, -0.3, 0.4, 0.1, 0.0, 0.0],
                # [0.3, -0.3, 0.3, 0.0, 0.0, 0.0],
            ]
        )

        experiment0 = CaseStudySafetyExperimentTrajOpt2(
            "CS-TestTrajOpt2", x_start,
            n_sims_per_start=1,
            plot_x_indices=[LoadSharingManipulator.P_X, LoadSharingManipulator.P_Y, LoadSharingManipulator.P_Z],
            plot_x_labels=["$p_x$", "$p_y$", "$p_z$"],
        )

        # Set Up Inputs to run_trajopt_with_synthesis
        dynamics = lsm0
        Q = None
        R = None
        P = None
        u0 = None
        uf = None
        Tf = 10.0

        obs_center = np.array([
            lsm0.nominal_scenario["obstacle_center_x"],
            lsm0.nominal_scenario["obstacle_center_y"],
            lsm0.nominal_scenario["obstacle_center_z"]
        ])

        constraints = []
        constraints.append(
            (
                optimize.NonlinearConstraint,
                lambda x, u: np.linalg.norm(x[0:3] - obs_center),
                lsm0.nominal_scenario["obstacle_width"] / 2.0,
                float('Inf'),
            )
        )

        def dynamics_update(t, x, u, params):
            """
            dxdt = dynamics_update(t,x,u, params)
            Description:
                This function defines the dynamics of the system.
            """
            # Constants
            m = lsm0.m
            gravity = 9.81

            theta = params.get("theta", theta_hat0)

            # Unpack the state
            p = x[0:3]
            v = x[3:6]

            # Algorithm
            f = np.zeros((6,))
            f[0:3] = v
            f[3:6] = (1.0 / m) * np.diag([lsm0.K_x, lsm0.K_y, lsm0.K_z]) @ p
            f[-1] = f[-1] - gravity

            F = (1.0 / m) * np.vstack(
                (np.zeros((3, lsm0.n_controls)), np.diag([lsm0.K_x, lsm0.K_y, lsm0.K_z]))
            )

            g = (1.0 / m) * np.vstack(
                (np.zeros((3, lsm0.n_controls)), np.eye(lsm0.n_controls))
            )

            return f + F @ theta + g @ u

        results_df = experiment0.run(
            dynamics, 0.1, dynamics_update,
            Tf=5.0,
            u0=u0, uf=uf,
            constraints=constraints,
            Q=Q, R=R, P=P,
            N_timepts=N_timepts,
        )

        # self.assertEqual(control_sequences.shape, (1, N_timepts, lsm0.n_controls))
        self.assertGreater(
            len(results_df["Simulation"] == 0),
            0,
        )

    def test_plot1(self):
        """
        test_plot1
        Description:
            Run the synthesis function AND test for a single
            initial condition.
        """

        N_timepts = 10

        # Create Load Sharing Manipulator
        scenario0 = {
            "obstacle_center_x": 0.0,
            "obstacle_center_y": 0.0,
            "obstacle_center_z": 0.3,
            "obstacle_width": 0.1,
        }
        th_dim = 3
        A = np.vstack((np.eye(th_dim), -np.eye(th_dim)))
        b = np.ones(th_dim * 2)
        Theta = pc.Polytope(A, b)
        lsm0 = LoadSharingManipulator(scenario0, Theta)

        theta_hat0 = lsm0.sample_Theta_space(1).reshape((th_dim,)).numpy()

        # Create Case Study Safety Experiment
        x_start = torch.tensor(
            [
                [0.3, -0.3, 0.4, 0.0, 0.0, 0.0],
                # [0.3, -0.3, 0.4, 0.1, 0.0, 0.0],
                # [0.3, -0.3, 0.3, 0.0, 0.0, 0.0],
            ]
        )

        experiment0 = CaseStudySafetyExperimentTrajOpt2(
            "CS-TestTrajOpt2", x_start,
            n_sims_per_start=1,
            plot_x_indices=[LoadSharingManipulator.P_X, LoadSharingManipulator.P_Y, LoadSharingManipulator.P_Z],
            plot_x_labels=["$p_x$", "$p_y$", "$p_z$"],
        )

        # Set Up Inputs to run_trajopt_with_synthesis
        dynamics = lsm0
        Q = None
        R = None
        P = None
        u0 = None
        uf = None
        Tf = 10.0

        obs_center = np.array([
            lsm0.nominal_scenario["obstacle_center_x"],
            lsm0.nominal_scenario["obstacle_center_y"],
            lsm0.nominal_scenario["obstacle_center_z"]
        ])

        constraints = []
        constraints.append(
            (
                optimize.NonlinearConstraint,
                lambda x, u: np.linalg.norm(x[0:3] - obs_center),
                lsm0.nominal_scenario["obstacle_width"] / 2.0,
                float('Inf'),
            )
        )

        def dynamics_update(t, x, u, params):
            """
            dxdt = dynamics_update(t,x,u, params)
            Description:
                This function defines the dynamics of the system.
            """
            # Constants
            m = lsm0.m
            gravity = 9.81

            theta = params.get("theta", theta_hat0)

            # Unpack the state
            p = x[0:3]
            v = x[3:6]

            # Algorithm
            f = np.zeros((6,))
            f[0:3] = v
            f[3:6] = (1.0 / m) * np.diag([lsm0.K_x, lsm0.K_y, lsm0.K_z]) @ p
            f[-1] = f[-1] - gravity

            F = (1.0 / m) * np.vstack(
                (np.zeros((3, lsm0.n_controls)), np.diag([lsm0.K_x, lsm0.K_y, lsm0.K_z]))
            )

            g = (1.0 / m) * np.vstack(
                (np.zeros((3, lsm0.n_controls)), np.eye(lsm0.n_controls))
            )

            return f + F @ theta + g @ u

        results_df = experiment0.run(
            dynamics, 0.1, dynamics_update,
            Tf=5.0,
            u0=u0, uf=uf,
            constraints=constraints,
            Q=Q, R=R, P=P,
            N_timepts=N_timepts,
        )

        # self.assertEqual(control_sequences.shape, (1, N_timepts, lsm0.n_controls))
        self.assertGreater(
            len(results_df["Simulation"] == 0),
            0,
        )

        fhs = experiment0.plot(
            lsm0,
            results_df,
        )
        f_title, f_fig = fhs[0]

        if "/neural_clbf/experiments/adaptive/tests" in os.getcwd():
            # Only save if we are running from inside tests directory
            f_fig.savefig("figures/trajopt2-test-plot1.png")

if __name__ == "__main__":
    unittest.main()