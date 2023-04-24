"""
test_case_study_safety_experiment.py
Description:
    This script will test the methods and features of the case study safety experiment.
"""
import numpy as np
import torch
import polytope as pc
import control as ct
import control.optimal as opt
import scipy as sp
import scipy.optimize as optimize
import time
import matplotlib.pyplot as plt

import unittest

from neural_clbf.systems.adaptive import LoadSharingManipulator
from neural_clbf.experiments.adaptive import CaseStudySafetyExperiment

class TestCaseStudySafetyExperiment(unittest.TestCase):
    def test_case_study_safety_experiment_init(self):
        """
        Description:
            Test initialization of the case study safety experiment.
            Should correctly be initialized.
        """
        # Test instantiation with valid parameters
        x0 = torch.tensor(
            [
                [0.3, -0.3, 0.4, 0.0, 0.0, 0.0],
                [0.3, -0.3, 0.4, 0.1, 0.0, 0.0],
                [0.3, -0.3, 0.3, 0.0, 0.0, 0.0],
                [0.3, -0.3, 0.4, 0.0, 0.0, 0.2],
            ]
        )
        case_study_name = "Safety Case Study"

        safety_case_study_experiment = CaseStudySafetyExperiment(
            case_study_name,
            x0,
            n_sims_per_start=1,
            t_sim=15.0,
            plot_x_indices=[LoadSharingManipulator.P_X, LoadSharingManipulator.P_Y, LoadSharingManipulator.P_Z],
            plot_x_labels=["$p_x$", "$p_y$", "$p_z$"],
        )

        self.assertEqual(safety_case_study_experiment.name, case_study_name)

    def test_case_study_safety_experiment_run_trajopt_with_synthesis1(self):
        """
        Description:
            This test evaluates the logic used in the `run_trajopt_with_synthesis` method.
            It does not explicitly call the method, but attempts to use a portion of its code.
        """

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
                [0.3, -0.3, 0.4, 0.1, 0.0, 0.0],
                [0.3, -0.3, 0.3, 0.0, 0.0, 0.0],
            ]
        )
        safety_case_study_experiment = CaseStudySafetyExperiment(
            "CS-TestTrajopt1", x_start,
            n_sims_per_start=1,
            t_sim=15.0,
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


        # Attempt to run the method's logic
        assert \
            theta_hat0.shape == (dynamics.n_params,), \
            f"theta_hat0 must be of shape ({dynamics.n_params},); received {theta_hat0.shape}"

        if Q is None:
            Q = np.diag([1.0 for i in range(dynamics.n_dims)])
        if R is None:
            R = np.diag([1.0 for i in range(dynamics.n_controls)])
        if P is None:
            P = np.diag([10000.0 for i in range(dynamics.n_dims)])  # get close to final point
        if u0 is None:
            u0 = np.zeros((dynamics.n_controls,))
            u0[0] = 10.0
        if uf is None:
            uf = np.zeros((dynamics.n_controls,))

        # Constants
        N_timepts = 25

        # Setup trajectory optimization
        # ==============================

        # Setup the output equation
        def output_equation(t, x, u, params):
            return x

        # Create optimized trajectory for each ic
        n_x0s = x_start.shape[0]
        theta_samples = lsm0.sample_Theta_space(n_x0s).numpy()
        opt_times = []
        for x0_index in range(x_start.shape[0]):
            # Get x0 and xf
            x0 = x_start[x0_index, :].numpy()
            theta_sample = theta_samples[x0_index, :].reshape((dynamics.n_params,))
            xf = dynamics.goal_point(
                torch.tensor(theta_sample).reshape(1, dynamics.n_params),
            ).reshape((dynamics.n_dims,))
            xf = xf.numpy()

            # Define system
            output_list = tuple([f'x_{i}' for i in range(dynamics.n_dims)])
            input_list = tuple([f'u_{i}' for i in range(dynamics.n_controls)])
            system = ct.NonlinearIOSystem(
                dynamics_update, output_equation,
                states=dynamics.n_dims, name='case-study-system',
                inputs=input_list, outputs=output_list,
                params={"theta": theta_sample},
            )

            # Setup the initial and final conditions

            # Setup the cost function
            traj_cost = opt.quadratic_cost(system, Q, R, x0=xf, u0=uf)
            term_cost = opt.quadratic_cost(system, P, 0, x0=xf)

            # Add constraint
            constraints.append(
                opt.input_poly_constraint(system, dynamics.U.A, dynamics.U.b)
            )

            # Setup the trajectory optimization problem
            timepts = np.linspace(0.0, Tf, N_timepts, endpoint=True)
            traj_opt_start = time.time()
            result = opt.solve_ocp(
                system, timepts, x0,
                traj_cost, constraints,
                terminal_cost=term_cost, initial_guess=u0,
            )
            traj_opt_end = time.time()
            opt_times.append(
                traj_opt_end - traj_opt_start,
            )

            # Simulate the system dynamics (open loop)
            resp = ct.input_output_response(
                system, timepts, result.inputs, x0,
                t_eval=timepts)
            t, y, u = resp.time, resp.outputs, resp.inputs

            # Analyze results
            # print(t)
            self.assertEqual(t.shape, (N_timepts,))
            print("y[:, -1] = ", y[:, -1])
            print("xf = ", xf)
            self.assertLess(
                np.linalg.norm(y[:, -1] - xf),
                1.0,
            )

        self.assertEqual(len(opt_times), n_x0s)


    def test_case_study_safety_experiment_run_trajopt_with_synthesis2(self):
        """
        test_case_study_safety_experiment_run_trajopt_with_synthesis2
        Description:
            This test will use the actual method to perform some analysis of
            the Load Sharing Manipulator.
        """

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
                [0.3, -0.3, 0.4, 0.1, 0.0, 0.0],
                [0.3, -0.3, 0.3, 0.0, 0.0, 0.0],
                [0.3, -0.3, 0.4, 0.0, 0.0, 0.2],
            ]
        )
        safety_case_study_experiment = CaseStudySafetyExperiment(
            "CS-TestTrajopt1", x_start,
            n_sims_per_start=1,
            t_sim=15.0,
            plot_x_indices=[LoadSharingManipulator.P_X, LoadSharingManipulator.P_Y, LoadSharingManipulator.P_Z],
            plot_x_labels=["$p_x$", "$p_y$", "$p_z$"],
        )

        # Run the experiment
        def lsm_update(t, x, u, params):
            """
            dxdt = lsm_update(t,x,u, params)
            Description:
                This function defines the dynamics of the system.
            """
            # Constants
            m = lsm0.m
            gravity = 9.81
            theta = params.get("theta", np.array([-0.15, 0.4, 0.1]))

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

        df_trajopt, trajopt_synth_times = safety_case_study_experiment.run_trajopt_with_synthesis(
            lsm0, 0.1, lsm_update,
            10.0,
            P=np.diag([1.0e5, 1.0e5, 1.0e5, 1.0e4, 1.0e4, 1.0e4]),
            N_timepts=20,
        )

        #print(trajopt_synth_times)

        for timing_i in trajopt_synth_times:
            self.assertGreater(
                timing_i,
                0.0,
            )


    def test_case_study_safety_experiment_plot_trajectory1(self):
        """
        test_case_study_safety_experiment_plot_trajectory1
        Description:
            This script tests whether or not the plot_Trajectory function
            works in the case study safety experiment class.
        """

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
                [0.3, -0.3, 0.4, 0.1, 0.0, 0.0],
                [0.3, -0.3, 0.3, 0.0, 0.0, 0.0],
            ]
        )
        safety_case_study_experiment = CaseStudySafetyExperiment(
            "CS-TestTrajopt1", x_start,
            n_sims_per_start=1,
            t_sim=15.0,
            plot_x_indices=[LoadSharingManipulator.P_X, LoadSharingManipulator.P_Y, LoadSharingManipulator.P_Z],
            plot_x_labels=["$p_x$", "$p_y$", "$p_z$"],
        )

        # Simulate with nominal controller
        nominal_df = safety_case_study_experiment.run_nominal_controlled(
            lsm0, 0.1,
        )

        # Plot the nominal trajectory
        fig_tuple1 = safety_case_study_experiment.plot_trajectory(
            nominal_df,
            goal_tolerance=lsm0.goal_tolerance,
            fig_name="Test 1: Nominal Trajectory",
        )

        # Save Plot
        fig_name, fig_obj = fig_tuple1
        plt.figure(fig_obj.number)
        plt.savefig("./figures/cs-se-plot_trajectory1.png")

if __name__ == "__main__":
    unittest.main()