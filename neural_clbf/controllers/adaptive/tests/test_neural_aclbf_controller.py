"""
test_neural_aclbf_controller.py
Description:
    Test the neural adaptive CLBF controller.
"""

from neural_clbf.controllers import NeuralCLBFController, NeuralaCLBFController
from neural_clbf.systems.adaptive import (
    ScalarCAPA2Demo, LoadSharingManipulator, ControlAffineParameterAffineSystem,
)
from neural_clbf.datamodules import EpisodicDataModuleAdaptive
from neural_clbf.experiments import ExperimentSuite
from neural_clbf.experiments.adaptive import (
    AdaptiveCLFContourExperiment
)

import unittest
import polytope as pc
import numpy as np
import torch

class TestNeuralaCLBFController(unittest.TestCase):
    def test_neural_aclbf_controller_simulate1(self):
        """
        test_neural_aclbf_controller_simulate1
        Description:
            This test verifies that the simulate function for
            the neural adaptive CLBF controller works as expected.
        """

        # Constants
        wall_pos = -0.5
        nominal_scenario = {"wall_position": wall_pos}
        controller_period = 0.05
        simulation_dt = 0.01

        num_sim_steps = 10

        # Define the system
        # Define the range of possible uncertain parameters
        lb = [-2.5]
        ub = [-1.5]
        Theta = pc.box2poly(np.array([lb, ub]).T)

        # Define the dynamics model
        dynamics_model = ScalarCAPA2Demo(
            nominal_scenario,
            Theta,
            dt=simulation_dt,
            controller_dt=controller_period,
            scenarios=[nominal_scenario],
            device="cpu",
        )

        # Define the controller
        datamodule, hyperparams, experiment_suite = self.define_datamodule_and_params_for_scalar(
            dynamics_model,
            lb, ub,
        )
        aclbf_controller = NeuralaCLBFController(
            dynamics_model,
            [nominal_scenario],
            datamodule,
            experiment_suite=experiment_suite,
            clbf_hidden_layers=2,
            clbf_hidden_size=64,
            clf_lambda=0.1,
            safe_level=0.5,
            controller_period=controller_period,
            clf_relaxation_penalty=1e2,
            num_init_epochs=10,
            epochs_per_episode=100,
            barrier=True,
            Gamma_factor=0.1,
            include_oracle_loss=False,
            include_estimation_error_loss=False,
        )

        # Simulate!
        x_init = torch.tensor([1.0, 3.0]).reshape(2, 1)
        theta_init = torch.tensor([-2.0, -1.5]).reshape(2, 1)

        x_sim, th_sim, th_h_sim = aclbf_controller.simulate(
            x_init,
            theta_init,
            num_sim_steps,
            aclbf_controller.u,
        )

        self.assertEqual(x_sim.shape, (2, num_sim_steps, dynamics_model.n_dims))
        self.assertEqual(th_sim.shape, (2, num_sim_steps, dynamics_model.n_params))
        self.assertEqual(th_h_sim.shape, (2, num_sim_steps, dynamics_model.n_params))

        # All th_sim values should be constant
        for batch_idx in range(2):
            self.assertTrue(
                torch.all(th_sim[batch_idx, 0, 0] == th_sim[batch_idx, 1:, :])
            )

    def define_datamodule_and_params_for_scalar(
            self,
            dynamics_model,
            Theta_lb: np.array,
            Theta_ub: np.array,
    ):
        """
        define_datamodule_and_params_for_scalar
        Description:
            This function defines the datamodule and hyperparameters
            for the scalar CAPA-2 system.
        """

        # Define the hyperparameters
        hyperparams = {
            "batch_size": 64,
            "num_workers": 0,
            "pin_memory": False,
            "sample_quotas": {"safe": 0.2, "unsafe": 0.2, "goal": 0.2},
            "use_oracle": False,
            "barrier": True,
            "safe_level": 0.5,
            "use_estim_err_loss": False,
        }

        # Define the datamodule
        initial_conditions = [
            (1.0, 3.0),  # p_x
        ]
        datamodule = EpisodicDataModuleAdaptive(
            dynamics_model,
            initial_conditions,
            trajectories_per_episode=1,
            trajectory_length=1,
            fixed_samples=10000,
            max_points=100000,
            val_split=0.2,
            batch_size=hyperparams["batch_size"],
            device="cpu",
        )

        # Define the experiment suite
        x_ub, x_lb = dynamics_model.state_limits
        V_contour_experiment = AdaptiveCLFContourExperiment(
            "V_Contour",
            n_grid=30,
            # X axis details
            x_domain=[(x_lb[ScalarCAPA2Demo.X_DEMO], x_ub[ScalarCAPA2Demo.X_DEMO])],  # plotting domain
            x_axis_index=ScalarCAPA2Demo.X_DEMO,
            x_axis_label="$p_x$",
            # Theta axis details
            theta_axis_index=ScalarCAPA2Demo.P_DEMO,
            theta_domain=[(Theta_lb[0], Theta_ub[0])],  # plotting domain for theta
            theta_axis_label="$\\theta$",  # "$\\dot{\\theta}$",
            plot_unsafe_region=False,
        )
        e_suite = ExperimentSuite([V_contour_experiment])

        return datamodule, hyperparams, e_suite

    def test_neural_aclbf_controller_solve_CLF_QP_numerically1(self):
        """
        test_neural_aclbf_controller_solve_CLF_QP_numerically1
        Description:
            Creates simple logic for solving Neural CLBF QP's numerically given
            the current values of the state and parameters.
            Only computes one QP solution.
        """

        # Constants
        wall_pos = -0.5
        nominal_scenario = {"wall_position": wall_pos}
        controller_period = 0.05
        simulation_dt = 0.01

        num_sim_steps = 10

        clf_relaxation_penalty = 1e2

        # Define the system
        # Define the range of possible uncertain parameters
        lb = [-2.5]
        ub = [-1.5]
        Theta = pc.box2poly(np.array([lb, ub]).T)

        # Define the dynamics model
        dynamics_model = ScalarCAPA2Demo(
            nominal_scenario,
            Theta,
            dt=simulation_dt,
            controller_dt=controller_period,
            scenarios=[nominal_scenario],
            device="cpu",
        )

        # Define the controller
        datamodule, hyperparams, experiment_suite = self.define_datamodule_and_params_for_scalar(
            dynamics_model,
            lb, ub,
        )
        aclbf_controller = NeuralaCLBFController(
            dynamics_model,
            [nominal_scenario],
            datamodule,
            experiment_suite=experiment_suite,
            clbf_hidden_layers=2,
            clbf_hidden_size=64,
            clf_lambda=0.1,
            safe_level=0.5,
            controller_period=controller_period,
            clf_relaxation_penalty=1e2,
            num_init_epochs=10,
            epochs_per_episode=100,
            barrier=True,
            Gamma_factor=0.1,
            include_oracle_loss=False,
            include_estimation_error_loss=False,
        )

        # Define inputs to CLF QP

        x = torch.tensor([1.5])
        u_ref = torch.tensor([0.0])
        theta_hat = torch.tensor([-2.0])

        # LFGammadV_V, list_LGi_V, LGammadVG_V,
        # relaxation_penalty,
        # Q = Q

        N_samples_per_dim = 10+1

        # Algorithm
        # Sample states
        X_upper, X_lower = dynamics_model.control_limits
        grid_pts_along_dim = []
        for dim_index in range(dynamics_model.n_dims):
            grid_pts_along_dim.append(
                torch.linspace(X_lower[dim_index], X_upper[dim_index], N_samples_per_dim),
            )

        grid_pts = torch.cartesian_prod(
            *grid_pts_along_dim,
        )
        grid_pts = grid_pts.reshape((grid_pts.shape[0], dynamics_model.n_dims))

        # Evaluate constraint function and clf condition for each of these.
        batch_size = grid_pts.shape[0]
        V_Theta = pc.extreme(dynamics_model.Theta)
        n_Theta = V_Theta.shape[0]

        constraint_function0 = torch.zeros(
            (N_samples_per_dim, dynamics_model.n_dims, n_Theta)
        )
        for corner_index in range(n_Theta):

            if torch.get_default_dtype() == torch.float32:
                theta_sample_np = np.float32(V_Theta[corner_index, :])
                v_Theta = torch.tensor(theta_sample_np)
            else:
                v_Theta = torch.tensor(V_Theta[corner_index, :])

            v_Theta = v_Theta.reshape((1, dynamics_model.n_dims))
            v_Theta = v_Theta.repeat((batch_size, 1))

            constraint_function0[:, :, corner_index] = \
                dynamics_model.closed_loop_dynamics(
                    x.repeat((N_samples_per_dim, 1)), grid_pts,
                    v_Theta,
                )

        # Find the set of all batch indicies where every dynamics evaluation
        # is negative

        # print(constraint_function0)
        rectified_relaxation_vector = torch.nn.functional.relu(
                constraint_function0 * clf_relaxation_penalty,
            )

        penalties = torch.sum(rectified_relaxation_vector, dim=2).reshape((batch_size, 1))

        obj = torch.zeros((batch_size, 1))
        obj[:, :] = torch.pow(grid_pts - u_ref.repeat(batch_size, 1), 2)
        obj[:, :] = obj[:, :] + penalties

        # print(obj)
        # print(torch.argmin(obj))

        u_min = obj[torch.argmin(obj), :]
        # print(u_min)
        self.assertEqual(torch.argmin(obj), 5)
        self.assertEqual(u_min, torch.tensor([0.0]))

    def test_neural_aclbf_controller_solve_CLF_QP_numerically2(self):
        """
        test_neural_aclbf_controller_solve_CLF_QP_numerically2
        Description:
            Tests an algorithm for computing batches of CLF QP solutions numerically.
        """

        # Constants
        wall_pos = -0.5
        nominal_scenario = {"wall_position": wall_pos}
        controller_period = 0.05
        simulation_dt = 0.01

        num_sim_steps = 10

        clf_relaxation_penalty = 1e2

        # Define the system
        # Define the range of possible uncertain parameters
        lb = [-2.5]
        ub = [-1.5]
        Theta = pc.box2poly(np.array([lb, ub]).T)

        # Define the dynamics model
        dynamics_model = ScalarCAPA2Demo(
            nominal_scenario,
            Theta,
            dt=simulation_dt,
            controller_dt=controller_period,
            scenarios=[nominal_scenario],
            device="cpu",
        )

        # Define the controller
        datamodule, hyperparams, experiment_suite = self.define_datamodule_and_params_for_scalar(
            dynamics_model,
            lb, ub,
        )
        aclbf_controller = NeuralaCLBFController(
            dynamics_model,
            [nominal_scenario],
            datamodule,
            experiment_suite=experiment_suite,
            clbf_hidden_layers=2,
            clbf_hidden_size=64,
            clf_lambda=0.1,
            safe_level=0.5,
            controller_period=controller_period,
            clf_relaxation_penalty=1e2,
            num_init_epochs=10,
            epochs_per_episode=100,
            barrier=True,
            Gamma_factor=0.1,
            include_oracle_loss=False,
            include_estimation_error_loss=False,
        )

        # Define inputs to CLF QP

        x = torch.tensor([[1.5], [-0.2], [-0.4], [2.5]])
        u_ref = torch.tensor([0.0])
        theta_hat = torch.tensor([[-2.0], [-2.5]])

        batch_size = x.shape[0]

        N_samples_per_dim = 30+1

        # Algorithm
        # Sample states
        X_upper, X_lower = dynamics_model.control_limits
        grid_pts_along_dim = []
        for dim_index in range(dynamics_model.n_dims):
            grid_pts_along_dim.append(
                torch.linspace(X_lower[dim_index], X_upper[dim_index], N_samples_per_dim),
            )

        grid_pts = torch.cartesian_prod(
            *grid_pts_along_dim,
        )
        grid_pts = grid_pts.reshape((N_samples_per_dim, dynamics_model.n_dims))

        # Evaluate constraint function and clf condition for each of these.
        V_Theta = pc.extreme(dynamics_model.Theta)
        n_Theta = V_Theta.shape[0]

        descent_expression = torch.zeros(
            (batch_size, N_samples_per_dim, n_Theta, dynamics_model.n_dims)
        )
        for b_ind in range(batch_size):
            x_i = x[b_ind, :]
            for corner_index in range(n_Theta):
                # Get Corner
                if torch.get_default_dtype() == torch.float32:
                    theta_sample_np = np.float32(V_Theta[corner_index, :])
                    v_Theta = torch.tensor(theta_sample_np)
                else:
                    v_Theta = torch.tensor(V_Theta[corner_index, :])

                v_Theta = v_Theta.reshape((1, dynamics_model.n_dims))
                v_Theta = v_Theta.repeat((N_samples_per_dim, 1))

                descent_expression[b_ind, :, corner_index, :] = \
                    dynamics_model.closed_loop_dynamics(
                        x_i.repeat((N_samples_per_dim, 1)),
                        grid_pts,
                        v_Theta,
                    )

        # Find the set of all batch indicies where every dynamics evaluation
        # is negative
        rectified_relaxation_vector = torch.nn.functional.relu(
                descent_expression * clf_relaxation_penalty,
            )

        penalties = torch.sum(
            rectified_relaxation_vector, dim=[2,3],
        ).reshape((batch_size, N_samples_per_dim))

        obj = torch.zeros((batch_size, N_samples_per_dim))
        for batch_index in range(batch_size):
            u_m_uref = grid_pts - u_ref.repeat(N_samples_per_dim, 1)
            u_m_uref = u_m_uref.unsqueeze(2)

            Q = torch.eye(dynamics_model.n_controls).repeat(N_samples_per_dim, 1, 1)
            # print(torch.bmm(
            #     torch.bmm(u_m_uref.mT, Q),
            #     u_m_uref,
            # ).squeeze().shape)
            obj[batch_index, :] = torch.bmm(
                torch.bmm(u_m_uref.mT, Q),
                u_m_uref,
            ).squeeze()

        obj[:, :] = obj[:, :] + penalties

        obj_min0 = torch.min(obj, dim=1)

        min_contr0 = grid_pts[obj_min0[1], :]
        self.assertGreaterEqual(
            min_contr0[1], 0,
        )

    def define_datamodule_and_params_for_lsm(
            self,
            lsm_in: ControlAffineParameterAffineSystem,
            Theta_lb: np.array,
            Theta_ub: np.array,
    ):
        """
        datamodule, hyperparams, e_suite = self.define_datamodule_and_params_for_lsm(lsm, Theta_lb, Theta_ub)
        Description:
            This function defines the datamodule and hyperparameters
            for the LoadSharingManipulator system.
        """

        # Define the hyperparameters
        hyperparams = {
            "batch_size": 64,
            "num_workers": 10,
            "pin_memory": False,
            "sample_quotas": {"safe": 0.2, "unsafe": 0.2, "goal": 0.2},
            "use_oracle": False,
            "barrier": True,
            "safe_level": 1.5,
            "use_estim_err_loss": False,
            "controller_period": 0.1,
        }

        # Define the datamodule
        initial_conditions = [
            (-0.4, 0.4),  # p_x
            (-0.4, 0.4),  # p_y
            (0.0, 0.7),  # p_z
            (-0.5, 0.5),  # v_x
            (-0.5, 0.5),  # v_y
            (-0.5, 0.5),  # v_z
        ]
        datamodule = EpisodicDataModuleAdaptive(
            lsm_in,
            initial_conditions,
            trajectories_per_episode=1,
            trajectory_length=1,
            fixed_samples=10000,
            max_points=100000,
            val_split=0.2,
            batch_size=hyperparams["batch_size"],
            device="cpu",
        )

        # Define the experiment suite
        x_ub, x_lb = lsm_in.state_limits
        V_contour_experiment = AdaptiveCLFContourExperiment(
            "V_Contour",
            n_grid=30,
            # X axis details
            x_domain=[(x_lb[ScalarCAPA2Demo.X_DEMO], x_ub[ScalarCAPA2Demo.X_DEMO])],  # plotting domain
            x_axis_index=ScalarCAPA2Demo.X_DEMO,
            x_axis_label="$p_x$",
            # Theta axis details
            theta_axis_index=ScalarCAPA2Demo.P_DEMO,
            theta_domain=[(Theta_lb[0], Theta_ub[0])],  # plotting domain for theta
            theta_axis_label="$\\theta$",  # "$\\dot{\\theta}$",
            plot_unsafe_region=False,
        )
        e_suite = ExperimentSuite([V_contour_experiment])

        return datamodule, hyperparams, e_suite

    def test_neural_aclbf_controller_solve_CLF_QP_numerically3(self):
        """
        test_neural_aclbf_controller_solve_CLF_QP_numerically3
        Description:
            Tests an algorithm for computing batches of CLF QP solutions numerically.
            Using the dynamical system for load carrying.
        """

        # Constants
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
        dynamics_model = LoadSharingManipulator(scenario0, Theta, dt=0.025)

        num_sim_steps = 10

        clf_relaxation_penalty = 1e2

        # Define the controller
        datamodule, hyperparams, e_suite = self.define_datamodule_and_params_for_lsm(
            dynamics_model, [1.0 for _ in range(th_dim)], [-1.0 for _ in range(th_dim)],
        )

        aclbf_controller = NeuralaCLBFController(
            dynamics_model,
            [scenario0],
            datamodule,
            experiment_suite=e_suite,
            clbf_hidden_layers=2,
            clbf_hidden_size=64,
            clf_lambda=0.1,
            safe_level=0.5,
            controller_period=hyperparams["controller_period"],
            clf_relaxation_penalty=1e2,
            num_init_epochs=10,
            epochs_per_episode=100,
            barrier=True,
            Gamma_factor=0.1,
            include_oracle_loss=False,
            include_estimation_error_loss=False,
        )

        # Define inputs to CLF QP

        x = torch.tensor(
            [
                [0.3, -0.3, 0.4, 0.0, 0.0, 0.0],
                [0.25, -0.25, 0.4, 0.1, 0.0, 0.0],
                [0.35, -0.25, 0.3, 0.0, 0.0, 0.0],
                [0.35, -0.35, 0.3, 0.0, 0.0, 0.0],
                [0.25, -0.35, 0.3, 0.0, 0.0, 0.0],
                [0.15, -0.2,  0.2, 0.0, 0.0, 0.0],

            ]
        )
        u_ref = torch.tensor([0.0, 0.0, 100.0])
        theta_hat = torch.tensor([[-2.0], [-2.5]])

        batch_size = x.shape[0]

        N_samples_per_dim = 30+1

        # Algorithm
        # Sample states
        U_upper, U_lower = dynamics_model.control_limits
        grid_pts_along_dim = []
        for dim_index in range(dynamics_model.n_controls):
            grid_pts_along_dim.append(
                torch.linspace(U_lower[dim_index], U_upper[dim_index], N_samples_per_dim),
            )

        grid_pts = torch.cartesian_prod(
            *grid_pts_along_dim,
        )
        N_samples = grid_pts.shape[0]
        grid_pts = grid_pts.reshape((N_samples_per_dim**dynamics_model.n_controls, dynamics_model.n_controls))

        # Evaluate constraint function and clf condition for each of these.
        V_Theta = pc.extreme(dynamics_model.Theta)
        n_Theta = V_Theta.shape[0]

        descent_expression = torch.zeros(
            (batch_size, N_samples, n_Theta, dynamics_model.n_dims)
        )
        for b_ind in range(batch_size):
            x_i = x[b_ind, :]
            for corner_index in range(n_Theta):
                # Get Corner
                if torch.get_default_dtype() == torch.float32:
                    theta_sample_np = np.float32(V_Theta[corner_index, :])
                    v_Theta = torch.tensor(theta_sample_np)
                else:
                    v_Theta = torch.tensor(V_Theta[corner_index, :])

                v_Theta = v_Theta.reshape((1, dynamics_model.n_params))
                v_Theta = v_Theta.repeat((N_samples, 1))

                descent_expression[b_ind, :, corner_index, :] = \
                    dynamics_model.closed_loop_dynamics(
                        x_i.repeat((N_samples, 1)),
                        grid_pts,
                        v_Theta,
                    )

        # Find the set of all batch indicies where every dynamics evaluation
        # is negative
        rectified_relaxation_vector = torch.nn.functional.relu(
                descent_expression * clf_relaxation_penalty,
            )

        penalties = torch.sum(
            rectified_relaxation_vector, dim=[2, 3],
        ).reshape((batch_size, N_samples))

        obj = torch.zeros((batch_size, N_samples))
        for batch_index in range(batch_size):
            u_m_uref = grid_pts - u_ref.repeat(N_samples, 1)
            u_m_uref = u_m_uref.unsqueeze(2)

            Q = torch.eye(dynamics_model.n_controls).repeat(N_samples, 1, 1)
            obj[batch_index, :] = torch.bmm(
                torch.bmm(u_m_uref.mT, Q),
                u_m_uref,
            ).squeeze()

        obj[:, :] = obj[:, :] + penalties

        obj_min0 = torch.min(obj, dim=1)

        min_contr0 = grid_pts[obj_min0[1], :]
        print(min_contr0)
        self.assertGreaterEqual(
            min_contr0[1, 0], 0,
        )

if __name__ == "__main__":
    unittest.main()