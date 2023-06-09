"""
test_aclf_controller6.py
Description:
    This test suite verifies that aCLFController 5 is working with the given definitions.
"""
import unittest
import polytope as pc
import torch

from neural_clbf.experiments import (
    ExperimentSuite,
)
from neural_clbf.experiments.adaptive_w_observed_parameters import (
    AdaptiveCLFContourExperiment3,
)
from neural_clbf.controllers.adaptive_with_observed_parameters import (
    aCLFController6,
)
from neural_clbf.systems.adaptive_w_scenarios import (
    AdaptivePusherSliderStickingForceInput3,
)

import matplotlib.pyplot as plt
from neural_clbf.controllers.adaptive_with_observed_parameters.adaptive_controller_utils import (
    create_grid_of_feasible_convex_combinations_of_V,
    create_uniform_samples_across_polytope,
)

class TestACLFController6(unittest.TestCase):
    def get_ps1(self):
        """
        get_ps1
        Description:
            Creates a simple pusher slider for testing.
        """

        # Constants
        nominal_scenario = {}
        for ind in range(2):
            nominal_scenario[f"obstacle_{ind}_center_x"] = 0.0
            nominal_scenario[f"obstacle_{ind}_center_y"] = 0.0
            nominal_scenario[f"obstacle_{ind}_radius"] = 0.1

        nominal_scenario["goal_x"] = 0.5
        nominal_scenario["goal_y"] = 0.5

        V_Theta = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])

        # Create Object
        aps = AdaptivePusherSliderStickingForceInput3(
            nominal_scenario, V_Theta,
            n_obstacles=2,
        )

        return aps

    def test_init1(self):
        """
        test_init1
        Description:
            Tests that we can correctly initialize such a controller.
        """
        # Constants
        aps = self.get_ps1()

        # Create experiments
        exp1 = AdaptiveCLFContourExperiment3(
            "test-aclf_experiment",
        )
        suite = ExperimentSuite([exp1])

        # Create controller
        controller0 = aCLFController6(
            aps, suite,
        )

        self.assertEqual(
            controller0.clf_lambda, 1.0,
        )
        self.assertEqual(controller0.controller_period, 0.01)


    def test_V_with_jacobian1(self):
        """
        test_V_with_jacobian1
        Description:
            This function tests that the proper compuation is done for V with jacobian in this new aCLFcontroller.
        """
        # Constants
        aps = self.get_ps1()

        # Create experiments
        exp1 = AdaptiveCLFContourExperiment3(
            "test-aclf_experiment",
        )
        suite = ExperimentSuite([exp1])

        # Create controller
        controller0 = aCLFController6(
            aps, suite,
        )

        # Algorithm
        x = torch.tensor([
            [0.1, -0.4, torch.pi/2],
        ])
        theta_hat = aps.sample_Theta_space(1)
        scen = aps.sample_scenario_space(1)

        V, JxV, JthV = controller0.V_with_jacobian(x, theta_hat, scen)

        self.assertGreaterEqual(
            V[0], 0
        )

    def test_V1(self):
        """
        test_V1
        Description:
            This function tests that V works properly.
        """
        # Constants
        aps = self.get_ps1()

        # Create experiments
        exp1 = AdaptiveCLFContourExperiment3(
            "test-aclf_experiment",
        )
        suite = ExperimentSuite([exp1])

        # Create controller
        controller0 = aCLFController6(
            aps, suite,
        )

        # Algorithm
        x = torch.tensor([
            [0.1, -0.4, torch.pi / 2],
        ])
        theta_hat = aps.sample_Theta_space(1)
        scen = aps.sample_scenario_space(1)

        V = controller0.V(x, theta_hat, scen)

        self.assertGreaterEqual(
            V[0], 0
        )

    def test_V_lie_derivatives1(self):
        """
        test_V_lie_derivatives1
        Description:
            Tests that the lie derivative computation works for the aclf controller.
        """
        # Constants
        aps = self.get_ps1()

        # Create experiments
        exp1 = AdaptiveCLFContourExperiment3(
            "test-aclf_experiment",
        )
        suite = ExperimentSuite([exp1])

        # Create controller
        controller0 = aCLFController6(
            aps, suite,
        )

        # Algorithm
        x = torch.tensor([
            [0.1, -0.4, torch.pi / 2],
            [0.1, -0.4, torch.pi / 2],
        ])
        theta_hat = aps.sample_Theta_space(2)
        scen = aps.sample_scenario_space(2)
        scen[:, -1] = -0.4
        scen[0, -2] = 0.0
        scen[1, -2] = 0.2

        Lf_V, LF_V, LFGammadV_V, Lg_V, list_LGi_V, LGammadVG_V = controller0.V_lie_derivatives(x, theta_hat, scen)

        self.assertTrue(
            torch.all(Lf_V == 0.0)
        )
        self.assertFalse(
            torch.all(Lg_V == 0.0),
        )
        for list_idx in range(len(list_LGi_V)):
            self.assertFalse(
                torch.all(list_LGi_V[list_idx] == 0.0),
            )

    def test_Vdot_for_scenario1(self):
        """
        test_Vdot_for_scenario1
        Description:
            Tests whether or not this shortcut function for computing "Vdot" (for barrier condition) is valid.
        """
        # Constants
        aps = self.get_ps1()

        # Create experiments
        exp1 = AdaptiveCLFContourExperiment3(
            "test-aclf_experiment",
        )
        suite = ExperimentSuite([exp1])

        # Create controller
        controller0 = aCLFController6(
            aps, suite,
        )

        # Algorithm
        x = torch.tensor([
            [0.1, -0.4, 0.0],
            [0.1, -0.4, 0.0],
        ])
        theta_hat = aps.sample_Theta_space(2)
        scen = aps.sample_scenario_space(2)
        scen[:, -1] = -0.4
        scen[0, -2] = 0.0
        scen[1, -2] = 0.2

        u = torch.tensor([
            [0.1, 0.05],
            [0.1, 0.05],
        ])

        Vdot = controller0.Vdot_for_scenario(
            x, theta_hat, scen, u,
        )

        self.assertGreaterEqual(Vdot[0], 0)
        self.assertLessEqual(Vdot[1], 0)


    def test__solve_CLF_QP_gurobi1(self):
        """
        test__solve_CLF_QP_gurobi1
        Description
            Tests that the gurobi solver works for the CLF QP.
            Batch size is 2.
        """

        # Constants
        aps = self.get_ps1()

        # Create experiments
        exp1 = AdaptiveCLFContourExperiment3(
            "test-aclf_experiment",
        )
        suite = ExperimentSuite([exp1])

        # Create controller
        controller0 = aCLFController6(
            aps, suite,
        )

        # Create test data
        x = torch.tensor([
            [0.1, -0.4, 0.0],
            [0.1, -0.4, 0.0],
        ])
        theta_hat = aps.sample_Theta_space(2)
        scen = aps.sample_scenario_space(2)
        scen[:, -1] = -0.4
        scen[0, -2] = 0.0
        scen[1, -2] = 0.2

        # Solve aCLF QP
        u, relaxation = controller0._solve_CLF_QP_gurobi(
            x, theta_hat, scen,
            controller0.u_reference(x, theta_hat, scen),
            controller0.V(x, theta_hat, scen),
            1e-3,
        )

        # Check that the solution is valid
        U = pc.qhull(
            aps.U_vertices.numpy(),
        )
        for batch_idx in range(2):
            self.assertTrue(
                U.__contains__(u[batch_idx, :].numpy()),
            )

    def test__solve_CLF_QP_cvxpylayers1(self):
        """
        test__solve_CLF_QP_cvxpylayers1
        Description
            Tests that the CvxPyLayers solver works for the CLF QP.
            Batch size is 2.
        """

        # Constants
        aps = self.get_ps1()

        # Create experiments
        exp1 = AdaptiveCLFContourExperiment3(
            "test-aclf_experiment",
        )
        suite = ExperimentSuite([exp1])

        # Create controller
        controller0 = aCLFController6(
            aps, suite,
        )

        # Create test data
        x = torch.tensor([
            [0.1, -0.4, 0.0],
            [0.1, -0.4, 0.0],
        ])
        theta_hat = aps.sample_Theta_space(2)
        scen = aps.sample_scenario_space(2)
        scen[:, -1] = -0.4
        scen[0, -2] = 0.0
        scen[1, -2] = 0.2

        # Solve aCLF QP
        u_cvxpylayers, relaxation_cvxpylayers = controller0._solve_CLF_QP_cvxpylayers(
            x, theta_hat, scen,
            controller0.u_reference(x, theta_hat, scen),
            controller0.V(x, theta_hat, scen),
            1e-3,
        )

        u_gurobi, relaxation_gurobi = controller0._solve_CLF_QP_gurobi(
            x, theta_hat, scen,
            controller0.u_reference(x, theta_hat, scen),
            controller0.V(x, theta_hat, scen),
            1e-3,
        )

        # Check that the solution is valid
        U = pc.qhull(
            aps.U_vertices.numpy(),
        )
        for batch_idx in range(2):
            self.assertTrue(
                U.__contains__(u_cvxpylayers[batch_idx, :].numpy()),
            )

        # print(u_gurobi)
        # print(u_cvxpylayers)

        self.assertTrue(
            torch.allclose(u_cvxpylayers, u_gurobi, atol=1e-3),
        )

    def test__solve_CLF_QP_analytically1(self):
        """
        test__solve_CLF_QP_analytically1
        Description
            Tests that the analytic solver works for the CLF QP.
            Batch size is 2.
        """

        # Constants
        aps = self.get_ps1()

        # Create experiments
        exp1 = AdaptiveCLFContourExperiment3(
            "test-aclf_experiment",
        )
        suite = ExperimentSuite([exp1])

        # Create controller
        controller0 = aCLFController6(
            aps, suite,
        )

        # Create test data
        x = torch.tensor([
            [0.1, -0.4, 0.0],
            [0.1, -0.4, 0.0],
        ])
        theta_hat = aps.sample_Theta_space(2)
        scen = aps.sample_scenario_space(2)
        scen[:, -1] = -0.4
        scen[0, -2] = 0.0
        scen[1, -2] = 0.2

        # Grid input space
        n_controls = aps.n_controls
        n_V_U = aps.U_vertices.shape[0]
        steps = 4

        ranges = []
        for vertex_index in range(n_V_U-1):
            ranges.append(
                torch.linspace(0, 1.0, steps),
            )

        grid_tuple = torch.meshgrid(
            *ranges,
            indexing='xy',
        )

        self.assertEqual(
            len(grid_tuple), n_V_U-1,
        )

        self.assertEqual(
            grid_tuple[0].shape, (steps, steps),
        )

        # Create grid as a single matrix
        grid_as_matrix = torch.zeros(
            (n_V_U-1, steps**n_controls),
        )
        for vertex_index in range(n_V_U-1):
            grid_as_matrix[vertex_index, :] = grid_tuple[vertex_index].flatten().squeeze()

        sum = torch.sum(grid_as_matrix, dim=0, keepdim=True)
        grid_as_matrix = torch.vstack(
            (grid_as_matrix, 1.0 - sum),
        )
        grid_as_matrix = grid_as_matrix / float(n_V_U)

        # Solve aCLF QP
        fig = plt.figure()
        plt.scatter(
            grid_as_matrix[0, :].numpy(),
            grid_as_matrix[1, :].numpy(),
        )

        fig.savefig("figures/test__solve_CLF_QP_analytically1.png")

    def test__solve_CLF_QP_analytically2(self):
        """
        test__solve_CLF_QP_analytically1
        Description
            Tests that the analytic solver works for the CLF QP.
            Batch size is 2.
        """

        # Constants
        aps = self.get_ps1()

        # Create experiments
        exp1 = AdaptiveCLFContourExperiment3(
            "test-aclf_experiment",
        )
        suite = ExperimentSuite([exp1])

        # Create controller
        controller0 = aCLFController6(
            aps, suite,
        )

        # Create test data
        x = torch.tensor([
            [0.1, -0.4, 0.0],
            [0.1, -0.4, 0.0],
        ])
        theta_hat = aps.sample_Theta_space(2)
        scen = aps.sample_scenario_space(2)
        scen[:, -1] = -0.4
        scen[0, -2] = 0.0
        scen[1, -2] = 0.2

        # Grid input space
        n_controls = aps.n_controls
        n_V_U = aps.U_vertices.shape[0]
        steps = 4

        ranges = []
        for vertex_index in range(n_V_U - 1):
            ranges.append(
                torch.linspace(0, 1.0, steps),
            )

        grid_tuple = torch.meshgrid(
            *ranges,
            indexing='xy',
        )

        self.assertEqual(
            len(grid_tuple), n_V_U - 1,
        )

        self.assertEqual(
            grid_tuple[0].shape, (steps, steps),
        )

        # Create grid as a single matrix
        grid_as_matrix = torch.zeros(
            (n_V_U - 1, steps ** n_controls),
        )
        for vertex_index in range(n_V_U - 1):
            grid_as_matrix[vertex_index, :] = grid_tuple[vertex_index].flatten().squeeze()

        sum = torch.sum(grid_as_matrix, dim=0, keepdim=True)
        grid_as_matrix = torch.vstack(
            (grid_as_matrix, float(n_V_U) - sum),
        )
        grid_as_matrix = grid_as_matrix / float(n_V_U)

        # Solve aCLF QP
        self.assertTrue(
            torch.allclose(grid_as_matrix, create_grid_of_feasible_convex_combinations_of_V(n_V_U, steps)),
        )

    def test__solve_CLF_QP_analytically3(self):
        """
        test__solve_CLF_QP_analytically3
        Description
            Tests that the analytic solver works for the CLF QP.

        """

        # Constants
        aps = self.get_ps1()

        # Create experiments
        exp1 = AdaptiveCLFContourExperiment3(
            "test-aclf_experiment",
        )
        suite = ExperimentSuite([exp1])

        # Create controller
        controller0 = aCLFController6(
            aps, suite,
        )

        # Create test data
        x = torch.tensor([
            [0.1, -0.4, 0.0],
            [0.1, -0.4, 0.0],
        ])
        theta_hat = aps.sample_Theta_space(2)
        scen = aps.sample_scenario_space(2)
        scen[:, -1] = -0.4
        scen[0, -2] = 0.0
        scen[1, -2] = 0.2

        # Create grid as a single matrix
        steps = 10
        n_V_U = aps.U_vertices.shape[0]
        grid_as_matrix = create_grid_of_feasible_convex_combinations_of_V(n_V_U, steps)

        # print(grid_as_matrix)

        # convert grid
        U_samples = aps.U_vertices.T @ grid_as_matrix

        U = pc.qhull(
            aps.U_vertices.cpu().numpy(),
        )
        contained_count = 0
        for u_index in range(U_samples.shape[1]):
            #print(U_samples[:, u_index].numpy())
            if U.__contains__(U_samples[:, u_index].numpy()):
                contained_count += 1

        self.assertTrue(
            contained_count > 0.9 * U_samples.shape[1],
        )

        # Solve aCLF QP
        bs = x.shape[0]
        num_u_options = grid_as_matrix.shape[1]
        obj = torch.zeros(
            (bs, num_u_options),
        )


    def test__solve_CLF_QP_analytically4(self):
        """
        test__solve_CLF_QP_analytically4
        Description
            Tests that the analytic solver works for the CLF QP.
        """

        # Constants
        aps = self.get_ps1()

        # Create experiments
        exp1 = AdaptiveCLFContourExperiment3(
            "test-aclf_experiment",
        )
        suite = ExperimentSuite([exp1])

        # Create controller
        controller0 = aCLFController6(
            aps, suite,
        )

        # Create test data
        x = torch.tensor([
            [0.1, -0.4, 0.0],
            [0.1, -0.4, 0.0],
        ])
        theta_hat = aps.sample_Theta_space(2)
        scen = aps.sample_scenario_space(2)
        scen[:, -1] = -0.4
        scen[0, -2] = 0.0
        scen[1, -2] = 0.2

        batch_size = x.shape[0]

        # Create grid as a single matrix
        steps = 20
        n_V_U = aps.U_vertices.shape[0]
        n_controls = aps.n_controls
        grid_as_matrix = create_grid_of_feasible_convex_combinations_of_V(n_V_U, steps)
        U_samples = aps.U_vertices.T @ grid_as_matrix
        # u_cand = torch.tensor(U_samples).repeat(batch_size, 1, 1)
        # print(u_cand.shape)
        n_samples = U_samples.shape[1]

        # print(grid_as_matrix)

        # Compute objective
        u_nominal = torch.tensor([
            [1.0, 0.3],
            [1.0, -0.3],
        ])
        u_nominal_copied = u_nominal.unsqueeze(2)
        u_nominal_copied = u_nominal_copied.repeat(1, 1, grid_as_matrix.shape[1])
        objective = torch.zeros(
            (batch_size, grid_as_matrix.shape[1]),
        )
        for batch_index in range(batch_size):
            tempU = U_samples.T
            tempU = tempU.unsqueeze(2)

            temp_u_nominal = u_nominal[batch_index, :].reshape(1, n_controls)
            temp_u_nominal = temp_u_nominal.repeat(n_samples, 1).unsqueeze(2)

            obj_i = torch.bmm(
                (tempU - temp_u_nominal).mT,
                torch.bmm(
                    torch.tensor([[1.0, 0.0], [0.0, 2.0]]).repeat(n_samples, 1, 1),
                    (tempU - temp_u_nominal),
                ),
            )
            # print(obj_i.shape)
            # print(obj_i)

            objective[batch_index, :] = obj_i.squeeze()

        print(objective)
        print(objective.shape)

    def test__solve_CLF_QP_analytically5(self):
        """
        test__solve_CLF_QP_analytically5
        Description
            Tests that the analytic solver works for the CLF QP.
        """

        # Constants
        aps = self.get_ps1()

        # Create experiments
        exp1 = AdaptiveCLFContourExperiment3(
            "test-aclf_experiment",
        )
        suite = ExperimentSuite([exp1])

        # Create controller
        controller0 = aCLFController6(
            aps, suite,
        )

        # Create test data
        x = torch.tensor([
            [0.2, 0.5, 0.0],
            [0.7, 0.6, 0.0],
        ])
        theta_hat = aps.sample_Theta_space(2)
        scen = aps.sample_scenario_space(2)
        scen[:, -1] = -0.4
        scen[0, -2] = 0.0
        scen[1, -2] = 0.2

        batch_size = x.shape[0]

        # Create grid as a single matrix
        steps = 20
        U_samples = create_uniform_samples_across_polytope(
            aps.U_vertices, steps,
        )
        # u_cand = torch.tensor(U_samples).repeat(batch_size, 1, 1)
        # print(u_cand.shape)
        n_samples = U_samples.shape[1]

        # print(grid_as_matrix)

        # Compute constraints
        n_V_Theta = aps.V_Theta.shape[0]
        u = U_samples.T
        lhs = torch.zeros(
            (batch_size, n_samples)
        )
        for batch_index in range(batch_size):
            temp_lhs = torch.zeros(
                (n_samples, n_V_Theta)
            )
            for v_index in range(n_V_Theta):
                temp_v = aps.V_Theta[v_index, :].reshape(1, aps.n_params)
                temp_v = temp_v.repeat(n_samples, 1)

                temp_x = x[batch_index, :].reshape(1, aps.n_dims)
                temp_x = temp_x.repeat(n_samples, 1)
                temp_scen = scen[batch_index, :].reshape(1, aps.n_scenario)
                temp_scen = temp_scen.repeat(n_samples, 1)

                temp_lhs[:, v_index] = controller0.Vdot_for_scenario(
                    temp_x, temp_v, temp_scen, u,
                ).squeeze()
                #print(temp_lhs)

                temp_lhs[:, v_index] = temp_lhs[:, v_index] + controller0.clf_lambda * controller0.V(temp_x, temp_v, temp_scen)



            # print(temp_lhs.shape)
            # print(temp_lhs)
            lhs[batch_index, :] = torch.max(temp_lhs, dim=1)[0]

        # print(temp_lhs.shape)

        # Compute maximum of all of these

        lhs_func = controller0.evaluate_QP_constraint_function_across(
            U_samples, x, scen,
        )

        # Compute Maximums
        for batch_index in range(batch_size):
            print(lhs[batch_index, :])
            print(torch.max(lhs_func[batch_index, :, :], dim=1)[0].squeeze())
            self.assertTrue(
                torch.allclose(
                    lhs[batch_index, :], torch.max(lhs_func[batch_index, :, :], dim=1)[0].squeeze(),
                ),
            )

    def test__solve_CLF_QP_analytically6(self):
        """
        test__solve_CLF_QP_analytically6
        Description:
            Testing putting all of the steps together to evaluate the QP using just sampling.
        """

        # Constants
        aps = self.get_ps1()
        n_controls = aps.n_controls

        # Create experiments
        exp1 = AdaptiveCLFContourExperiment3(
            "test-aclf_experiment",
        )
        suite = ExperimentSuite([exp1])

        # Create controller
        controller0 = aCLFController6(
            aps, suite,
        )

        # Create test data
        x = torch.tensor([
            [0.2, 0.5, 0.0],
            [0.7, 0.6, 0.0],
        ])
        theta_hat = aps.sample_Theta_space(2)
        scen = aps.sample_scenario_space(2)
        scen[:, -1] = -0.4
        scen[0, -2] = 0.0
        scen[1, -2] = 0.2

        batch_size = x.shape[0]

        # Create grid as a single matrix
        steps = 20
        U_samples = create_uniform_samples_across_polytope(
            aps.U_vertices, steps,
        )
        # u_cand = torch.tensor(U_samples).repeat(batch_size, 1, 1)
        # print(u_cand.shape)
        n_samples = U_samples.shape[1]

        # print(grid_as_matrix)

        # Compute constraints
        lhs_func = controller0.evaluate_QP_constraint_function_across(
            U_samples, x, scen,
        )
        lhs = torch.max(lhs_func, dim=1)[0]

        # Compute objective
        u_nominal = torch.tensor([
            [1.0, 0.3],
            [1.0, -0.3],
        ])
        u_nominal_copied = u_nominal.unsqueeze(2)
        u_nominal_copied = u_nominal_copied.repeat(1, 1, n_samples)
        objective = torch.zeros(
            (batch_size, n_samples),
        )
        for batch_index in range(batch_size):
            tempU = U_samples.T
            tempU = tempU.unsqueeze(2)

            temp_u_nominal = u_nominal[batch_index, :].reshape(1, n_controls)
            temp_u_nominal = temp_u_nominal.repeat(n_samples, 1).unsqueeze(2)

            obj_i = torch.bmm(
                (tempU - temp_u_nominal).mT,
                torch.bmm(
                    torch.tensor([[1.0, 0.0], [0.0, 2.0]]).repeat(n_samples, 1, 1),
                    (tempU - temp_u_nominal),
                ),
            )
            relaxation = torch.max(lhs_func[batch_index, :, :], dim=1)[0].squeeze()
            relaxation = torch.relu(-relaxation)
            obj_i = obj_i + relaxation.reshape(n_samples, 1, 1)

            print("objective.shape = ", objective.shape)
            print("objective[batch_index, :].shape = ", objective[batch_index, :].shape)
            objective[batch_index, :] = obj_i.squeeze()

        print(objective)
        print(objective.shape)

        # TODO: Check that constraints + relaxation value is always negative

    def test__solve_CLF_QP_analytically7(self):
        """
        test__solve_CLF_QP_analytically7
        Description:
            Testing putting all of the steps together to evaluate the QP using just sampling.
        """

        # Constants
        aps = self.get_ps1()
        n_controls = aps.n_controls

        # Create experiments
        exp1 = AdaptiveCLFContourExperiment3(
            "test-aclf_experiment",
        )
        suite = ExperimentSuite([exp1])

        # Create controller
        controller0 = aCLFController6(
            aps, suite,
        )

        # Create test data
        x = torch.tensor([
            [0.2, 0.5, 0.0],
            [0.7, 0.6, 0.0],
        ])
        theta_hat = aps.sample_Theta_space(2)
        scen = aps.sample_scenario_space(2)
        scen[:, -1] = -0.4
        scen[0, -2] = 0.0
        scen[1, -2] = 0.2

        batch_size = x.shape[0]

        # Create grid as a single matrix
        steps = 20
        U_samples = create_uniform_samples_across_polytope(
            aps.U_vertices, steps,
        )
        # u_cand = torch.tensor(U_samples).repeat(batch_size, 1, 1)
        # print(u_cand.shape)
        n_samples = U_samples.shape[1]

        # print(grid_as_matrix)

        # Compute constraints
        lhs_func = controller0.evaluate_QP_constraint_function_across(
            U_samples, x, scen,
        )
        lhs = torch.max(lhs_func, dim=1)[0]

        # Compute objective
        objective = controller0.evaluate_QP_objective_function_across(
            U_samples, x, theta_hat, scen,
            lhs,
        )



if __name__ == '__main__':
    unittest.main()