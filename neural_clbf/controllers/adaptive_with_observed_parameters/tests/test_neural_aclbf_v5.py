"""
test_neural_aclbf_v5
Description:
    Testing the version of the barrier where we have the scenario parameters given as input to the NN.
"""
import unittest
import polytope as pc
import torch

from neural_clbf.systems.adaptive_w_scenarios import (
    AdaptivePusherSliderStickingForceInput_NObstacles, ControlAffineParameterAffineSystem2,
)
from neural_clbf.systems.adaptive_w_scenarios import AdaptivePusherSliderStickingForceInput_NObstacles as apssfi
from neural_clbf.controllers.adaptive_with_observed_parameters import (
    NeuralaCLBFControllerV5,
)
from neural_clbf.datamodules.adaptive_w_scenarios import (
    EpisodicDataModuleAdaptiveWScenarios,
)
from neural_clbf.experiments import (
    ExperimentSuite,
)
from neural_clbf.experiments.adaptive_w_observed_parameters import (
    AdaptiveCLFContourExperiment3,
)

class TestNeuralACLBFv5(unittest.TestCase):
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

        nominal_scenario["goal_x"] = 0.5
        nominal_scenario["goal_y"] = 0.5

        s_width = 0.09
        Theta = pc.box2poly([
            (-0.03 + s_width/2.0, 0.03 + s_width/2.0),
            (-0.03, 0.03),
        ])

        # Create Object
        aps = AdaptivePusherSliderStickingForceInput_NObstacles(
            nominal_scenario, Theta,
        )

        return aps

    def get_ps_with_neuralaclbfcontroller1(self) -> (ControlAffineParameterAffineSystem2, NeuralaCLBFControllerV5):
        """
        system, controller = self.get_lsm_with_neuralaclbfcontroller1()
        Description:
            Creates a LoadSharingManipulator example system for tests.
        """

        # Constants
        theta_lb = [0.175, 0.4, 0.2]
        theta_ub = [0.225, 0.65, 0.3]

        rollout_experiment_horizon = 10.0

        start_x = torch.tensor([
            [0.3, -0.3, 0.4],
            [0.35, -0.25, 0.3],
            [0.15, -0.2, 0.2],
        ])

        # Get System
        sys0 = self.get_ps1()

        # Initialize the DataModule
        initial_conditions = [
            (-0.6, -0.4),  # p_x
            (-0.6, -0.4),  # p_y
            (0.0, torch.pi/4),  # p_z
        ]
        datamodule = EpisodicDataModuleAdaptiveWScenarios(
            sys0,
            initial_conditions,
            trajectories_per_episode=10,
            trajectory_length=20,
            fixed_samples=10000,
            max_points=100000,
            val_split=0.1,
            batch_size=64,
            quotas={"safe": 0.2, "unsafe": 0.2, "goal": 0.2},
            device="cpu",
            num_workers=10,
        )

        # Define Experiment Suite
        contour_exp_theta_index = 0
        lb_Vcontour = theta_lb[contour_exp_theta_index]
        ub_Vcontour = theta_ub[contour_exp_theta_index]
        theta_range_Vcontour = ub_Vcontour - lb_Vcontour
        V_contour_experiment = AdaptiveCLFContourExperiment3(
            "V_Contour",
            x_domain=[
                (sys0.state_limits[1][apssfi.S_X],
                 sys0.state_limits[0][apssfi.S_X]),
            ],
            theta_domain=[(lb_Vcontour - 0.2 * theta_range_Vcontour, ub_Vcontour + 0.2 * theta_range_Vcontour)],
            n_grid=30,
            x_axis_index=apssfi.S_X,
            theta_axis_index=contour_exp_theta_index,
            x_axis_label="$r_1$",
            theta_axis_label="$\\theta_" + str(contour_exp_theta_index) + "$",  # "$\\dot{\\theta}$",
            plot_unsafe_region=False,
        )
        # rollout_experiment2 = RolloutStateParameterSpaceExperimentMultiple(
        #     "Rollout (Multiple Slices)",
        #     start_x,
        #     [LoadSharingManipulator.P_X, apssfi.V_X, LoadSharingManipulator.P_Y],
        #     ["$r_1$", "$v_1$", "$r_2$"],
        #     [LoadSharingManipulator.P_X_DES, apssfi.P_X_DES, LoadSharingManipulator.P_Y],
        #     ["$\\theta_1 (r_1^{(d)})$", "$\\theta_1 (r_1^{(d)})$", "$\\theta_1 (r_2^{(d)})$"],
        #     scenarios=sys0.scenarios,
        #     n_sims_per_start=1,
        #     t_sim=rollout_experiment_horizon,
        # )
        experiment_suite = ExperimentSuite([V_contour_experiment])

        # Define the controller
        controller = NeuralaCLBFControllerV5(
            sys0,
            datamodule,
            experiment_suite,
            controller_period=sys0.controller_dt,
        )

        return sys0, controller

    def test_init1(self):
        """
        test_init1
        Description:
            This test verifies that we can correctly initialize the Neural aCLBF
        """
        # Constants
        theta_lb = [-0.03, -0.1]
        theta_ub = [-0.02, 0.1]

        # Get System
        sys0 = self.get_ps1()

        # Initialize the DataModule
        initial_conditions = [
            (-0.6, -0.4),  # s_x
            (-0.6, -0.4),  # s_y
            (0.0, 0.7),  # s_theta
        ]
        datamodule = EpisodicDataModuleAdaptiveWScenarios(
            sys0,
            initial_conditions,
            trajectories_per_episode=10,
            trajectory_length=20,
            fixed_samples=10000,
            max_points=100000,
            val_split=0.1,
            batch_size=64,
            quotas={"safe": 0.2, "unsafe": 0.2, "goal": 0.2},
            device="cpu",
            num_workers=10,
        )

        # Define Experiment Suite
        contour_exp_theta_index = 0
        lb_Vcontour = theta_lb[contour_exp_theta_index]
        ub_Vcontour = theta_ub[contour_exp_theta_index]
        theta_range_Vcontour = ub_Vcontour - lb_Vcontour
        V_contour_experiment = AdaptiveCLFContourExperiment3(
            "V_Contour",
            x_domain=[
                (sys0.state_limits[1][apssfi.S_X],
                 sys0.state_limits[0][apssfi.S_X]),
            ],
            theta_domain=[(lb_Vcontour - 0.2 * theta_range_Vcontour, ub_Vcontour + 0.2 * theta_range_Vcontour)],
            n_grid=30,
            x_axis_index=apssfi.S_X,
            theta_axis_index=contour_exp_theta_index,
            x_axis_label="$r_1$",
            theta_axis_label="$\\theta_" + str(contour_exp_theta_index) + "$",  # "$\\dot{\\theta}$",
            plot_unsafe_region=False,
        )
        # rollout_experiment2 = RolloutStateParameterSpaceExperimentMultiple(
        #     "Rollout (Multiple Slices)",
        #     start_x,
        #     [LoadSharingManipulator.P_X, LoadSharingManipulator.V_X, LoadSharingManipulator.P_Y],
        #     ["$r_1$", "$v_1$", "$r_2$"],
        #     [LoadSharingManipulator.P_X_DES, LoadSharingManipulator.P_X_DES, LoadSharingManipulator.P_Y],
        #     ["$\\theta_1 (r_1^{(d)})$", "$\\theta_1 (r_1^{(d)})$", "$\\theta_1 (r_2^{(d)})$"],
        #     scenarios=sys0.scenarios,
        #     n_sims_per_start=1,
        #     t_sim=rollout_experiment_horizon,
        # )
        experiment_suite = ExperimentSuite([V_contour_experiment])

        # Define the controller
        controller = NeuralaCLBFControllerV5(
            sys0,
            datamodule,
            experiment_suite,
            controller_period=sys0.controller_dt,
        )

        self.assertEqual(
            controller.controller_period, sys0.controller_dt,
        )

    def test_V_with_jacobian1(self):
        """
        test_V_with_jacobian1
        Description:
            Verifies that the function V_with_jacobian works for a test input.
        """
        # Constants
        system0, controller = self.get_ps_with_neuralaclbfcontroller1()

        # Get the test input
        x = torch.tensor([
            [0.1, 0.4, 0.0],
        ])
        theta = torch.tensor([
            [0.025, 0.0],
        ])
        scen = torch.tensor([
            [0.0, 0.5, 0.0, -0.3, 0.6, 0.0],
        ])

        # Call function
        V, dVdx, dVdtheta = controller.V_with_jacobian(x, theta, scen)



if __name__ == '__main__':
    unittest.main()