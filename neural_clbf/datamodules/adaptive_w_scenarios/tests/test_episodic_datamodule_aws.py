"""
test_episodic_datamodule_aws.py
Description:
    Tests for episodic datamodule with adaptive parameters and scenario
    vectors.
"""

import unittest
import torch
import polytope as pc


from neural_clbf.systems.adaptive_w_scenarios import ControlAffineParameterAffineSystem2
from neural_clbf.datamodules.adaptive_w_scenarios import(
    EpisodicDataModuleAdaptiveWScenarios
)
from neural_clbf.systems.adaptive_w_scenarios import (
    AdaptivePusherSliderStickingForceInput_NObstacles,
)

class TestEpisodicDataModuleAWS(unittest.TestCase):
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

        Theta = pc.box2poly([(0.0, 1.0), (0.0, 1.0)])

        # Create Object
        aps = AdaptivePusherSliderStickingForceInput_NObstacles(
            nominal_scenario, Theta,
        )

        return aps

    def get_shared_edm1(self):
        """
        get_shared_edm1
        Description:
            Defines a shared edm for the first few tests.
        """
        # Constants

        # Algorithm
        if hasattr(self, edm1):
            return self.edm1
        else:
            # Create edm
            sys0 = self.get_ps1()

            X0 = torch.tensor([
                [-0.6, -0.4],
                [-0.6, -0.4],
                [torch.pi / 2.0, torch.pi],
            ])

            self.edm1 = EpisodicDataModuleAdaptiveWScenarios(
                sys0, X0, trajectories_per_episode=10, trajectory_length=10, fixed_samples=200,
            )

        return None



    def test_init1(self):
        """
        test_init1
        Description:
            Tests that the EpisodicDataModuleAdaptiveWScenarios
            initialization.
        """

        # Constants
        sys0 = self.get_ps1()

        X0 = torch.tensor([
            [-0.6, -0.4],
            [-0.6, -0.4],
            [torch.pi/2.0, torch.pi],
        ])

        # Create Object
        edm = EpisodicDataModuleAdaptiveWScenarios(
            sys0, X0,
        )

        self.assertEqual(edm.trajectories_per_episode, 100)
        self.assertEqual(edm.batch_size, 64)

    def test_sample_trajectories1(self):
        """
        test_sample_trajectories1
        Description:
            Tests the ability to sample trajectories.
        """
        # Constants
        sys0 = self.get_ps1()

        X0 = torch.tensor([
            [-0.6, -0.4],
            [-0.6, -0.4],
            [torch.pi / 2.0, torch.pi],
        ])

        # Create EpisodicDataModule
        edm = EpisodicDataModuleAdaptiveWScenarios(
            sys0, X0, trajectory_length=4,
        )

        # Algorithms
        x_sim, theta_sim, theta_hat_sim, scen_sim = edm.sample_trajectories(sys0.nominal_simulator)

        self.assertEqual(
            x_sim.shape, (edm.trajectories_per_episode * edm.trajectory_length, sys0.n_dims)
        )

        self.assertEqual(
            theta_sim.shape, (edm.trajectories_per_episode * edm.trajectory_length, sys0.n_params)
        )

        self.assertEqual(
            theta_hat_sim.shape, (edm.trajectories_per_episode * edm.trajectory_length, sys0.n_params)
        )

        self.assertEqual(
            scen_sim.shape, (edm.trajectories_per_episode * edm.trajectory_length, sys0.n_scenario)
        )

    def test_sample_fixed1(self):
        """
        test_sample_fixed1()
        Description:
            Testing the function sample_fixed() for our own ability to sample points correctly.
        """
        # Constants
        sys0 = self.get_ps1()

        X0 = torch.tensor([
            [-0.6, -0.4],
            [-0.6, -0.4],
            [torch.pi / 2.0, torch.pi],
        ])

        # Create Episodic Data Module
        edm = EpisodicDataModuleAdaptiveWScenarios(
            sys0, X0, trajectory_length=4, fixed_samples=100,
        )

        # Sample!
        x_samples, theta_samples, theta_hat_samples, scen_samples = edm.sample_fixed()

        self.assertEqual(
            x_samples.shape, (edm.fixed_samples, sys0.n_dims),
        )

        self.assertEqual(
            theta_samples.shape, (edm.fixed_samples, sys0.n_params),
        )

        self.assertEqual(
            theta_hat_samples.shape, (edm.fixed_samples, sys0.n_params),
        )

        self.assertEqual(
            scen_samples.shape, (edm.fixed_samples, sys0.n_scenario),
        )

    def test_prepare_data1(self):
        """
        test_prepare_data1
        Description:
            Tests the prepare_data function in our datamodule.
        """
        # Constants
        sys0 = self.get_ps1()

        X0 = torch.tensor([
            [-0.6, -0.4],
            [-0.6, -0.4],
            [torch.pi / 2.0, torch.pi],
        ])

        # Create Episodic Data Module
        edm = EpisodicDataModuleAdaptiveWScenarios(
            sys0, X0, trajectory_length=4, fixed_samples=100,
        )

        # Sample!
        edm.prepare_data()

        n_samples = edm.trajectories_per_episode*edm.trajectory_length+edm.fixed_samples
        n_validation_samples = n_samples * edm.val_split
        n_training_samples = n_samples - n_validation_samples
        self.assertEqual(
            edm.x_training.shape,
            (n_training_samples, sys0.n_dims)
        )

    def test_prepare_data2(self):
        """
        test_prepare_data2
        Description:
            Tests the prepare_data function in our datamodule. Uses the quotas keyword.
        """
        # Constants
        sys0 = self.get_ps1()

        X0 = torch.tensor([
            [-0.6, -0.4],
            [-0.6, -0.4],
            [torch.pi / 2.0, torch.pi],
        ])

        # Create Episodic Data Module
        edm = EpisodicDataModuleAdaptiveWScenarios(
            sys0, X0, trajectory_length=4, fixed_samples=100,
            quotas={'unsafe': 0.1, 'goal': 0.1},
        )

        # Sample!
        edm.prepare_data()

        n_samples = edm.trajectories_per_episode*edm.trajectory_length+edm.fixed_samples
        n_validation_samples = n_samples * edm.val_split
        n_training_samples = n_samples - n_validation_samples
        self.assertEqual(
            edm.x_training.shape,
            (n_training_samples, sys0.n_dims)
        )

    def test_add_data1(self):
        """
        test_add_data1
        Description:
            Tests the ability of the data module to add data to the current dataset.
        """
        # Constants
        sys0 = self.get_ps1()

        X0 = torch.tensor([
            [-0.6, -0.4],
            [-0.6, -0.4],
            [torch.pi / 2.0, torch.pi],
        ])

        # Create Episodic Data Module
        edm = EpisodicDataModuleAdaptiveWScenarios(
            sys0, X0, trajectory_length=4, fixed_samples=100,
            quotas={'unsafe': 0.1, 'goal': 0.1},
        )

        # Sample!
        edm.prepare_data()
        n_samples0 = edm.trajectories_per_episode*edm.trajectory_length+edm.fixed_samples
        edm.add_data(
            sys0.nominal_simulator,
        )

        self.assertNotEqual(n_samples0, edm.x_training.shape[0])
        self.assertEqual(n_samples0*(1.0-edm.val_split)*2, edm.x_training.shape[0])


if __name__ == '__main__':
    unittest.main()