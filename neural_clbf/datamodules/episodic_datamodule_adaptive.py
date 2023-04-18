"""DataModule for aggregating data points over a series of episodes, with additional
sampling from fixed sets.

Code based on the Pytorch Lightning example at
pl_examples/domain_templates/reinforce_learn_Qnet.py
"""
from typing import List, Callable, Tuple, Dict, Optional, Union

import torch
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader

import polytope as pc
import numpy as np

from neural_clbf.systems import ControlAffineSystem
from neural_clbf.systems.adaptive import ControlAffineParameterAffineSystem


class EpisodicDataModuleAdaptive(pl.LightningDataModule):
    """
    DataModule for sampling from a replay buffer
    """

    def __init__(
        self,
        model: ControlAffineParameterAffineSystem,
        initial_domain: List[Tuple[float, float]],
        trajectories_per_episode: int = 100,
        trajectory_length: int = 5000,
        fixed_samples: int = 100000,
        max_points: int = 10000000,
        val_split: float = 0.1,
        batch_size: int = 64,
        quotas: Optional[Dict[str, float]] = None,
        device: str = "cpu",
        num_workers: int = 10,
    ):
        """Initialize the DataModule

        args:
            model: the dynamics model to use in simulation
            initial_domain: the initial_domain to sample from, expressed as a list of
                             tuples denoting the min/max range for each dimension
            trajectories_per_episode: the number of rollouts to conduct at each episode
            trajectory_length: the number of samples to collect in each trajectory
            fixed_samples: the number of uniform samples to collect
            val_split: the fraction of sampled data to reserve for validation
            batch_size: the batch size
            quotas: a dictionary specifying the minimum percentage of the
                    fixed samples that should be taken from the safe,
                    unsafe, boundary, and goal sets. Expects keys to be either "safe",
                    "unsafe", "boundary", or "goal".
        """
        super().__init__()

        self.model = model
        self.n_dims = model.n_dims  # copied for convenience
        self.n_params = model.n_params

        # Save the parameters
        self.trajectories_per_episode = trajectories_per_episode
        self.trajectory_length = trajectory_length
        self.fixed_samples = fixed_samples
        self.max_points = max_points
        self.val_split = val_split
        self.batch_size = batch_size
        if quotas is not None:
            self.quotas = quotas
        else:
            self.quotas = {}

        # Define the sampling intervals for initial conditions as a hyper-rectangle
        assert len(initial_domain) == self.n_dims
        self.initial_domain = initial_domain

        # Save the min, max, central point, and range tensors
        self.x_max, self.x_min = model.state_limits
        self.x_center = (self.x_max + self.x_min) / 2.0
        self.x_range = self.x_max - self.x_min

        # Save the min, max, central point, and range tensors for the parameters
        V_Theta = pc.extreme(self.model.Theta)
        self.theta_max = [np.max(V_Theta[:, i]) for i in range(V_Theta.shape[1])]
        self.theta_min = [np.min(V_Theta[:, i]) for i in range(V_Theta.shape[1])]

        theta_limits = np.vstack((self.theta_max, self.theta_min))
        self.center = [np.mean(theta_limits[:, i]) for i in range(theta_limits.shape[1])]

        self.device = device  # Save the device
        self.num_workers = num_workers  # Save the number of workers (CPUs?)

        # Data buffers
        self.training_data = None
        self.validation_data = None

    def sample_trajectories(
        self, simulator: Callable[[torch.Tensor, int], torch.Tensor]
    ) -> torch.Tensor:
        """
        Generate new data points by simulating a bunch of trajectories

        args:
            simulator: a function that simulates the given initial conditions out for
                       the specified number of timesteps
        """


        # Start by sampling from initial conditions from the given region
        x_init = torch.zeros(
            (self.trajectories_per_episode, self.n_dims),
            device=self.device,
        ).uniform_(
            0.0, 1.0
        )
        for i in range(self.n_dims):
            min_val, max_val = self.initial_domain[i]
            x_init[:, i] = x_init[:, i] * (max_val - min_val) + min_val

        # Simulate each initial condition out for the specified number of steps
        theta_init = torch.zeros((self.trajectories_per_episode, self.model.n_params), device=self.device)
        if self.trajectories_per_episode > 0:
            theta_init[:, :] = self.model.sample_Theta_space(self.trajectories_per_episode).type_as(x_init)
        x_sim, theta_sim, theta_hat_sim = simulator(x_init, theta_init, self.trajectory_length)

        # Reshape the data into a single replay buffer
        x_sim = x_sim.view(-1, self.n_dims)
        theta_sim = theta_sim.view(-1, self.n_params)
        theta_hat_sim = theta_hat_sim.view(-1, self.n_params)

        # Return the sampled data
        return x_sim, theta_sim, theta_hat_sim

    def sample_fixed(self) -> torch.Tensor:
        """
        Description:
            Generate new data points by sampling uniformly from the state and parameter space
        """

        x_samples = []
        theta_samples = []
        theta_hat_samples = []

        # Figure out how many points are to be sampled at random, how many from the
        # goal, safe, or unsafe regions specifically
        allocated_samples = 0
        for region_name, quota in self.quotas.items():
            num_samples = int(self.fixed_samples * quota)
            allocated_samples += num_samples

            if region_name == "goal":
                _, x_sample_group, theta_sample_group = self.model.sample_goal(num_samples)
            elif region_name == "safe":
                _, x_sample_group, theta_sample_group = self.model.sample_safe(num_samples)
            elif region_name == "unsafe":
                _, x_sample_group, theta_sample_group = self.model.sample_unsafe(num_samples)
            elif region_name == "boundary":
                _, x_sample_group, theta_sample_group = self.model.sample_boundary(num_samples)

            # Append new samples to the respective lists
            x_samples.append(x_sample_group)
            theta_samples.append(theta_sample_group)
            theta_hat_samples.append(self.model.sample_Theta_space(num_samples))


        # Sample all remaining points uniformly at random
        free_samples = self.fixed_samples - allocated_samples
        assert free_samples >= 0
        x_samples.append(self.model.sample_state_space(free_samples))

        # Sample parameter estimates
        theta_hat_samples.append(
            self.model.sample_Theta_space(free_samples)
        )

        # Sample parameter estimates
        theta_samples.append(
            self.model.sample_Theta_space(free_samples)
        )
        for sample_cluster in x_samples:
            print("sample_cluster.shape = ", sample_cluster.shape)
        #
        # print("x_samples = ", torch.vstack(x_samples))
        # print("theta_samples = ", torch.vstack(theta_samples))
        # print("theta_hat_samples = ", torch.vstack(theta_hat_samples))


        return torch.vstack(x_samples), torch.vstack(theta_samples), torch.vstack(theta_hat_samples)

    def prepare_data(self):
        """Create the dataset"""
        # Get some data points from simulations
        x_sim, theta_sim, theta_hat_sim = self.sample_trajectories(self.model.nominal_simulator)

        # Augment those points with samples from the fixed range
        x_sample, theta_sample, theta_hat_sample = self.sample_fixed()
        print("x_sim.device = ", x_sim.device, ", x_sample.device = ", x_sample.device)
        x = torch.cat((x_sim, x_sample), dim=0)
        theta = torch.cat((theta_sim, theta_sample), dim=0)
        theta_hat = torch.cat((theta_hat_sim, theta_hat_sample), dim=0)

        # Randomly split data into training and test sets
        random_indices = torch.randperm(x.shape[0], device=self.device)
        val_pts = int(x.shape[0] * self.val_split)
        validation_indices = random_indices[:val_pts]
        training_indices = random_indices[val_pts:]
        self.x_training = x[training_indices]
        self.x_validation = x[validation_indices]

        self.theta_h_training = theta_hat[training_indices]
        self.theta_h_validation = theta_hat[validation_indices]

        self.theta_training = theta[training_indices]
        self.theta_validation = theta[validation_indices]

        print("Full dataset:")
        print(f"\t{self.x_training.shape[0]} training xs")
        print(f"\t{self.x_validation.shape[0]} validation xs")
        print(f"\t{self.theta_training.shape[0]} training thetas")
        print(f"\t{self.theta_validation.shape[0]} validation thetas")
        print(f"\t{self.theta_h_training.shape[0]} training theta estimates")
        print(f"\t{self.theta_h_validation.shape[0]} validation theta estimates")
        print("\t----------------------")
        print(f"\t{self.model.goal_mask(self.x_training, self.theta_training).sum()} goal points")
        print(f"\t({self.model.goal_mask(self.x_validation, self.theta_validation).sum()} val)")
        print(f"\t{self.model.safe_mask(self.x_training, self.theta_training).sum()} safe points")
        print(f"\t({self.model.safe_mask(self.x_validation, self.theta_validation).sum()} val)")
        print(f"\t{self.model.unsafe_mask(self.x_training, self.theta_training).sum()} unsafe points")
        print(f"\t({self.model.unsafe_mask(self.x_validation, self.theta_validation).sum()} val)")
        print(f"\t{self.model.boundary_mask(self.x_training, self.theta_training).sum()} boundary points")
        print(f"\t({self.model.boundary_mask(self.x_validation, self.theta_validation).sum()} val)")

        # Turn these into tensor datasets
        self.training_data = TensorDataset(
            self.x_training,
            self.theta_training,
            self.theta_h_training,
            self.model.goal_mask(self.x_training, self.theta_training),
            self.model.safe_mask(self.x_training, self.theta_training),
            self.model.unsafe_mask(self.x_training, self.theta_training),
        )
        self.validation_data = TensorDataset(
            self.x_validation,
            self.theta_validation,
            self.theta_h_validation,
            self.model.goal_mask(self.x_validation, self.theta_validation),
            self.model.safe_mask(self.x_validation, self.theta_validation),
            self.model.unsafe_mask(self.x_validation, self.theta_validation),
        )

    def add_data(self, simulator: Callable[[torch.Tensor, int], torch.Tensor]):
        """
        Augment the training and validation datasets by simulating and sampling

        args:
            simulator: a function that simulates the given initial conditions out for
                       the specified number of timesteps
        """
        print("\nAdding data!\n")
        # Get some data points from simulations
        x_sim, theta_sim, theta_hat_sim = self.sample_trajectories(simulator)

        # # Augment those points with samples from the fixed range
        x_sample, theta_sample, theta_hat_sample = self.sample_fixed()
        x = torch.cat((x_sim.type_as(x_sample), x_sample), dim=0)
        x = x.type_as(self.x_training)
        theta = torch.cat((theta_sim, theta_sample), dim=0)
        theta = theta.type_as(self.theta_training)
        theta_hat = torch.cat((theta_hat_sim, theta_hat_sample), dim=0)
        theta_hat = theta_hat.type_as(self.theta_h_training)

        print(f"Sampled {x.shape[0]} new states, {theta.shape[0]} true parameters and {theta_hat.shape[0]} new parameter estimates")

        # Randomly split data into training and test sets
        random_indices = torch.randperm(x.shape[0], device=self.device)
        val_pts = int(x.shape[0] * self.val_split)
        validation_indices = random_indices[:val_pts]
        training_indices = random_indices[val_pts:]

        print(f"\t{training_indices.shape[0]} train, {validation_indices.shape[0]} val")

        # Augment the existing data with the new points
        self.x_training = torch.cat((self.x_training, x[training_indices]))
        self.x_validation = torch.cat((self.x_validation, x[validation_indices]))

        self.theta_training = torch.cat((self.theta_training, theta[training_indices]))
        self.theta_validation = torch.cat((self.theta_validation, theta[validation_indices]))

        self.theta_h_training = torch.cat((self.theta_h_training, theta_hat[training_indices]))
        self.theta_h_validation = torch.cat((self.theta_h_validation, theta_hat[validation_indices]))

        # If we've exceeded the maximum number of points, forget the oldest
        if self.x_training.shape[0] + self.x_validation.shape[0] > self.max_points:
            print("Sample budget exceeded! Forgetting...")
            # Figure out how many training and validation points we should have
            n_val = int(self.max_points * self.val_split)
            n_train = self.max_points - n_val
            # And then keep only the most recent points
            self.x_training = self.x_training[-n_train:]
            self.x_validation = self.x_validation[-n_val:]
            self.theta_training = self.theta_training[-n_train:]
            self.theta_validation = self.theta_validation[-n_val:]
            self.theta_h_training = self.theta_h_training[-n_train:]
            self.theta_h_validation = self.theta_h_validation[-n_val:]

        print("Full dataset:")
        print(f"\t{self.x_training.shape[0]} training states")
        print(f"\t{self.x_validation.shape[0]} validation states")
        print(f"\t{self.theta_training.shape[0]} training thetas")
        print(f"\t{self.theta_validation.shape[0]} validation thetas")
        print(f"\t{self.theta_h_training.shape[0]} training theta estimates")
        print(f"\t{self.theta_h_validation.shape[0]} validation theta estimates")
        print("\t----------------------")
        print(f"\t{self.model.goal_mask(self.x_training, self.theta_training).sum()} goal points")
        print(f"\t({self.model.goal_mask(self.x_validation, self.theta_validation).sum()} val)")
        print(f"\t{self.model.safe_mask(self.x_training, self.theta_training).sum()} safe points")
        print(f"\t({self.model.safe_mask(self.x_validation, self.theta_validation).sum()} val)")
        print(f"\t{self.model.unsafe_mask(self.x_training, self.theta_training).sum()} unsafe points")
        print(f"\t({self.model.unsafe_mask(self.x_validation, self.theta_validation).sum()} val)")

        # Save the new datasets
        self.training_data = TensorDataset(
            self.x_training,
            self.theta_training,
            self.theta_h_training,
            self.model.goal_mask(self.x_training, self.theta_training),
            self.model.safe_mask(self.x_training, self.theta_training),
            self.model.unsafe_mask(self.x_training, self.theta_training),
        )
        self.validation_data = TensorDataset(
            self.x_validation,
            self.theta_validation,
            self.theta_h_validation,
            self.model.goal_mask(self.x_validation, self.theta_validation),
            self.model.safe_mask(self.x_validation, self.theta_validation),
            self.model.unsafe_mask(self.x_validation, self.theta_validation),
        )

    def setup(self, stage=None):
        """Setup -- nothing to do here"""
        pass

    def train_dataloader(self):
        """
        train_dataloader
        Description:
            Make the DataLoader for training data
        """
        # Check to make sure self.training_data exists
        if self.training_data is None:
            self.prepare_data()

        # Construct Dataloader
        dl = DataLoader(
            self.training_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        # for x in dl:
        #     for x_k in x:
        #         x_k = x_k.to(self.device, non_blocking=True)

        return dl

    def val_dataloader(self):
        """Make the DataLoader for validation data"""

        # Check to make sure self.validation_data exists
        if self.validation_data is None:
            self.prepare_data()

        dl = DataLoader(
            self.validation_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        # for x in dl:
        #     for x_k in x:
        #         x_k = x_k.to(self.device, non_blocking=True)
            # x = x.to(self.device, non_blocking=True)

        return dl
