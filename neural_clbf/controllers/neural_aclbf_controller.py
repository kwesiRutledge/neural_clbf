import itertools
from typing import Tuple, List, Optional
from collections import OrderedDict
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from neural_clbf.systems.adaptive import ControlAffineParameterAffineSystem
from neural_clbf.systems.utils import ScenarioList
from neural_clbf.controllers.adaptive.aclf_controller import aCLFController
from neural_clbf.controllers.controller_utils import normalize_with_angles
from neural_clbf.datamodules.episodic_datamodule import EpisodicDataModule
from neural_clbf.experiments import ExperimentSuite

import polytope as pc


class NeuralaCLBFController(pl.LightningModule, aCLFController):
    """
    A neural aCLBF controller. Differs from the CLFController in that it uses a
    neural network to learn the CLF, and it turns it from a CLF to a CLBF by making sure
    that a level set of the CLF separates the safe and unsafe regions.

    More specifically, the CLBF controller looks for a V such that

    V(goal,theta_hat) = 0
    V >= 0
    V(safe, theta_hat) < c
    V(unsafe, theta_hat) > c
    dV/dt <= -lambda V

    This proves forward invariance of the c-sublevel set of V, and since the safe set is
    a subset of this sublevel set, we prove that the unsafe region is not reachable from
    the safe region. We also prove convergence to a point.
    """

    def __init__(
        self,
        dynamics_model: ControlAffineParameterAffineSystem,
        scenarios: ScenarioList,
        datamodule: EpisodicDataModule,
        experiment_suite: ExperimentSuite,
        clbf_hidden_layers: int = 2,
        clbf_hidden_size: int = 48,
        clf_lambda: float = 1.0,
        safe_level: float = 1.0,
        clf_relaxation_penalty: float = 50.0,
        controller_period: float = 0.01,
        primal_learning_rate: float = 1e-3,
        epochs_per_episode: int = 5,
        penalty_scheduling_rate: float = 0.0,
        num_init_epochs: int = 5,
        barrier: bool = True,
        add_nominal: bool = False,
        normalize_V_nominal: bool = False,
        saved_Vnn: torch.nn.Sequential = None,
    ):
        """Initialize the controller.

        args:
            dynamics_model: the control-affine dynamics of the underlying system
            scenarios: a list of parameter scenarios to train on
            experiment_suite: defines the experiments to run during training
            clbf_hidden_layers: number of hidden layers to use for the CLBF network
            clbf_hidden_size: number of neurons per hidden layer in the CLBF network
            clf_lambda: convergence rate for the CLBF
            safe_level: safety level set value for the CLBF
            clf_relaxation_penalty: the penalty for relaxing CLBF conditions.
            controller_period: the timestep to use in simulating forward Vdot
            primal_learning_rate: the learning rate for SGD for the network weights,
                                  applied to the CLBF decrease loss
            epochs_per_episode: the number of epochs to include in each episode
            penalty_scheduling_rate: the rate at which to ramp the rollout relaxation
                                     penalty up to clf_relaxation_penalty. Set to 0 to
                                     disable penalty scheduling (use constant penalty)
            num_init_epochs: the number of epochs to pretrain the controller on the
                             linear controller
            barrier: if True, train the CLBF to act as a barrier functions. If false,
                     effectively trains only a CLF.
            add_nominal: if True, add the nominal V
            normalize_V_nominal: if True, normalize V_nominal so that its average is 1
        """
        super(NeuralaCLBFController, self).__init__(
            dynamics_model=dynamics_model,
            scenarios=scenarios,
            experiment_suite=experiment_suite,
            clf_lambda=clf_lambda,
            clf_relaxation_penalty=clf_relaxation_penalty,
            controller_period=controller_period,
        )
        self.save_hyperparameters()

        # Save the provided model
        # self.dynamics_model = dynamics_model
        self.scenarios = scenarios
        self.n_scenarios = len(scenarios)

        # Save the datamodule
        self.datamodule = datamodule

        # Save the experiments suits
        self.experiment_suite = experiment_suite

        # Save the other parameters
        self.safe_level = safe_level
        self.unsafe_level = safe_level
        self.primal_learning_rate = primal_learning_rate
        self.epochs_per_episode = epochs_per_episode
        self.penalty_scheduling_rate = penalty_scheduling_rate
        self.num_init_epochs = num_init_epochs
        self.barrier = barrier
        self.add_nominal = add_nominal
        self.normalize_V_nominal = normalize_V_nominal
        self.V_nominal_mean = 1.0

        # Compute and save the center and range of the state variables
        x_max, x_min = dynamics_model.state_limits
        self.x_center = (x_max + x_min) / 2.0
        self.x_range = (x_max - x_min) / 2.0

        # Scale to get the input between (-k, k), centered at 0
        self.k = 1.0
        self.x_range = self.x_range / self.k
        # We shouldn't scale or offset any angle dimensions
        self.x_center[self.dynamics_model.angle_dims] = 0.0
        self.x_range[self.dynamics_model.angle_dims] = 1.0

        # Some of the dimensions might represent angles. We want to replace these
        # dimensions with two dimensions: sin and cos of the angle. To do this, we need
        # to figure out how many numbers are in the expanded state
        n_angles = len(self.dynamics_model.angle_dims)
        self.n_dims_extended = self.dynamics_model.n_dims + n_angles

        # Some of the unknown parameter dimensions might represent angles. We want to replace these
        # dimensions with two dimensions: sin and cos of the angle. To do this, we need
        # to figure out how many numbers are in the expanded state
        n_param_angles = len(self.dynamics_model.parameter_angle_dims)
        self.n_params_extended = self.dynamics_model.n_params + n_param_angles

        # Compute and save the center and range of the Theta variables?

        # Define the CLBF network, which we denote V
        self.clbf_hidden_layers = clbf_hidden_layers
        self.clbf_hidden_size = clbf_hidden_size
        # We're going to build the network up layer by layer, starting with the input
        self.V_layers: OrderedDict[str, nn.Module] = OrderedDict()
        self.V_layers["input_linear"] = nn.Linear(
            self.n_dims_extended+self.n_params_extended, self.clbf_hidden_size
        )
        self.V_layers["input_activation"] = nn.Tanh()
        for i in range(self.clbf_hidden_layers):
            self.V_layers[f"layer_{i}_linear"] = nn.Linear(
                self.clbf_hidden_size, self.clbf_hidden_size
            )

            if i < self.clbf_hidden_layers - 1:
                self.V_layers[f"layer_{i}_activation"] = nn.Tanh()
        # self.V_layers["output_linear"] = nn.Linear(self.clbf_hidden_size, 1)
        self.V_nn = nn.Sequential(self.V_layers)

        if saved_Vnn is not None:
            for layer_idx in range(len(saved_Vnn)):
                if isinstance(saved_Vnn[layer_idx], nn.Linear):
                    self.V_nn[layer_idx].weight = saved_Vnn[layer_idx].weight

    def prepare_data(self):
        return self.datamodule.prepare_data()

    def setup(self, stage: Optional[str] = None):
        return self.datamodule.setup(stage)

    def train_dataloader(self):
        return self.datamodule.train_dataloader()

    def val_dataloader(self):
        return self.datamodule.val_dataloader()

    def test_dataloader(self):
        return self.datamodule.test_dataloader()

    def V_with_jacobian(self, x: torch.Tensor, theta_hat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the CLBF value and its Jacobian

        args:
            x: bs x self.dynamics_model.n_dims the points at which to evaluate the CLBF
            theta_hat: bs x self.dynamics_model.n_params the points at which to evaluate the CLBF
        returns:
            V: bs tensor of CLBF values
            JVx: bs x 1 x self.dynamics_model.n_dims Jacobian of each row of V wrt x
        """
        # Apply the offset and range to normalize about zero
        x_norm = normalize_with_angles(self.dynamics_model, x)
        theta_hat_norm = normalize_with_angles(self.dynamics_model, theta_hat, angle_dims = self.dynamics_model.parameter_angle_dims)

        # Compute the CLBF layer-by-layer, computing the Jacobian alongside

        # We need to initialize the Jacobian to reflect the normalization that's already
        # been done to x
        bs = x_norm.shape[0]
        # JVx = torch.zeros(
        #     (bs, self.n_dims_extended, self.dynamics_model.n_dims)
        # ).type_as(x)
        # # and for each non-angle dimension, we need to scale by the normalization
        # for dim in range(self.dynamics_model.n_dims):
        #     JVx[:, dim, dim] = 1.0 / self.x_range[dim].type_as(x)
        #
        # # And adjust the Jacobian for the angle dimensions
        # for offset, sin_idx in enumerate(self.dynamics_model.angle_dims):
        #     cos_idx = self.dynamics_model.n_dims + offset
        #     JVx[:, sin_idx, sin_idx] = x_norm[:, cos_idx]
        #     JVx[:, cos_idx, sin_idx] = -x_norm[:, sin_idx]
        #
        # # Create jacobian with respect to theta
        # JVth = torch.zeros(
        #     (bs, self.n_params_extended, self.dynamics_model.n_params)
        # ).type_as(theta_hat)
        # # and for each non-angle dimension, we need to scale by the normalization
        # for dim in range(self.dynamics_model.n_params):
        #     JVth[:, dim, dim] = 1.0 / self.x_range[dim].type_as(theta_hat)
        #
        # # And adjust the Jacobian for the angle dimensions
        # for offset, sin_idx in enumerate(self.dynamics_model.parameter_angle_dims):
        #     cos_idx = self.dynamics_model.n_dims + offset
        #     JVth[:, sin_idx, sin_idx] = theta_hat_norm[:, cos_idx]
        #     JVth[:, cos_idx, sin_idx] = -theta_hat_norm[:, sin_idx]
        JVxth = torch.zeros(
            (bs, self.n_dims_extended+self.n_params_extended, self.dynamics_model.n_dims+self.dynamics_model.n_params)
        )
        # and for each non-angle dimension, we need to scale by the normalization
        for dim in range(self.dynamics_model.n_dims):
            JVxth[:, dim, dim] = 1.0 / self.x_range[dim].type_as(x)

        for dim in range(self.dynamics_model.n_params):
            dim_mod = dim-self.n_dims_extended
            JVxth[:, dim, dim] = 1.0 / self.x_range[dim_mod].type_as(theta_hat)

        # And adjust the Jacobian for the angle dimensions
        for offset, sin_idx in enumerate(self.dynamics_model.angle_dims):
            cos_idx = self.dynamics_model.n_dims + offset
            JVxth[:, sin_idx, sin_idx] = x_norm[:, cos_idx]
            JVxth[:, cos_idx, sin_idx] = -x_norm[:, sin_idx]

        for offset, sin_idx in enumerate(self.dynamics_model.parameter_angle_dims):
            sin_idx = sin_idx + self.n_dims_extended
            cos_idx = self.n_dims_extended + self.dynamics_model.n_params + offset
            JVxth[:, sin_idx, sin_idx] = theta_hat_norm[:, cos_idx - (self.n_dims_extended )]
            JVxth[:, cos_idx, sin_idx] = -theta_hat_norm[:, sin_idx - (self.n_dims_extended)]

        # Now step through each layer in V
        x_theta_norm = torch.zeros((bs, self.n_dims_extended+self.n_params_extended)).type_as(x)
        # x_theta_norm = torch.cat([x_norm, theta_hat_norm], dim=1)
        x_theta_norm[:, :self.n_dims_extended] = x_norm
        x_theta_norm[:, self.n_dims_extended:] = theta_hat_norm
        V = x_theta_norm
        #print("JVxth shape before loops", JVxth.shape)
        for layer in self.V_nn:
            V = layer(V)

            if isinstance(layer, nn.Linear):
                JVxth = torch.matmul(layer.weight, JVxth)
            elif isinstance(layer, nn.Tanh):
                JVxth = torch.matmul(torch.diag_embed(1 - V ** 2), JVxth)
            elif isinstance(layer, nn.ReLU):
                JVxth = torch.matmul(torch.diag_embed(torch.sign(V)), JVxth)

        # Compute the final activation
        JVxth = torch.bmm(V.unsqueeze(1), JVxth)
        V = 0.5 * (V * V).sum(dim=1)

        if self.add_nominal:
            # Get the nominal Lyapunov function
            P = self.dynamics_model.P.type_as(x)
            x0 = self.dynamics_model.goal_point.type_as(x)

            xtheta0 = torch.zeros(x0.shape)
            xtheta0[:, :self.dynamics_model.n_dims] = x0
            xtheta0[:, self.dynamics_model.n_dims:] = theta0

            xtheta = torch.zeros(xtheta0.shape)
            xtheta[:, :self.dynamics_model.n_dims] = x
            xtheta[:, self.dynamics_model.n_dims:] = theta0

            # Reshape to use pytorch's bilinear function
            # P_x = P[:self.dynamics_model.n_dims, :self.dynamics_model.n_dims]
            # P_x = P_x.reshape(1, self.dynamics_model.n_dims, self.dynamics_model.n_dims)
            # V_nominal = 0.5 * F.bilinear(x - x0, x - x0, P_x).squeeze()
            #
            # # Reshape again to calculate the gradient
            # P_x = P_x.reshape(self.dynamics_model.n_dims, self.dynamics_model.n_dims)
            # JVx_nominal = F.linear(x - x0, P_x) + \
            #               2 * F.linear(theta_hat, P[self.dynamics_model.n_dims:, :self.dynamics_model.n_dims])
            # JVx_nominal = JVx_nominal.reshape(x.shape[0], 1, self.dynamics_model.n_dims)
            #
            # # Create gradient with respect to V
            # P_th = P[self.dynamics_model.n_dims:, self.dynamics_model.n_dims:]
            # JVth_nominal = F.linear(theta_hat, P_th) + \
            #               2 * F.linear(x, P[:, self.dynamics_model.n_dims, self.dynamics_model.n_dims:])
            # JVth_nominal = JVth_nominal.reshape(x.shape[0], 1, self.dynamics_model.n_dims)
            P = P.reshape(1, self.dynamics_model.n_dims+self.dynamics_model.n_params, self.dynamics_model.n_dims+self.dynamics_model.n_params)
            V_nominal = 0.5 * F.bilinear(xtheta - xtheta0, xtheta - xtheta0, P)

            # Reshape again to calculate the gradient
            P = P.reshpae(self.dynamics_model.n_dims+self.dynamics_model.n_params, self.dynamics_model.n_dims+self.dynamics_model.n_params)
            JVxth_nominal = F.linear(xtheta - xtheta0, P)
            JVxth_nominal = JVxth_nominal.reshape(x.shape[0], 1, self.dynamics_model.n_dims+self.dynamics_model.n_params)

            if self.normalize_V_nominal:
                V_nominal /= self.V_nominal_mean
                JVxth_nominal /= self.V_nominal_mean

            V = V + V_nominal
            JVxth = JVxth + JVxth_nominal

        # At the very last second split the gradient.
        JVx = torch.zeros((bs, 1, self.dynamics_model.n_dims))
        JVx[:, 0, :] = JVxth[:, 0, :self.dynamics_model.n_dims]

        JVth = torch.zeros((bs, 1, self.dynamics_model.n_params))
        JVth[:, 0, :] = JVxth[:, 0, self.dynamics_model.n_dims:]

        return V, JVx, JVth

    def forward(self, x_theta_pair):
        """Determine the control input for a given state using a QP

        args:
            x_theta_pair: bs x (self.dynamics_model.n_dims + self.dynamics_model.n_params) tensor of state
        returns:
            u: bs x self.dynamics_model.n_controls tensor of control inputs
        """
        # Create the state-theta_hat pair
        return self.u(x_theta_pair)

    def boundary_loss(
        self,
        x: torch.Tensor,
        theta_hat: torch.Tensor,
        theta: torch.Tensor,
        goal_mask: torch.Tensor,
        safe_mask: torch.Tensor,
        unsafe_mask: torch.Tensor,
        accuracy: bool = False,
    ) -> List[Tuple[str, torch.Tensor]]:
        """
        Evaluate the loss on the CLBF due to boundary conditions

        args:
            x: the points at which to evaluate the loss,
            goal_mask: the points in x marked as part of the goal
            safe_mask: the points in x marked safe
            unsafe_mask: the points in x marked unsafe
            accuracy: if True, return the accuracy (from 0 to 1) as well as the losses
        returns:
            loss: a list of tuples containing ("category_name", loss_value).
        """
        eps = 1e-2
        # Compute loss to encourage satisfaction of the following conditions...
        loss = []

        # print("x", x.shape)
        # print("theta_hat", theta_hat.shape)

        V = self.V(x, theta_hat)

        #   1.) CLBF should be minimized on the goal point
        goal_as_batch = self.dynamics_model.goal_point.type_as(x)
        goal_as_batch = goal_as_batch.reshape(1, self.dynamics_model.n_dims) # Transform goal_point as a batch of one element
        V_goal_pt = self.V(goal_as_batch, theta_hat[0, :])
        goal_term = 1e1 * V_goal_pt.mean()
        loss.append(("CLBF goal term", goal_term))

        # Only train these terms if we have a barrier requirement
        if self.barrier:
            #   2.) 0 < V <= safe_level in the safe region
            V_safe = V[safe_mask]
            safe_violation = F.relu(eps + V_safe - self.safe_level)
            safe_V_term = 1e2 * safe_violation.mean()
            loss.append(("CLBF safe region term", safe_V_term))
            if accuracy:
                safe_V_acc = (safe_violation <= eps).sum() / safe_violation.nelement()
                loss.append(("CLBF safe region accuracy", safe_V_acc))

            #   3.) V >= unsafe_level in the unsafe region
            V_unsafe = V[unsafe_mask]
            unsafe_violation = F.relu(eps + self.unsafe_level - V_unsafe)
            unsafe_V_term = 1e2 * unsafe_violation.mean()
            loss.append(("CLBF unsafe region term", unsafe_V_term))
            if accuracy:
                unsafe_V_acc = (
                    unsafe_violation <= eps
                ).sum() / unsafe_violation.nelement()
                loss.append(("CLBF unsafe region accuracy", unsafe_V_acc))

        return loss

    def descent_loss(
        self,
        x: torch.Tensor,
        theta_hat: torch.Tensor,
        theta: torch.Tensor,
        goal_mask: torch.Tensor,
        safe_mask: torch.Tensor,
        unsafe_mask: torch.Tensor,
        accuracy: bool = False,
        requires_grad: bool = False,
    ) -> List[Tuple[str, torch.Tensor]]:
        """
        Evaluate the loss on the CLBF due to the descent condition

        args:
            x: the points at which to evaluate the loss,
            theta: the parameter estimate points at which to evaluate the loss,
            goal_mask: the points in x marked as part of the goal
            safe_mask: the points in x marked safe
            unsafe_mask: the points in x marked unsafe
            accuracy: if True, return the accuracy (from 0 to 1) as well as the losses
            requires_grad: if True, use a differentiable QP solver
        returns:
            loss: a list of tuples containing ("category_name", loss_value).
        """
        # Compute loss to encourage satisfaction of the following conditions...
        loss = []
        bs = x.shape[0]
        n_controls = self.dynamics_model.n_controls
        n_params = self.dynamics_model.n_params

        # The CLBF decrease condition requires that V is decreasing everywhere where
        # V <= safe_level. We'll encourage this in three ways:
        #
        #   1) Minimize the relaxation needed to make the QP feasible.
        #   2) Compute the CLBF decrease at each point by linearizing
        #   3) Compute the CLBF decrease at each point by simulating

        # First figure out where this condition needs to hold
        eps = 0.1
        V = self.V(x, theta_hat)
        if self.barrier:
            condition_active = torch.sigmoid(10 * (self.safe_level + eps - V))
        else:
            condition_active = torch.tensor(1.0)

        # Get the control input and relaxation from solving the QP, and aggregate
        # the relaxation across scenarios
        u_qp, qp_relaxation = self.solve_CLF_QP(x, theta_hat, requires_grad=requires_grad)
        qp_relaxation = torch.mean(qp_relaxation, dim=-1)

        # Minimize the qp relaxation to encourage satisfying the decrease condition
        qp_relaxation_loss = (qp_relaxation * condition_active).mean()
        loss.append(("QP relaxation", qp_relaxation_loss))

        # Now compute the decrease using linearization
        eps = 1.0
        clbf_descent_term_lin = torch.tensor(0.0).type_as(x)
        clbf_descent_acc_lin = torch.tensor(0.0).type_as(x)
        # Get the current value of the CLBF and its Lie derivatives
        Lf_Va, LF_Va, LFGammadVa_Va, Lg_V, list_LGi_V, LGammadVG_V = self.V_lie_derivatives(x, theta_hat)
        for i, s in enumerate(self.scenarios):
            # Use the dynamics to compute the derivative of V
            sum_LG_V = torch.zeros((bs, self.n_scenarios, n_controls))
            for theta_dim in range(n_params):
                sum_LG_V = sum_LG_V + torch.bmm(theta[:, theta_dim].reshape((bs, 1, 1)), list_LGi_V[theta_dim])
            Vdot = Lf_Va[:, i, :].unsqueeze(1) + \
                   torch.bmm(LF_Va[:, i, :].unsqueeze(1), theta.reshape((theta.shape[0], theta.shape[1], 1))) + \
                   LFGammadVa_Va[:, i, :].unsqueeze(1) + \
                   torch.bmm(
                        Lg_V[:, i, :].unsqueeze(1) + sum_LG_V + LGammadVG_V,
                        u_qp.reshape(-1, self.dynamics_model.n_controls, 1),
                    )
            Vdot = Vdot.reshape(V.shape)
            violation = F.relu(eps + Vdot + self.clf_lambda * V)
            violation = violation * condition_active
            clbf_descent_term_lin = clbf_descent_term_lin + violation.mean()
            clbf_descent_acc_lin = clbf_descent_acc_lin + (violation <= eps).sum() / (
                violation.nelement() * self.n_scenarios
            )

        loss.append(("CLBF descent term (linearized)", clbf_descent_term_lin))
        if accuracy:
            loss.append(("CLBF descent accuracy (linearized)", clbf_descent_acc_lin))

        # Now compute the decrease using simulation
        eps = 1.0
        clbf_descent_term_sim = torch.tensor(0.0).type_as(x)
        clbf_descent_acc_sim = torch.tensor(0.0).type_as(x)
        for s in self.scenarios:
            xdot = self.dynamics_model.closed_loop_dynamics(x, u_qp, theta, params=s)
            x_next = x + self.dynamics_model.dt * xdot
            theta_hat_next = theta_hat # TODO: Define theta estimator!
            V_next = self.V(x_next, theta_hat_next)
            violation = F.relu(
                eps + (V_next - V) / self.controller_period + self.clf_lambda * V
            )
            violation = violation * condition_active

            clbf_descent_term_sim = clbf_descent_term_sim + violation.mean()
            clbf_descent_acc_sim = clbf_descent_acc_sim + (violation <= eps).sum() / (
                violation.nelement() * self.n_scenarios
            )
        loss.append(("CLBF descent term (simulated)", clbf_descent_term_sim))
        if accuracy:
            loss.append(("CLBF descent accuracy (simulated)", clbf_descent_acc_sim))

        return loss

    def initial_loss(self, x: torch.Tensor, theta_hat:torch.Tensor) -> List[Tuple[str, torch.Tensor]]:
        """
        Compute the loss during the initialization epochs, which trains the net to
        match the local linear lyapunov function
        inputs
            x: the states at which to evaluate the loss
            theta_hat: the parameter estimate points at which to evaluate the loss
        """
        # Constants
        bs = x.shape[0]
        V_Theta = pc.extreme(self.dynamics_model.Theta)

        loss = []

        # The initial losses should decrease exponentially to zero, based on the epoch
        epoch_count = max(self.current_epoch - self.num_init_epochs, 0)
        decrease_factor = 0.8 ** epoch_count

        #   1.) Compare the CLBF to the nominal solution
        # Get the learned CLBF
        V = self.V(x, theta_hat)
        V_corners = []
        for v_Theta_np in V_Theta:
            v_Theta = torch.Tensor(v_Theta_np.T)
            v_Theta = v_Theta.reshape((1, self.dynamics_model.n_dims))
            v_Theta = v_Theta.repeat((bs, 1))
            V_corners.append(self.V(x, v_Theta))

        # Get the nominal Lyapunov function
        P = self.dynamics_model.P.type_as(x)
        x0 = self.dynamics_model.goal_point.type_as(x)
        # Reshape to use pytorch's bilinear function
        P = P.reshape(1, self.dynamics_model.n_dims, self.dynamics_model.n_dims)
        V_nominal = 0.5 * F.bilinear(x - x0, x - x0, P).squeeze()

        if self.normalize_V_nominal:
            self.V_nominal_mean = V_nominal.mean()
            V_nominal /= self.V_nominal_mean

        # Compute the error between the two
        clbf_mse_loss = (V - V_nominal) ** 2
        clbf_mse_loss = decrease_factor * clbf_mse_loss.mean()
        for corner_idx in range(len(V_Theta)):
            V_corner = (V_corners[corner_idx] - V_nominal) ** 2
            clbf_mse_loss = clbf_mse_loss + decrease_factor * V_corner.mean()

        loss.append(("CLBF MSE", clbf_mse_loss))

        return loss

    def training_step(self, batch, batch_idx):
        """Conduct the training step for the given batch"""
        # Extract the input and masks from the batch
        x, theta, theta_hat, goal_mask, safe_mask, unsafe_mask = batch

        # Compute the losses
        component_losses = {}
        component_losses.update(self.initial_loss(x, theta_hat))
        component_losses.update(
            self.boundary_loss(x, theta_hat, theta, goal_mask, safe_mask, unsafe_mask)
        )
        component_losses.update(
            self.descent_loss(x, theta_hat, theta, goal_mask, safe_mask, unsafe_mask, requires_grad=True)
        )

        # Compute the overall loss by summing up the individual losses
        total_loss = torch.tensor(0.0).type_as(x)
        # For the objectives, we can just sum them
        for _, loss_value in component_losses.items():
            if not torch.isnan(loss_value):
                total_loss = total_loss + loss_value

        batch_dict = {"loss": total_loss, **component_losses}

        return batch_dict

    def training_epoch_end(self, outputs):
        """
        Description
            This function is called after every epoch is completed.
        """
        # Outputs contains a list for each optimizer, and we need to collect the losses
        # from all of them if there is a nested list
        if isinstance(outputs[0], list):
            outputs = itertools.chain(*outputs)

        # Gather up all of the losses for each component from all batches
        losses = {}
        for batch_output in outputs:
            for key in batch_output.keys():
                # if we've seen this key before, add this component loss to the list
                if key in losses:
                    losses[key].append(batch_output[key])
                else:
                    # otherwise, make a new list
                    losses[key] = [batch_output[key]]

        # Average all the losses
        avg_losses = {}
        for key in losses.keys():
            key_losses = torch.stack(losses[key])
            avg_losses[key] = torch.nansum(key_losses) / key_losses.shape[0]

        # Log the overall loss...
        self.log("Total loss / train", avg_losses["loss"], sync_dist=True)
        # And all component losses
        for loss_key in avg_losses.keys():
            # We already logged overall loss, so skip that here
            if loss_key == "loss":
                continue
            # Log the other losses
            self.log(loss_key + " / train", avg_losses[loss_key], sync_dist=True)

    def validation_step(self, batch, batch_idx):
        """Conduct the validation step for the given batch"""
        # Extract the input and masks from the batch
        x, theta, theta_hat, goal_mask, safe_mask, unsafe_mask = batch

        # Get the various losses
        component_losses = {}
        component_losses.update(
            self.boundary_loss(x, theta_hat, theta, goal_mask, safe_mask, unsafe_mask)
        )
        component_losses.update(self.descent_loss(x, theta_hat, theta, goal_mask, safe_mask, unsafe_mask))

        # Compute the overall loss by summing up the individual losses
        total_loss = torch.tensor(0.0).type_as(x)
        # For the objectives, we can just sum them
        for _, loss_value in component_losses.items():
            if not torch.isnan(loss_value):
                total_loss = total_loss + loss_value

        # Also compute the accuracy associated with each loss
        component_losses.update(
            self.boundary_loss(x, theta_hat, theta, goal_mask, safe_mask, unsafe_mask, accuracy=True)
        )
        component_losses.update(
            self.descent_loss(x, theta_hat, theta, goal_mask, safe_mask, unsafe_mask, accuracy=True)
        )

        batch_dict = {"val_loss": total_loss, **component_losses}

        return batch_dict

    def validation_epoch_end(self, outputs):
        """This function is called after every epoch is completed."""
        # Gather up all of the losses for each component from all batches
        losses = {}
        for batch_output in outputs:
            for key in batch_output.keys():
                # if we've seen this key before, add this component loss to the list
                if key in losses:
                    losses[key].append(batch_output[key])
                else:
                    # otherwise, make a new list
                    losses[key] = [batch_output[key]]

        # Average all the losses
        avg_losses = {}
        for key in losses.keys():
            key_losses = torch.stack(losses[key])
            avg_losses[key] = torch.nansum(key_losses) / key_losses.shape[0]

        # Log the overall loss...
        self.log("Total loss / val", avg_losses["val_loss"], sync_dist=True)
        # And all component losses
        for loss_key in avg_losses.keys():
            # We already logged overall loss, so skip that here
            if loss_key == "val_loss":
                continue
            # Log the other losses
            self.log(loss_key + " / val", avg_losses[loss_key], sync_dist=True)

        # **Now entering spicetacular automation zone**
        # We automatically run experiments every few epochs

        # Only plot every 5 epochs
        if self.current_epoch % 5 != 0:
            return

        self.experiment_suite.run_all_and_log_plots(
            self, self.logger, self.current_epoch
        )

    @pl.core.decorators.auto_move_data
    def simulator_fn(
        self,
        x_init: torch.Tensor,
        num_steps: int,
        relaxation_penalty: Optional[float] = None,
    ):
        # Choose parameters randomly
        random_scenario = {}
        for param_name in self.scenarios[0].keys():
            param_max = max([s[param_name] for s in self.scenarios])
            param_min = min([s[param_name] for s in self.scenarios])
            random_scenario[param_name] = random.uniform(param_min, param_max)

        return self.dynamics_model.simulate(
            x_init,
            num_steps,
            self.u,
            guard=self.dynamics_model.out_of_bounds_mask,
            controller_period=self.controller_period,
            params=random_scenario,
        )

    def on_validation_epoch_end(self):
        """This function is called at the end of every validation epoch"""
        # We want to generate new data at the end of every episode
        if self.current_epoch > 0 and self.current_epoch % self.epochs_per_episode == 0:
            if self.penalty_scheduling_rate > 0:
                relaxation_penalty = (
                    self.clf_relaxation_penalty
                    * self.current_epoch
                    / self.penalty_scheduling_rate
                )
            else:
                relaxation_penalty = self.clf_relaxation_penalty

            # Use the models simulation function with this controller
            def simulator_fn_wrapper(x_init: torch.Tensor, num_steps: int):
                return self.simulator_fn(
                    x_init,
                    num_steps,
                    relaxation_penalty=relaxation_penalty,
                )

            self.datamodule.add_data(simulator_fn_wrapper)

    def configure_optimizers(self):
        clbf_params = list(self.V_nn.parameters())

        clbf_opt = torch.optim.SGD(
            clbf_params,
            lr=self.primal_learning_rate,
            weight_decay=1e-6,
        )

        self.opt_idx_dict = {0: "clbf"}

        return [clbf_opt]
