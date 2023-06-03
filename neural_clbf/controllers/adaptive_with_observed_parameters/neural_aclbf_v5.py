import itertools
from typing import Tuple, List, Optional, Callable
from collections import OrderedDict
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from neural_clbf.systems.adaptive import ControlAffineParameterAffineSystem
from neural_clbf.systems.adaptive_w_scenarios import ControlAffineParameterAffineSystem2
from neural_clbf.systems.utils import ScenarioList, Scenario
from neural_clbf.controllers.adaptive_with_observed_parameters import (
    aCLFController5,
)
from neural_clbf.controllers.controller_utils import (
    normalize_with_angles, normalize_theta_with_angles, normalize_scenario_with_angles,
)
from neural_clbf.datamodules.episodic_datamodule import EpisodicDataModule
from neural_clbf.experiments import ExperimentSuite

from neural_clbf.controllers.adaptive.adaptive_control_utils import (
    center_and_radius_to_vertices, violation_weighting_function2,
)

import polytope as pc
import numpy as np


class NeuralaCLBFControllerV5(aCLFController5, pl.LightningModule):
    """
    A neural aCLBF controller. Differs from the CLFController in that it uses a
    neural network to learn the CLF, and it turns it from a CLF to a CLBF by making sure
    that a level set of the CLF separates the safe and unsafe regions.

    More specifically, the CLBF controller looks for a V such that

    V(goal(scenario),theta_hat,scenario) = 0
    V >= 0
    V(safe, theta_hat, scenario) < c
    V(unsafe, theta_hat, scenario) > c
    dV/dt <= -lambda V

    This proves forward invariance of the c-sublevel set of V, and since the safe set is
    a subset of this sublevel set, we prove that the unsafe region is not reachable from
    the safe region. We also prove convergence to a point.
    """

    def __init__(
        self,
        dynamics_model: ControlAffineParameterAffineSystem2,
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
        learn_shape_epochs: int = 0,
        learn_boundary_epochs: int = 0,
        barrier: bool = True,
        add_nominal: bool = False,
        normalize_V_nominal: bool = False,
        saved_Vnn: torch.nn.Sequential = None,
        Gamma_factor: float = None,
        include_oracle_loss: bool = True,
        include_estimation_error_loss: bool = False,
        include_radially_unbounded_loss1: bool = False,
        include_radially_unbounded_loss2: bool = False,
        Q_u: np.array = None,
        max_iters_cvxpylayer: int = 50000000,
        show_debug_messages: bool = False,
        diff_qp_layer_to_use: str = "cvxpylayer",
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
            learn_shape_epochs: the number of epochs during which the shape of the CLBF
                                is learned. After this, the descent and other losses are added.
            learn_boundary_epochs: the number of epochs during which the boundaries of the CLBF
                                is learned. This is learned after a general shape has been created
                                and before the gradient losses are added.
            barrier: if True, train the CLBF to act as a barrier functions. If false,
                     effectively trains only a CLF.
            add_nominal: if True, add the nominal V
            normalize_V_nominal: if True, normalize V_nominal so that its average is 1
        """
        super(NeuralaCLBFControllerV5, self).__init__(
            dynamics_model,
            experiment_suite,
            clf_lambda=clf_lambda,
            clf_relaxation_penalty=clf_relaxation_penalty,
            controller_period=controller_period,
            Gamma_factor=Gamma_factor,
            Q_u=Q_u,
            max_iters_cvxpylayer=max_iters_cvxpylayer,
            show_debug_messages=show_debug_messages,
            diff_qp_layer_to_use=diff_qp_layer_to_use,
        )

        self.save_hyperparameters()

        # Save the provided model
        # self.dynamics_model = dynamics_model
        # self.scenarios = scenarios
        # self.n_scenarios = len(scenarios)

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
        self.learn_shape_epochs = learn_shape_epochs
        self.learn_boundary_epochs = learn_boundary_epochs

        self.barrier = barrier
        self.add_nominal = add_nominal
        self.normalize_V_nominal = normalize_V_nominal
        self.V_nominal_mean = 1.0
        self.include_oracle_loss = include_oracle_loss
        self.include_estimation_error_loss = include_estimation_error_loss
        self.include_radially_unbounded_loss1 = include_radially_unbounded_loss1
        self.include_radially_unbounded_loss2 = include_radially_unbounded_loss2

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

        # Compute and save the center and range of the unknown parameter set
        self.theta_center, self.theta_range = self.calculate_center_and_range(self.dynamics_model.Theta)

        # Compute and save the center and range of the scenario (known paramters) set
        self.scen_center, self.scen_range = self.calculate_center_and_range(self.dynamics_model.scenario_set)

        # Define the CLBF network, which we denote V
        self.clbf_hidden_layers = clbf_hidden_layers
        self.clbf_hidden_size = clbf_hidden_size
        # We're going to build the network up layer by layer, starting with the input
        self.V_layers: OrderedDict[str, nn.Module] = OrderedDict()
        self.V_layers["input_linear"] = nn.Linear(
            self.n_dims_extended+self.n_params_extended+self.dynamics_model.n_scenario, self.clbf_hidden_size
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

        # Define training outputs for convenient passing of data during on_train_epoch_end()
        self.training_step_outputs = []
        self.validation_step_outputs = []

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

    def V_with_jacobian(self, x: torch.Tensor, theta_hat: torch.Tensor, scen: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the CLBF value and its Jacobian

        args:
            x: bs x self.dynamics_model.n_dims the points at which to evaluate the CLBF
            theta_hat: bs x self.dynamics_model.n_params the points at which to evaluate the CLBF
            scen: a tensor of batch_size x self.dynamics_model.n_scenario points in the scenario space which determine the scenario values (constant over time)
        returns:
            V: bs tensor of CLBF values
            JVx: bs x 1 x self.dynamics_model.n_dims Jacobian of each row of V wrt x
        """
        # Apply the offset and range to normalize about zero
        x_norm = normalize_with_angles(self.dynamics_model, x)
        theta_hat_norm = normalize_theta_with_angles(self.dynamics_model, theta_hat)
        scen_norm = normalize_scenario_with_angles(self.dynamics_model, scen, angle_dims=[])

        # Compute the CLBF layer-by-layer, computing the Jacobian alongside

        # We need to initialize the Jacobian to reflect the normalization that's already
        # been done to x
        bs = x_norm.shape[0]
        n_dims = self.dynamics_model.n_dims
        n_params = self.dynamics_model.n_params
        n_scenario = self.dynamics_model.n_scenario

        n_dims_extended = self.n_dims_extended
        n_params_extended = self.n_params_extended

        JV_x_th_s = torch.zeros(
            (bs, n_dims_extended+n_params_extended+n_scenario, self.dynamics_model.n_dims+self.dynamics_model.n_params+n_scenario),
            device=x.device,
        )
        # and for each non-angle dimension, we need to scale by the normalization
        for dim in range(self.dynamics_model.n_dims):
            JV_x_th_s[:, dim, dim] = 1.0 / self.x_range[dim].type_as(x)

        for dim in range(self.dynamics_model.n_params):
            JV_x_th_s[:, dim+self.n_dims_extended, dim+n_dims] = 1.0 / self.theta_range[dim].type_as(theta_hat)

        for dim in range(self.dynamics_model.n_scenario):
            JV_x_th_s[:, dim+n_dims_extended+n_params_extended, dim + n_dims + n_params] = 1.0 / self.scen_range[dim].type_as(scen)

        # And adjust the Jacobian for the angle dimensions
        for offset, sin_idx in enumerate(self.dynamics_model.angle_dims):
            cos_idx = self.dynamics_model.n_dims + offset
            JV_x_th_s[:, sin_idx, sin_idx] = x_norm[:, cos_idx]
            JV_x_th_s[:, cos_idx, sin_idx] = -x_norm[:, sin_idx]

        for offset, sin_idx in enumerate(self.dynamics_model.parameter_angle_dims):
            sin_idx = sin_idx + self.n_dims_extended
            cos_idx = self.n_dims_extended + self.dynamics_model.n_params + offset
            JV_x_th_s[:, sin_idx, sin_idx] = theta_hat_norm[:, cos_idx - (self.n_dims_extended )]
            JV_x_th_s[:, cos_idx, sin_idx] = -theta_hat_norm[:, sin_idx - (self.n_dims_extended)]

        # TODO: Maybe add support for scenario having angle dims?


        # Now step through each layer in V
        x_theta_scen_norm = torch.zeros(
            (bs, self.n_dims_extended+self.n_params_extended+n_scenario),
            device=x.device,
        ).type_as(x)
        # x_theta_norm = torch.cat([x_norm, theta_hat_norm], dim=1)

        x_theta_scen_norm[:, :self.n_dims_extended] = x_norm
        x_theta_scen_norm[:, self.n_dims_extended:self.n_dims_extended+self.n_params_extended] = theta_hat_norm
        x_theta_scen_norm[:, self.n_dims_extended+self.n_params_extended:] = scen_norm
        V = x_theta_scen_norm
        #print("JVxth shape before loops", JVxth.shape)
        for layer in self.V_nn:
            V = layer(V)

            if isinstance(layer, nn.Linear):
                JV_x_th_s = torch.matmul(layer.weight, JV_x_th_s)
            elif isinstance(layer, nn.Tanh):
                JV_x_th_s = torch.matmul(torch.diag_embed(1 - V ** 2), JV_x_th_s)
            elif isinstance(layer, nn.ReLU):
                JV_x_th_s = torch.matmul(torch.diag_embed(torch.sign(V)), JV_x_th_s)

            if self.show_debug_messages:
                smallJV_idcs = torch.norm(JV_x_th_s, dim=(1, 2)) <= 1e-6
                if (torch.sum(smallJV_idcs) > 0) & (isinstance(layer, nn.Tanh)):
                    print("smallJV_idcs: ", smallJV_idcs)
                    print("x_norm = ", x_norm)
                    print("x_theta_norm[smallJV_idcs, :]: ", x_theta_scen_norm[smallJV_idcs, :])
                    print("x_theta_norm: ", x_theta_scen_norm)
                    print("# of Small JV_idcs: ", torch.sum(smallJV_idcs))
                    print("Last Tanh weight: ", torch.diag_embed(1 - V**2))

                    print("V = ", V)
                    print("(V * V).sum(dim=1) = ", (V * V).sum(dim=1))
                    print("(V * V).sum(dim=1) = ", 0.5 * (V * V).sum(dim=1))



        # Compute the final activation
        JV_x_th_s = torch.bmm(V.unsqueeze(1), JV_x_th_s)
        V = 0.5 * (V * V).sum(dim=1)

        # largeV_idcs = V.flatten() >= 1e10
        # if torch.sum(largeV_idcs) > 0:
        #     print("V.dtype = ", V.dtype)
        #     print("largeVs: ", V[largeV_idcs])
        #     print("x_theta_norm.dtype = ", x_theta_norm.dtype)
        #     print("xtheta_norm[largeV_idcs, :]: ", x_theta_norm[largeV_idcs, :])

        if self.add_nominal:
            # Get the nominal Lyapunov function
            P = self.dynamics_model.P.type_as(x)
            theta0 = torch.Tensor(
                np.mean(pc.extreme(self.dynamics_model.Theta), axis=0)
            )
            x0 = self.dynamics_model.goal_point(theta0).type_as(x)

            xtheta0 = torch.zeros(x0.shape, device=x.device)
            xtheta0[:, :self.dynamics_model.n_dims] = x0
            xtheta0[:, self.dynamics_model.n_dims:] = theta0

            xtheta = torch.zeros(xtheta0.shape, device=x.device)
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
            P = P.reshape(1, self.dynamics_model.n_dims, self.dynamics_model.n_dims)
            V_nominal = 0.5 * F.bilinear(x - x0, x - x0, P)

            # Reshape again to calculate the gradient
            P = P.reshape(self.dynamics_model.n_dims, self.dynamics_model.n_dims)
            JV_x_th_s_nominal = F.linear(x - x0, P)
            JV_x_th_s_nominal = JV_x_th_s_nominal.reshape(x.shape[0], 1, self.dynamics_model.n_dims)

            if self.normalize_V_nominal:
                V_nominal /= self.V_nominal_mean
                JV_x_th_s_nominal /= self.V_nominal_mean

            V = V + V_nominal
            JV_x_th_s = JV_x_th_s + JVxth_nominal

        # At the very last second split the gradient.
        JVx = torch.zeros((bs, 1, self.dynamics_model.n_dims), device=x.device)
        JVx[:, 0, :] = JV_x_th_s[:, 0, :self.dynamics_model.n_dims]

        JVth = torch.zeros((bs, 1, self.dynamics_model.n_params), device=x.device)
        JVth[:, 0, :] = JV_x_th_s[:, 0, n_dims:n_dims+n_params]

        return V, JVx, JVth

    def forward(self, x_theta_scen_triple):
        """Determine the control input for a given state using a QP

        args:
            x_theta_scen_triple: tuple (x, theta, scen) of bs x self.dynamics_model.n_dims tensor of state,
                bs x self.dynamics_model.n_params tensor of unknown parameters and bs x self.dynamics_model.n_scenario
                tensor of scenario parameters
        returns:
            u: bs x self.dynamics_model.n_controls tensor of control inputs
        """
        # Create the state-theta_hat pair
        return self.u(x_theta_scen_triple)

    def boundary_loss(
        self,
        x: torch.Tensor,
        theta_hat: torch.Tensor,
        theta: torch.Tensor,
        scen: torch.Tensor,
        goal_mask: torch.Tensor,
        safe_mask: torch.Tensor,
        unsafe_mask: torch.Tensor,
        accuracy: bool = False,
    ) -> List[Tuple[str, torch.Tensor]]:
        """
        Evaluate the loss on the CLBF due to boundary conditions

        args:
            x: the points at which to evaluate the loss,
            theta_hat: the set of theta estimate points at which to evaluate the loss,
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

        V = self.V(x, theta_hat, scen)

        #   1.) CLBF should be minimized on the goal point
        goal_as_batch = self.dynamics_model.goal_point(theta, scen).type_as(x)
        V_goal_pt = self.V(goal_as_batch, theta, scen)
        goal_term = 1e2 * V_goal_pt.mean()
        loss.append(("CLBF goal term", goal_term))

        # Only train these terms if we have a barrier requirement
        if self.barrier:
            #   2.) 0 < V <= safe_level in the safe region
            V_safe = V[safe_mask]
            if len(V_safe) > 0:
                # Nonzero number of safe points found in data
                safe_violation = F.relu(eps + V_safe - self.safe_level)
                safe_V_term = 1e2 * safe_violation.mean()
                loss.append(("CLBF safe region term", safe_V_term))
                if accuracy:
                    safe_V_acc = (safe_violation <= eps).sum() / safe_violation.nelement()
                    loss.append(("CLBF safe region accuracy", safe_V_acc))
            else:
                # No safe points found in data
                loss.append(("CLBF safe region term", torch.tensor(0.0, device=x.device)))
                if accuracy:
                    loss.append(("CLBF safe region accuracy", torch.tensor(1.0, device=x.device)))

            #   3.) V >= unsafe_level in the unsafe region
            V_unsafe = V[unsafe_mask]
            if len(V_unsafe) > 0:
                # Nonzero number of unsafe points found in data
                unsafe_violation = F.relu(eps + self.unsafe_level - V_unsafe)
                unsafe_V_term = 1e2 * unsafe_violation.mean()
                loss.append(("CLBF unsafe region term", unsafe_V_term))
                if accuracy:
                    unsafe_V_acc = (
                        unsafe_violation <= eps
                    ).sum() / unsafe_violation.nelement()
                    loss.append(("CLBF unsafe region accuracy", unsafe_V_acc))
            else:
                # No unsafe points found in data
                loss.append(("CLBF unsafe region term", torch.tensor(0.0, device=x.device)))
                if accuracy:
                    loss.append(("CLBF unsafe region accuracy", torch.tensor(1.0, device=x.device)))

        return loss

    def descent_loss(
        self,
        x: torch.Tensor,
        theta_hat: torch.Tensor,
        theta: torch.Tensor,
        scen: torch.Tensor,
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
            theta: the parameter points at which to evaluate the loss,
            theta_hat: the parameter estimate points at which to evaluate the loss,
            scen: a tensor of batch_size x self.dynamics_model.n_scenario points in the scenario space which determine the scenario values (constant over time)
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
        # V_Theta = pc.extreme(self.dynamics_model.Theta)
        # n_V_Theta = V_Theta.shape[0]

        eps0 = 2.5e-2
        V_Theta = center_and_radius_to_vertices(
            theta_hat, eps0*torch.ones(theta_hat.shape, device=theta_hat.device),
        )
        n_V_Theta = len(V_Theta)

        # The aCLBF decrease condition requires that V is decreasing everywhere where
        # Va <= safe_level. We'll encourage this in three ways:
        #
        #   1) Minimize the relaxation needed to make the QP feasible.
        #   2) Compute the aCLBF decrease at each point by linearizing
        #   3) Compute the aCLBF decrease at each point by simulating

        # First figure out where this condition needs to hold
        eps = 0.1
        Va = self.V(x, theta_hat, scen)
        if self.barrier:
            condition_active = torch.sigmoid(10 * (self.safe_level + eps - Va))
        else:
            condition_active = torch.tensor(1.0)

        # Create a value of V for each corner in theta
        V_Theta_list = []
        for v_Theta_index in range(n_V_Theta):
            # theta0 = torch.Tensor(v_Theta)
            # theta_corner = theta0.repeat((bs, 1))
            theta_corner = V_Theta[v_Theta_index]

            V_Theta_list.append(
                self.V(x, theta_corner, scen)
            )

        # Get the control input and relaxation from solving the QP, and aggregate
        # the relaxation across scenarios
        u_qp, qp_relaxation = self.solve_CLF_QP(
            x, theta_hat, scen, requires_grad=requires_grad,
        )
        qp_relaxation = torch.mean(qp_relaxation, dim=-1)

        # Minimize the qp relaxation to encourage satisfying the decrease condition
        qp_relaxation_loss = (qp_relaxation * condition_active).mean()
        loss.append(("QP relaxation", qp_relaxation_loss))

        # Now compute the decrease using linearization
        eps = 1.0
        clbf_descent_term_lin = torch.tensor(0.0).type_as(x)
        clbf_descent_acc_lin = torch.tensor(0.0).type_as(x)
        # Get the current value of the CLBF and its Lie derivatives
        # Lf_Va, LF_Va, LFGammadVa_Va, Lg_V, list_LGi_V, LGammadVG_V = self.V_lie_derivatives(x, theta_hat)
        for v_Theta_index in range(n_V_Theta):
            # theta0 = torch.Tensor(v_Theta)
            # theta_corner = theta0.repeat((bs, 1))
            theta_corner = V_Theta[v_Theta_index]

            # Use the dynamics to compute the derivative of V at each corner of V_Theta
            Vdot = self.Vdot_for_scenario(x, theta_corner, scen, u_qp)
            # print("Vdot = ", Vdot)

            # sum_LG_V = torch.zeros((bs, self.n_scenarios, n_controls))
            # for theta_dim in range(n_params):
            #     sum_LG_V = sum_LG_V + torch.bmm(theta[:, theta_dim].reshape((bs, 1, 1)), list_LGi_V[theta_dim])
            # Vdot = Lf_Va[:, i, :].unsqueeze(1) + \
            #        torch.bmm(LF_Va[:, i, :].unsqueeze(1), theta.reshape((theta.shape[0], theta.shape[1], 1))) + \
            #        LFGammadVa_Va[:, i, :].unsqueeze(1) + \
            #        torch.bmm(
            #             Lg_V[:, i, :].unsqueeze(1) + sum_LG_V + LGammadVG_V,
            #             u_qp.reshape(-1, self.dynamics_model.n_controls, 1),
            #         )

            Vdot = Vdot.reshape(Va.shape)
            violation = F.relu(eps + Vdot + self.clf_lambda * Va)
            violation = violation * condition_active
            clbf_descent_term_lin = clbf_descent_term_lin + (1./n_V_Theta) * violation.mean()
            clbf_descent_acc_lin = clbf_descent_acc_lin + (violation <= eps).sum() / (
                violation.nelement()
            )

        loss.append(("CLBF descent term (linearized)", clbf_descent_term_lin))
        if accuracy:
            loss.append(("CLBF descent accuracy (linearized)", clbf_descent_acc_lin))

        # Now compute the decrease using simulation
        eps = 1.0
        clbf_descent_term_sim = torch.tensor(0.0).type_as(x)
        clbf_descent_acc_sim = torch.tensor(0.0).type_as(x)

        aclbf_estimation_error_term_sim = torch.tensor(0.0).type_as(x)
        aclbf_estimation_error_acc_sim = torch.tensor(0.0).type_as(x)

        xdot = self.dynamics_model.closed_loop_dynamics(x, u_qp, theta_hat, scen)
        x_next = x + self.dynamics_model.dt * xdot
        theta_hat_next = theta_hat + \
                         self.dynamics_model.dt * self.closed_loop_estimator_dynamics(x, theta_hat, u_qp, scen)

        V_next = self.V(x_next, theta_hat_next, scen)
        violation = F.relu(
            eps + (V_next - Va) / self.controller_period + self.clf_lambda * Va
        )
        violation = violation * condition_active

        clbf_descent_term_sim = clbf_descent_term_sim + \
                                (violation_weighting_function2(
                                    self.dynamics_model, x, theta_hat, scen,
                                ) * violation).mean()
        clbf_descent_acc_sim = clbf_descent_acc_sim + (violation <= eps).sum() / (
            violation.nelement()
        )

        # Compute the estimation error
        if self.include_estimation_error_loss:
            theta_err = theta - theta_hat
            theta_err_norm = torch.norm(theta_err, dim=1)
            theta_err_next = theta - theta_hat_next
            theta_err_next_norm = torch.norm(theta_err_next, dim=1)
            violation_estim = F.relu(eps + (theta_err_next_norm - theta_err_norm))
            violation_estim = violation_estim * condition_active

            aclbf_estimation_error_term_sim = aclbf_estimation_error_term_sim + violation_estim.mean()
            aclbf_estimation_error_acc_sim = aclbf_estimation_error_acc_sim + (violation_estim <= eps).sum() / (
                violation_estim.nelement()
            )

        # Add all of these losses to the loss struct
        loss.append(("CLBF descent term (simulated)", clbf_descent_term_sim))
        if accuracy:
            loss.append(("CLBF descent accuracy (simulated)", clbf_descent_acc_sim))

        loss.append(("aCLBF estimation error term (simulated)", aclbf_estimation_error_term_sim))
        if accuracy:
            loss.append(("aCLBF estimation error accuracy (simulated)", aclbf_estimation_error_acc_sim))

        # Oracle Loss
        oracle_weight = float(len(V_Theta))  # Number of vertices in the polytope

        # Compute the shadow descent loss
        eps = 1.0
        oracle_aclf_descent_term_sim = torch.tensor(0.0).type_as(x)
        oracle_aclf_descent_acc_sim = torch.tensor(0.0).type_as(x)
        if self.include_oracle_loss:

            xdot = self.dynamics_model.closed_loop_dynamics(x, u_qp, theta_hat, scen)
            x_next = x + self.dynamics_model.dt * xdot
            theta_hat_next = theta_hat + \
                             self.dynamics_model.dt * self.closed_loop_estimator_dynamics(x, theta_hat, u_qp, scen)

            V_oracle = self.V_oracle(x, theta_hat, theta, scen)
            V_oracle_next = self.V_oracle(x_next, theta_hat_next, theta, scen)

            violation = F.relu(
                eps + (V_oracle_next - V_oracle) / self.controller_period + self.clf_lambda * V_oracle
            )
            violation = violation * condition_active

            oracle_aclf_descent_term_sim = oracle_aclf_descent_term_sim + oracle_weight * violation.mean()
            oracle_aclf_descent_acc_sim = oracle_aclf_descent_acc_sim + (violation <= eps).sum() / (
                    violation.nelement()
            )

        loss.append(("Oracle CLBF descent term (simulated)", oracle_aclf_descent_term_sim))
        if accuracy:
            loss.append(("Oracle CLBF descent accuracy (simulated)", oracle_aclf_descent_acc_sim))

        return loss

    def radially_unbounded_loss1(
        self,
        x: torch.Tensor,
        theta_hat: torch.Tensor,
        theta: torch.Tensor,
        scen: torch.Tensor,
        goal_mask: torch.Tensor,
        safe_mask: torch.Tensor,
        unsafe_mask: torch.Tensor,
        N_levels: int = 10,
    ):
        """
        list_of_loss_tuples = aclbf.radially_unbounded_loss(x, theta_hat)
        Description:
            This function computes a loss value that encourages the aclbf to be radially unbounded.
        Args:
            scen: a tensor of batch_size x self.dynamics_model.n_scenario points in the scenario space which determine the scenario values (constant over time)
        """
        # Constants
        bs = x.shape[0]
        Theta = self.dynamics_model.Theta
        V_Theta = pc.extreme(Theta)

        # Compute comparison points
        goal_tolerance = 0.1
        level_height = goal_tolerance / 2.0 # TODO: Define a goal_tolerance() method for all systems
        #level_height = self.dynamics_model.goal_tolerance / 2.0

        x_ub, x_lb = self.dynamics_model.state_limits
        x_width = x_ub - x_lb
        x_width_max = torch.max(x_width)

        theta_min = torch.min(torch.tensor(V_Theta), dim=0)[0]
        theta_max = torch.max(torch.tensor(V_Theta), dim=0)[0]
        theta_width = theta_max - theta_min
        theta_width_max = torch.max(theta_width)

        # Compute aCLF Value on whole batch
        V = self.V(x, theta_hat, scen)

        # Compute the loss
        loss = []

        radially_unbounded_loss = torch.tensor([0.0]).type_as(x)
        max_dist = x_width_max + theta_width_max
        goal_as_batch = self.dynamics_model.goal_point(theta_hat, scen).type_as(x)
        goal_theta_hat = torch.cat((goal_as_batch, theta_hat), dim=1)
        x_theta_hat = torch.cat((x, theta_hat), dim=1)

        #violations = torch.zeros((bs,)).type_as(x)
        last_level_count = -1
        curr_level_count = 0
        for curr_level in torch.arange(level_height, max_dist, level_height):
            # Compute mask of all points that are within curr_level of the goal points
            dist_to_goal = torch.norm(x_theta_hat - goal_theta_hat, dim=1)
            level_mask = dist_to_goal <= curr_level
            level_mask = level_mask.logical_and(safe_mask)

            if torch.all(level_mask == False):
                continue  # If there are no points in this level, then skip the rest of the computations.

            curr_level_count = torch.sum(level_mask)

            if last_level_count == curr_level_count:
                continue  # If there are no new points in this level, then skip the rest of the computations.

            V_level = V[level_mask]

            max_V_level = torch.max(V_level)

            # Compute mask of all points outside of this level
            outside_level_mask = dist_to_goal > curr_level
            outside_level_mask = outside_level_mask.logical_and(safe_mask)

            if torch.all(outside_level_mask == False):
                break  # If there are no more points outside of this level, then we can stop computing the loss.

            #outside_level_and_smaller_than_V_top = V[outside_level_mask] < max_V_level

            violations = F.relu(max_V_level - V[outside_level_mask])


            # Update level counts
            last_level_count = curr_level_count

            radially_unbounded_loss = radially_unbounded_loss + violations.mean()

        # Create loss
        loss.append(("Radially Unbounded Loss 1", radially_unbounded_loss))

        return loss

    def radially_unbounded_loss2(
        self,
        x: torch.Tensor,
        theta_hat: torch.Tensor,
        theta: torch.Tensor,
        scen: torch.Tensor,
        goal_mask: torch.Tensor,
        safe_mask: torch.Tensor,
        unsafe_mask: torch.Tensor,
        N_levels: int = 10,
    ):
        """
        list_of_loss_tuples = aclbf.radially_unbounded_loss(x, theta_hat)
        Description:
            This function computes a loss value that encourages the aclbf to be radially unbounded.
        Args:
            scen: a tensor of batch_size x self.dynamics_model.n_scenario points in the scenario space which determine the scenario values (constant over time)
        """
        # Constants
        bs = x.shape[0]
        Theta = self.dynamics_model.Theta
        V_Theta = pc.extreme(Theta)

        # Compute comparison points
        goal_tolerance = 0.1
        level_height = goal_tolerance / 2.0 # TODO: Define a goal_tolerance() method for all systems
        #level_height = self.dynamics_model.goal_tolerance / 2.0

        x_ub, x_lb = self.dynamics_model.state_limits
        x_width = x_ub - x_lb
        x_width_max = torch.max(x_width)

        theta_min = torch.min(torch.tensor(V_Theta), dim=0)[0]
        theta_max = torch.max(torch.tensor(V_Theta), dim=0)[0]
        theta_width = theta_max - theta_min
        theta_width_max = torch.max(theta_width)

        # Compute aCLF Value on whole batch
        V = self.V(x, theta_hat, scen)

        # Compute the loss
        loss = []

        radially_unbounded_loss = torch.tensor([0.0]).type_as(x)
        max_dist = x_width_max + theta_width_max
        goal_as_batch = self.dynamics_model.goal_point(theta_hat, scen).type_as(x)
        goal_theta_hat = torch.cat((goal_as_batch, theta_hat), dim=1)
        x_theta_hat = torch.cat((x, theta_hat), dim=1)

        violation = torch.zeros((bs,)).type_as(x)
        violation[safe_mask] = 0.5*torch.norm(x_theta_hat[safe_mask] - goal_theta_hat[safe_mask], dim=1) - V[safe_mask]

        radially_unbounded_loss = radially_unbounded_loss + F.relu(violation).mean()

        # Create loss
        loss.append(("Radially Unbounded Loss 2", radially_unbounded_loss))

        return loss

    def initial_loss(self, x: torch.Tensor, theta_hat: torch.Tensor, scen: torch.Tensor) -> List[Tuple[str, torch.Tensor]]:
        """
        Compute the loss during the initialization epochs, which trains the net to
        match the local linear lyapunov function
        inputs
            x: the states at which to evaluate the loss
            theta_hat: the parameter estimate points at which to evaluate the loss
            scen: a tensor of batch_size x self.dynamics_model.n_scenario points in the scenario space which determine the scenario values (constant over time)
        """
        # Constants
        bs = x.shape[0]
        V_Theta = pc.extreme(self.dynamics_model.Theta)
        n_params = self.dynamics_model.n_params

        loss = []

        # The initial losses should decrease exponentially to zero, based on the epoch
        epoch_count = max(self.current_epoch - self.num_init_epochs - self.learn_shape_epochs, 0)
        decrease_factor = 0.8 ** epoch_count

        #   1.) Compare the CLBF to the nominal solution
        # Get the learned CLBF
        V = self.V(x, theta_hat, scen)
        V_corners = []
        for v_Theta_np in V_Theta:
            v_Theta = torch.tensor(v_Theta_np.T, device=x.device, dtype=torch.get_default_dtype())
            v_Theta = v_Theta.reshape((1, n_params))
            v_Theta = v_Theta.repeat((bs, 1))
            V_corners.append(self.V(x, v_Theta, scen))

        # Get the nominal Lyapunov function
        P = self.dynamics_model.P.type_as(x)
        x0 = self.dynamics_model.goal_point(theta_hat, scen).type_as(x)
        # Reshape to use pytorch's bilinear function
        P = P.reshape(1, self.dynamics_model.n_dims, self.dynamics_model.n_dims)
        P_theta = torch.eye(self.dynamics_model.n_dims).type_as(x).unsqueeze(0)
        V_nominal = 0.5 * F.bilinear(x - x0, x - x0, P)
        # print("V_nominal", V_nominal)

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
        x, theta, theta_hat, scen, goal_mask, safe_mask, unsafe_mask = batch

        # Compute the losses
        component_losses = {}
        component_losses.update(self.initial_loss(x, theta_hat, scen))
        if self.current_epoch >= self.learn_shape_epochs:
            component_losses.update(
                self.boundary_loss(x, theta_hat, theta, scen, goal_mask, safe_mask, unsafe_mask)
            )
        if self.current_epoch >= self.learn_shape_epochs + self.learn_boundary_epochs:
            component_losses.update(
                self.descent_loss(x, theta_hat, theta, scen, goal_mask, safe_mask, unsafe_mask, requires_grad=True)
            )
            if self.include_radially_unbounded_loss1:
                component_losses.update(
                    self.radially_unbounded_loss1(x, theta_hat, theta, scen, goal_mask, safe_mask, unsafe_mask)
                )
            if self.include_radially_unbounded_loss2:
                component_losses.update(
                    self.radially_unbounded_loss2(x, theta_hat, theta, scen, goal_mask, safe_mask, unsafe_mask)
                )

        # Compute the overall loss by summing up the individual losses
        total_loss = torch.tensor(0.0).type_as(x)
        # For the objectives, we can just sum them
        for _, loss_value in component_losses.items():
            if not torch.isnan(loss_value):
                total_loss = total_loss + loss_value.to(x.device)

        batch_dict = {"loss": total_loss, **component_losses}

        self.training_step_outputs.append(batch_dict)

        return batch_dict

    def on_train_epoch_end(self):
        """
        Description
            This function is called after every epoch is completed.
        """

        # # Outputs contains a list for each optimizer, and we need to collect the losses
        # # from all of them if there is a nested list
        # if isinstance(outputs[0], list):
        #     outputs = itertools.chain(*outputs)

        # batch_dict = self.training_step_outputs[0]
        # print(batch_dict)

        # Gather up all of the losses for each component from all batches
        losses = {}
        for batch_output in self.training_step_outputs:
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

        # Free memory
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        """Conduct the validation step for the given batch"""
        # Extract the input and masks from the batch
        x, theta, theta_hat, scen, goal_mask, safe_mask, unsafe_mask = batch

        # Get the various losses
        component_losses = {}
        component_losses.update(self.initial_loss(x, theta_hat, scen))
        component_losses.update(
            self.boundary_loss(x, theta_hat, theta, scen, goal_mask, safe_mask, unsafe_mask)
        )
        component_losses.update(self.descent_loss(x, theta_hat, theta, scen, goal_mask, safe_mask, unsafe_mask))
        if self.include_radially_unbounded_loss1:
            component_losses.update(
                self.radially_unbounded_loss1(x, theta_hat, theta, scen, goal_mask, safe_mask, unsafe_mask)
            )
        if self.include_radially_unbounded_loss2:
            component_losses.update(
                self.radially_unbounded_loss2(x, theta_hat, theta, scen, goal_mask, safe_mask, unsafe_mask)
            )

        # Compute the overall loss by summing up the individual losses
        total_loss = torch.tensor(0.0, device=x.device).type_as(x)
        # For the objectives, we can just sum them
        for _, loss_value in component_losses.items():
            if not torch.isnan(loss_value):
                total_loss = total_loss + loss_value.to(x.device)

        # Also compute the accuracy associated with each loss
        component_losses.update(
            self.boundary_loss(x, theta_hat, theta, scen, goal_mask, safe_mask, unsafe_mask, accuracy=True)
        )
        component_losses.update(
            self.descent_loss(x, theta_hat, theta, scen, goal_mask, safe_mask, unsafe_mask, accuracy=True)
        )

        batch_dict = {"val_loss": total_loss, **component_losses}

        self.validation_step_outputs.append(batch_dict)

        return batch_dict

    def on_validation_epoch_end(self):
        """
        on_validation_epoch_end
        Description:
            This function is called at the end of every validation epoch.
        """

        # Gather up all of the losses for each component from all batches
        losses = {}
        for batch_output in self.validation_step_outputs:
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
            # print("key: ", key)
            # print("losses[key]: ", losses[key])

            key_losses = torch.stack(losses[key])
            avg_losses[key] = torch.nansum(key_losses) / key_losses.shape[0]

        # Log the overall loss...
        self.log("Total loss / val", avg_losses["val_loss"], sync_dist=True)
        # self.logger.log_metrics(
        #     {"Total loss / val", avg_losses["val_loss"]},
        # )
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

        # Free memory
        self.validation_step_outputs.clear()

        # ========================================================
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
            def simulator_fn_wrapper(x_init: torch.Tensor, theta_init: torch.Tensor, scen_init: torch.Tensor, num_steps: int):
                return self.simulator_fn(
                    x_init,
                    theta_init,
                    scen_init,
                    num_steps,
                    relaxation_penalty=relaxation_penalty,
                )

            self.datamodule.add_data(simulator_fn_wrapper)

        return

    #@pl.core.decorators.auto_move_data
    def simulator_fn(
        self,
        x_init: torch.Tensor,
        theta_init: torch.Tensor,
        scen: torch.Tensor,
        num_steps: int,
        relaxation_penalty: Optional[float] = None,
    ):
        """
        simulator_fn
        Description:
            This function is used to simulate the system using the learned controller.
        args:
            scen: a tensor of batch_size x self.dynamics_model.n_scenario points in the scenario space which determine the scenario values (constant over time)
        """
        # Choose parameters randomly
        return self.simulate(
            x_init,
            theta_init,
            scen,
            num_steps,
            self.u,
            guard=self.dynamics_model.out_of_bounds_mask,
            controller_period=self.controller_period,
        )

    def simulate(
        self,
        x_init: torch.Tensor,
        theta: torch.Tensor,
        scen: torch.Tensor,
        num_steps: int,
        controller: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        controller_period: Optional[float] = None,
        guard: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Simulate the targeted dynamical system for the specified number of steps using the given controller

        args:
            x_init - bs x n_dims tensor of initial conditions
            theta - bs x n_params tensor of parameters
            scen - a tensor of batch_size x self.dynamics_model.n_scenario points in the scenario space which determine the scenario values (constant over time)
            num_steps - a positive integer
            controller - a mapping from state to control action
            controller_period - the period determining how often the controller is run
                                (in seconds). If none, defaults to self.dt
            guard - a function that takes a bs x n_dims tensor and returns a length bs
                    mask that's True for any trajectories that should be reset to x_init
            params - a dictionary giving the parameter values for the system. If None,
                     default to the nominal parameters used at initialization
        returns
            x_sim - bs x num_steps x self.n_dims tensor of simulated trajectories. If an error
                    occurs on any trajectory, the simulation of all trajectories will stop and
                    the second dimension will be less than num_steps
            th_sim -    bs x num_steps x self.n_params tensor of randomly chosen parameters
                        for each trajectory. Parameters should not change.
            th_h_sim -  bs x num_steps x self.n_params tensor of estimated parameters.
                        Estimate may change.
        usage
            x_sim, th_sim, th_h_sim = simulate(x, theta, N_sim, silly_control, 0.01)
        """
        # Create a tensor to hold the simulation results
        batch_size = x_init.shape[0]

        dynamical_model = self.dynamics_model
        n_dims = dynamical_model.n_dims
        n_params = dynamical_model.n_params
        n_controls = dynamical_model.n_controls

        # P = self.P
        # P = P.reshape(
        #     self.n_dims + self.n_params,
        #     self.n_dims + self.n_params
        # )

        # Set up Simulator Variables
        x_sim = torch.zeros(batch_size, num_steps, n_dims).type_as(x_init)
        x_sim[:, 0, :] = x_init

        th_sim = torch.zeros(batch_size, num_steps, n_params).type_as(theta)
        th_sim[:, 0, :] = theta

        th_h_sim = torch.zeros(batch_size, num_steps, n_params).type_as(theta)
        th_h_sim[:, 0, :] = self.dynamics_model.sample_Theta_space(batch_size)

        scen_sim = torch.zeros(
            batch_size, num_steps, self.dynamics_model.n_scenario,
            device=self.device,
        ).type_as(scen)
        scen_sim[:, 0, :] = scen

        u = torch.zeros(x_init.shape[0], n_controls).type_as(x_init)

        # Compute controller update frequency
        if controller_period is None:
            controller_period = self.controller_period
        controller_update_freq = int(controller_period / self.dynamics_model.dt)

        # Run the simulation until it's over or an error occurs
        t_sim_final = 0
        for tstep in range(1, num_steps):
            try:
                # Get the current state, theta and theta_hat
                x_current = x_sim[:, tstep - 1, :]
                theta_current = th_sim[:, tstep - 1, :]
                theta_hat_current = th_h_sim[:, tstep - 1, :]
                scen_current = scen_sim[:, tstep - 1, :]

                # Get the control input at the current state if it's time
                if tstep == 1 or tstep % controller_update_freq == 0:
                    u = controller(x_current, theta_hat_current, scen)

                # Simulate forward using the dynamics
                xdot = dynamical_model.closed_loop_dynamics(x_current, u, theta, scen)
                x_sim[:, tstep, :] = x_current + self.dynamics_model.dt * xdot
                th_sim[:, tstep, :] = theta_current
                scen_sim[:, tstep, :] = scen_current

                # Compute theta hat evolution
                th_h_dot = self.closed_loop_estimator_dynamics(
                    x_current, theta_hat_current, u,
                    scen,
                )
                th_h_sim[:, tstep, :] = theta_hat_current + self.dynamics_model.dt * th_h_dot

                # If the guard is activated for any trajectory, reset that trajectory
                # to a random state
                if guard is not None:
                    guard_activations = guard(x_sim[:, tstep, :])
                    n_to_resample = int(guard_activations.sum().item())
                    x_new = dynamical_model.sample_state_space(n_to_resample).type_as(x_sim)
                    x_sim[guard_activations, tstep, :] = x_new

                # Update the final simulation time if the step was successful
                t_sim_final = tstep
            except ValueError:
                break
        # TODO: Check this function for unused/misused variable names.

        return x_sim[:, : t_sim_final + 1, :], th_sim[:, : t_sim_final + 1, :], \
            th_h_sim[:, : t_sim_final + 1, :], scen_sim[:, : t_sim_final + 1, :]

    def configure_optimizers(self):
        clbf_params = list(self.V_nn.parameters())

        clbf_opt = torch.optim.SGD(
            clbf_params,
            lr=self.primal_learning_rate,
            weight_decay=1e-6,
        )

        self.opt_idx_dict = {0: "clbf"}

        return [clbf_opt]

    def calculate_center_and_range(self, P:pc.polytope):
        """
        calculate_center_and_range
        Description

        Usage
            mean1, range1 = self.calculate_center_and_range(Theta)

        Output
            mean1:
            range1
        """

        # Constants
        V_P = pc.extreme(P)

        # Algorithm
        lower_bounds, upper_bounds = [], []
        for dim_index in range(P.dim):
            # Compute min or max
            min_val_i = torch.min(
                torch.tensor([v[dim_index] for v in V_P])
            )
            max_val_i = torch.max(
                torch.tensor([v[dim_index] for v in V_P])
            )

            lower_bounds.append(min_val_i)
            upper_bounds.append(max_val_i)

        mean1 = (torch.tensor(lower_bounds) + torch.tensor(upper_bounds))/2.0
        range1 = (torch.tensor(upper_bounds) - torch.tensor(lower_bounds))/2.0
        return mean1, range1

    # def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
    #     """
    #     Description:
    #         Configure gradient clipping for the optimizer
    #
    #     """
    #     if self.current_epoch > 5:
    #         gradient_clip_val = gradient_clip_val * 2
    #
    #     # Lightning will handle the gradient clipping
    #     self.clip_gradients(optimizer, gradient_clip_val=gradient_clip_val,
    #                         gradient_clip_algorithm=gradient_clip_algorithm)
