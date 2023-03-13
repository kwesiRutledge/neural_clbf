from typing import Tuple, Optional, Union, List, Callable

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import torch
import torch.nn.functional as F

import polytope as pc

from neural_clbf.systems.adaptive import ControlAffineParameterAffineSystem
from neural_clbf.systems.utils import Scenario, ScenarioList
from neural_clbf.controllers.controller import Controller
from neural_clbf.experiments import ExperimentSuite


class aCLFController(Controller):
    """
    A generic adaptive CLF-based controller, using the quadratic Lyapunov function found for
    the linearized system.

    This controller and all subclasses assumes continuous-time dynamics.
    """

    def __init__(
        self,
        dynamics_model: ControlAffineParameterAffineSystem,
        scenarios: ScenarioList,
        experiment_suite: ExperimentSuite,
        clf_lambda: float = 1.0,
        clf_relaxation_penalty: float = 50.0,
        controller_period: float = 0.01,
        Gamma_factor: float = None,
    ):
        """Initialize the controller.

        args:
            dynamics_model: the control-affine dynamics of the underlying system
            scenarios: a list of parameter scenarios to train on
            experiment_suite: defines the experiments to run during training
            clf_lambda: convergence rate for the CLF
            clf_relaxation_penalty: the penalty for relaxing CLF conditions.
            controller_period: the timestep to use in simulating forward Vdot
        """
        super(aCLFController, self).__init__(
            dynamics_model=dynamics_model,
            experiment_suite=experiment_suite,
            controller_period=controller_period,
        )

        # Save the provided model
        # self.dynamics_model = dynamics_model
        self.scenarios = scenarios
        self.n_scenarios = len(scenarios)

        # Save the experiments suits
        self.experiment_suite = experiment_suite

        # Save the other parameters
        self.clf_lambda = clf_lambda
        self.safe_level: Union[torch.Tensor, float]
        self.unsafe_level: Union[torch.Tensor, float]
        self.clf_relaxation_penalty = clf_relaxation_penalty
        self.Gamma_factor = Gamma_factor
        if self.Gamma_factor is None:
            self.Gamma_factor = 1.0
        self.Gamma = torch.eye(self.dynamics_model.n_params) * self.Gamma_factor


        # Since we want to be able to solve the CLF-QP differentiably, we need to set
        # up the CVXPyLayers optimization. First, we define variables for each control
        # input and the relaxation in each scenario
        u = cp.Variable(self.dynamics_model.n_controls)
        clf_relaxations = []
        for scenario in self.scenarios:
            clf_relaxations.append(cp.Variable(1, nonneg=True))

        # Next, we define the parameters that will be supplied at solve-time: the value
        # of the Lyapunov function, its Lie derivatives, the relaxation penalty, and
        # the reference control input
        Va_param = cp.Parameter(1, nonneg=True)
        n_params = self.dynamics_model.n_params

        Lf_Va_params = []
        Lg_Va_params = []
        LF_Va_params = []
        LFGammadV_Va_params = []
        LG_Va_params = []
        LGammadVaG_params  = []  # LGammaVaG[scenario_idx] = (dVa/dx) * (\sum_i Gamma[i,:] * (dVa/dtheta).T * G_i)

        Theta_vertices = pc.extreme(self.dynamics_model.Theta)
        n_Theta_v = Theta_vertices.shape[0]
        for scenario in self.scenarios:
            Lf_Va_param_cluster = []
            Lg_Va_param_cluster = []
            LF_Va_param_cluster = []
            LGammadVaG_param_cluster = []
            LFGammadV_Va_param_cluster = []
            LG_Va_cluster = []
            for theta_idx in range(n_Theta_v):

                Lf_Va_param_cluster.append(cp.Parameter(1))
                Lg_Va_param_cluster.append(cp.Parameter(self.dynamics_model.n_controls))
                LF_Va_param_cluster.append(cp.Parameter(self.dynamics_model.n_params))
                LGammadVaG_param_cluster.append(cp.Parameter(self.dynamics_model.n_controls))

                # LFGammadV_Va_clusters = []
                # for v_Theta in Theta_vertices:
                #     LFGammadV_Va_clusters.append(cp.Parameter(1))
                # LFGammadV_Va_params.append(LFGammadV_Va_clusters)
                LFGammadV_Va_param_cluster.append(cp.Parameter(1))

                LG_Va_clump = []
                for theta_dim in range(self.dynamics_model.n_params):
                    LG_Va_clump.append(cp.Parameter(self.dynamics_model.n_controls))
                LG_Va_cluster.append(LG_Va_clump)

            # Add all clusters to main set of params
            Lf_Va_params.append(Lf_Va_param_cluster)
            Lg_Va_params.append(Lg_Va_param_cluster)
            LF_Va_params.append(LF_Va_param_cluster)
            LGammadVaG_params.append(LGammadVaG_param_cluster)
            LFGammadV_Va_params.append(LFGammadV_Va_param_cluster)
            LG_Va_params.append(LG_Va_cluster)

        clf_relaxation_penalty_param = cp.Parameter(1, nonneg=True)
        u_ref_param = cp.Parameter(self.dynamics_model.n_controls)

        # These allow us to define the constraints

        constraints = []
        for i in range(self.n_scenarios):
            for v_Theta_index in range(n_Theta_v):
                v_Theta = Theta_vertices[v_Theta_index]

                # Create the sum G_i term
                sum_LG_i_Va = v_Theta[0] * LG_Va_params[i][v_Theta_index][0]
                for theta_index in range(1, len(v_Theta)):
                    sum_LG_i_Va = sum_LG_i_Va + v_Theta[theta_index] * LG_Va_params[i][v_Theta_index][theta_index]

                # CLF decrease constraint (with relaxation)
                constraints.append(
                    Lf_Va_params[i][v_Theta_index]
                    + LF_Va_params[i][v_Theta_index] @ v_Theta + LFGammadV_Va_params[i][v_Theta_index]
                    + (Lg_Va_params[i][v_Theta_index] + sum_LG_i_Va + LGammadVaG_params[i][v_Theta_index]) @ u
                    + self.clf_lambda * Va_param
                    - clf_relaxations[i]
                    <= 0
                )

        # Control limit constraints
        upper_lim, lower_lim = self.dynamics_model.control_limits
        for control_idx in range(self.dynamics_model.n_controls):
            constraints.append(u[control_idx] >= lower_lim[control_idx])
            constraints.append(u[control_idx] <= upper_lim[control_idx])

        # And define the objective
        objective_expression = cp.sum_squares(u - u_ref_param)
        for r in clf_relaxations:
            objective_expression = objective_expression + cp.multiply(clf_relaxation_penalty_param, r)
        objective = cp.Minimize(objective_expression)

        # Finally, create the optimization problem
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()
        variables = [u] + clf_relaxations
        parameters = [Va_param, u_ref_param, clf_relaxation_penalty_param]
        for s_idx in range(len(scenarios)):
            parameters = parameters + Lf_Va_params[s_idx] + Lg_Va_params[s_idx] + LF_Va_params[s_idx]
            # for theta_dim_idx in range(n_params):
            parameters = parameters + LFGammadV_Va_params[s_idx]
            for LG_Va_cluster in LG_Va_params[s_idx]:
                parameters = parameters + LG_Va_cluster
            parameters = parameters + LGammadVaG_params[s_idx]

        self.differentiable_qp_solver = CvxpyLayer(
            problem, variables=variables, parameters=parameters
        )

    def V_with_jacobian(self, x: torch.Tensor, theta_hat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the CLF value and its Jacobian

        args:
            x: bs x self.dynamics_model.n_dims the points at which to evaluate the CLF
            theta_hat: bs x self.dynamics_model.n_params the estimated parameters of the dynamics model
        returns:
            Va: bs tensor of CLF values
            JxV: bs x 1 x self.dynamics_model.n_dims Jacobian of each row of Va wrt x
            JthV: bs x 1 x self.dynamics_model.n_params Jacobian of each row of Va wrt theta_hat
        """
        # Create batches of x-theta pairs
        x_theta = torch.cat([x, theta_hat], dim=1)

        # First, get the Lyapunov function value and gradient at this state
        P = self.dynamics_model.P.type_as(x_theta)
        # Reshape to use pytorch's bilinear function
        P = P.reshape(
            1,
            self.dynamics_model.n_dims+self.dynamics_model.n_params,
            self.dynamics_model.n_dims+self.dynamics_model.n_params
        )
        Va = 0.5 * F.bilinear(x_theta, x_theta, P).squeeze()
        Va = Va.reshape(x_theta.shape[0])

        # Reshape again for the gradient calculation
        P = P.reshape(
            self.dynamics_model.n_dims+self.dynamics_model.n_params, self.dynamics_model.n_dims+self.dynamics_model.n_params
        )
        JxV = F.linear(x, P[:self.dynamics_model.n_dims, :self.dynamics_model.n_dims]) + \
              2*F.linear(theta_hat, P[self.dynamics_model.n_dims:self.dynamics_model.n_dims+self.dynamics_model.n_params, :self.dynamics_model.n_dims])
        JxV = JxV.reshape(x.shape[0], 1, self.dynamics_model.n_dims)

        JthV = F.linear(theta_hat, P[self.dynamics_model.n_dims:self.dynamics_model.n_dims+self.dynamics_model.n_params, self.dynamics_model.n_dims:self.dynamics_model.n_dims+self.dynamics_model.n_params]) + \
                2*F.linear(x, P[:self.dynamics_model.n_dims, self.dynamics_model.n_dims:self.dynamics_model.n_dims+self.dynamics_model.n_params])
        JthV = JthV.reshape(x_theta.shape[0], 1, self.dynamics_model.n_params)

        return Va, JxV, JthV

    def V(self, x: torch.Tensor, theta_hat: torch.Tensor) -> torch.Tensor:
        """Compute the value of the CLF"""
        V, _, _ = self.V_with_jacobian(x, theta_hat)
        return V

    def V_lie_derivatives(
        self, x: torch.Tensor, theta_hat: torch.Tensor, scenarios: Optional[ScenarioList] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """Compute the Lie derivatives of the CLF V along the control-affine dynamics

        args:
            x: bs x self.dynamics_model.n_dims tensor of state
            theta_hat: bs x self.dynamics_model.n_params the estimated parameters of the dynamics model
            scenarios: optional list of scenarios. Defaults to self.scenarios
        returns:
            Lf_V: bs x len(scenarios) x 1 tensor of Lie derivatives of V
                  along f
            LF_V: bs x len(scenarios) x self.dynamics_nodel.n_params tensor
                    of Lie derivatives of V along F
            LFGammadV_V: bs x len(scenarios) x 1 tensor
            Lg_V: bs x len(scenarios) x self.dynamics_model.n_controls tensor
                  of Lie derivatives of V along g
            list_LGi_V: list of bs x len(scenarios) x self.n_controls tensors each one representing
                        the Lie derivatives of V along G_i (one i for each dimension of the unknown parameter theta)
            LGammadVG_V: bs x len(scenarios) x self.dynamics_model.n_controls

        usage:
            Lf_V, LF_V, LFGammadV_V, Lg_V, list_LGi_V, LGammadVG_V = self.V_lie_derivatives(x, theta_hat)
        """
        # Constants

        if scenarios is None:
            scenarios = self.scenarios
        n_scenarios = len(scenarios)
        Gamma = self.Gamma
        n_dims = self.dynamics_model.n_dims


        # Get the Jacobian of V for each entry in the batch
        V, gradV_x, gradV_theta = self.V_with_jacobian(x, theta_hat)

        # We need to compute Lie derivatives for each scenario
        batch_size = x.shape[0]
        Lf_V = torch.zeros(batch_size, n_scenarios, 1)
        LF_V = torch.zeros(batch_size, n_scenarios, self.dynamics_model.n_params)
        LFGammadV_V = torch.zeros(batch_size, n_scenarios, 1)
        Lg_V = torch.zeros(batch_size, n_scenarios, self.dynamics_model.n_controls)
        LGammadVG_V = torch.zeros(batch_size, n_scenarios, self.dynamics_model.n_controls)

        Gamma_copied = Gamma.repeat(batch_size, 1, 1)

        Lf_V = Lf_V.type_as(x)
        LF_V = LF_V.type_as(x)
        LFGammadV_V = LFGammadV_V.type_as(x)
        Lg_V = Lg_V.type_as(x)
        LGammadVG_V = LGammadVG_V.type_as(x)

        list_LGi_V = []
        for param_index in range(self.dynamics_model.n_params):
            LGi_V = torch.zeros(batch_size, n_scenarios, self.dynamics_model.n_controls)
            LGi_V = LGi_V.type_as(x)
            list_LGi_V.append(LGi_V)


        for i in range(n_scenarios):
            # Get the dynamics f and g for this scenario
            s = scenarios[i]
            f = self.dynamics_model._f(x, s)
            F = self.dynamics_model._F(x, s)
            g = self.dynamics_model._g(x, s)
            G = self.dynamics_model._G(x, s)

            # Multiply these with the Jacobian to get the Lie derivatives
            Lf_V[:, i, :] = torch.bmm(gradV_x, f).squeeze(1)
            LF_V[:, i, :] = torch.bmm(gradV_x, F).squeeze(1)
            Lg_V[:, i, :] = torch.bmm(gradV_x, g).squeeze(1)

            LF_V_scenario = LF_V[:, i, :].unsqueeze(1)
            LFGamma = torch.bmm(LF_V_scenario, Gamma_copied)
            # LFGammadV_V[:, i, :] = torch.bmm(LFGamma, gradV_theta.mT).squeeze(1)
            LFGammadV_V[:, i, :] = torch.bmm(LFGamma, gradV_theta.transpose(-2,-1)).squeeze(1) # TODO: Reevaluate if we want to hide mT

            # list_LGi_V
            for param_index in range(self.dynamics_model.n_params):
                G_p = G[:, :, :, param_index].reshape((batch_size, self.dynamics_model.n_dims, self.dynamics_model.n_controls))
                list_LGi_V[param_index][:, i, :] = torch.bmm(gradV_x, G_p).squeeze(1)

            # LGammadVG
            for mode_index in range(self.dynamics_model.n_params):
                G_p = G[:, :, :, mode_index].reshape((batch_size, self.dynamics_model.n_dims, self.dynamics_model.n_controls))

                # Compute Complicated Term
                # gradVx_Gamma = torch.bmm(gradV_x, Gamma_copied)
                # Gamma_gradVtheta = torch.bmm(Gamma_copied, gradV_theta.mT)
                Gamma_gradVtheta = torch.bmm(Gamma_copied, gradV_theta.transpose(-2, -1)) # TODO: Reevaluate if hiding mT is fine

                LGammadVG_V_current = torch.zeros((batch_size, 1, self.dynamics_model.n_controls))
                LGammadVG_V_current[:, :, :] = LGammadVG_V[:, i, :].unsqueeze(1)

                Gamma_gradVtheta_i = Gamma_gradVtheta[:, mode_index, 0].reshape((batch_size, 1, 1))
                coeff = torch.kron(
                    Gamma_gradVtheta_i,
                    torch.eye(n_dims).type_as(x).reshape((1, n_dims, n_dims)),
                )

                GammadVG_i = torch.bmm(coeff, G_p)
                LGammadVG_V[:, i, :] = (LGammadVG_V_current + torch.matmul(gradV_x, GammadVG_i)).squeeze(1)

        eps = 1e-6
        for b_idx in range(Lg_V.shape[0]):
            if torch.norm(Lg_V[b_idx, :, :]) < eps:
                print("Lg_V is zero")
                print("x", x[b_idx, :])
                print("theta_hat", theta_hat[b_idx, :])
                print("dVdx", gradV_x[b_idx, :, :])
                # print("dVdtheta", gradV_theta.mT[b_idx, :, :])
                print("dVdtheta", gradV_theta.transpose(-2, -1)[b_idx, :, :]) # TODO: Reevaluate if hiding mT is fine, or if we want to upgrade torch
                print("V", V[b_idx])
                print("Lf_V ", Lf_V[b_idx, :, :])

        # return the Lie derivatives
        return Lf_V, LF_V, LFGammadV_V, Lg_V, list_LGi_V, LGammadVG_V

    def Vdot_for_scenario(self, scenario_idx: int,  x: torch.Tensor, theta_hat: torch.Tensor, u: torch.Tensor, scenarios: Optional[ScenarioList] = None) -> torch.Tensor:
        """
        Vdot_for_scenario
        Description
            This function computes the modified version of V-dot which aCLFs use to guarantee convergence.
        Input
        Input
            scenario_idx: an integer indicating which scenario to calculate Vdot for
            x: a bs x self.dynamics_model.n_dims tensor containing the states in the current batch
            theta_hat: a bs x self.dynamics_model.n_params tensor containing the parameters in the current batch
            u: a bs x self.dynamics_model.n_controls tensor containing the inputs at each state-parameter constant

        """
        # Constants
        bs = x.shape[0]
        n_controls = self.dynamics_model.n_controls
        n_params = self.dynamics_model.n_params
        if scenarios is None:
            scenarios = self.scenarios

        n_scenarios = len(scenarios)

        # Compute Lie Derivatives
        Lf_Va, LF_Va, LFGammadVa_Va, Lg_V, list_LGi_V, LGammadVG_V = self.V_lie_derivatives(x, theta_hat)

        # Use the dynamics to compute the derivative of V at each corner of V_Theta
        sum_LG_V = torch.zeros((bs, n_scenarios, n_controls))
        for theta_dim in range(n_params):
            sum_LG_V = sum_LG_V + torch.bmm(theta_hat[:, theta_dim].reshape((bs, 1, 1)), list_LGi_V[theta_dim])

        # Should we use the true theta in this computation? I don't think so.
        Vdot = Lf_Va[:, scenario_idx, :].unsqueeze(1) + \
            torch.bmm(LF_Va[:, scenario_idx, :].unsqueeze(1), theta_hat.reshape((theta_hat.shape[0], theta_hat.shape[1], 1))) + \
            LFGammadVa_Va[:, scenario_idx, :].unsqueeze(1) + \
            torch.bmm(
                Lg_V[:, scenario_idx, :].unsqueeze(1) + sum_LG_V + LGammadVG_V,
                u.reshape(-1, self.dynamics_model.n_controls, 1),
           )

        return Vdot

    def u_reference(self, x: torch.Tensor, theta_hat: torch.Tensor) -> torch.Tensor:
        """Determine the reference control input."""
        # Here we use the nominal controller as the reference, but subclasses can
        # override this
        return self.dynamics_model.u_nominal(x, theta_hat)

    def _solve_CLF_QP_gurobi(
        self,
        x: torch.Tensor,
        u_ref: torch.Tensor,
        V: torch.Tensor,
        Lf_V: torch.Tensor,
        Lg_V: torch.Tensor,
        LF_V: torch.Tensor,
        LFGammadV_V: torch.Tensor,
        list_LGi_V: torch.Tensor,
        LGammadVG_V: torch.Tensor,
        relaxation_penalty: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Determine the control input for a given state using a QP. Solves the QP using
        Gurobi, which does not allow for backpropagation.

        args:
            x: bs x self.dynamics_model.n_dims tensor of state
            u_ref: bs x self.dynamics_model.n_controls tensor of reference controls
            V: bs x n_scenarios x 1 tensor of CLF values,
            Lf_V: bs x n_scenarios x 1 tensor of CLF Lie derivatives,
            Lg_V: bs x n_scenarios x self.dynamics_model.n_controls tensor of CLF Lie derivatives,
            LF_V: bs x n_scenarios x self.dynamics_model.n_params tensor of CLF Lie derivatives,
            LFGammadV_V: bs x n_scenarios x 1 tensor of CLF Lie derivatives, dV/dx * F(x) * Gamma * dV/dtheta.T
            list_LGi_V: bs x n_scenarios x self.dynamics_model.n_controls x self.dynamics_model.n_params tensor of CLF Lie derivatives,
            LGammadVG_V: bs x self.dynamics_model.n_controls tensor of CLF Lie derivatives,
            relaxation_penalty: the penalty to use for CLF relaxation.
        returns:
            u: bs x self.dynamics_model.n_controls tensor of control inputs
            relaxation: bs x 1 tensor of how much the CLF had to be relaxed in each
                        case
        """
        # To find the control input, we want to solve a QP constrained by
        #
        # L_f V + LF_V * theta + (L_g V + LGammadVG + \sum LG_V[:,:,i] * theta_i ) u + lambda V <= 0
        #
        # To ensure that this QP is always feasible, we relax the constraint
        #
        # L_f V + LF_V * theta + (L_g V + LGammadVG + \sum LG_V[:,:,i] * theta_i ) u + lambda V - r <= 0
        #                              r >= 0
        #
        # and add the cost term relaxation_penalty * r.
        #
        # We want the objective to be to minimize
        #
        #           ||u - u_ref||^2 + relaxation_penalty * r^2
        #
        # This reduces to (ignoring constant terms)
        #
        #           u^T I u - 2 u_ref^T u + relaxation_penalty * r^2

        # Constants
        batch_size = x.shape[0]
        n_controls = self.dynamics_model.n_controls
        n_params = self.dynamics_model.n_params
        n_scenarios = self.n_scenarios
        allow_relaxation = not (relaxation_penalty == float("inf"))

        Theta_vertices = pc.extreme(self.dynamics_model.Theta)
        n_Theta_vertices = Theta_vertices.shape[0]

        # Create the set of Lie derivatives
        Thetas_Lf_V_scenarios, Thetas_Lg_V_scenarios, Thetas_LF_V_scenarios, Thetas_LFGammadV_V_scenarios = [], [], [], []
        Thetas_list_LGi_V_scenarios, Thetas_LGammadVG_V_scenarios = [], []
        for s_idx in range(len(self.scenarios)):

            # Create lie derivatives ONCE
            Thetas_Lf_V, Thetas_Lg_V, Thetas_LF_V, Thetas_LFGammadV_V = [], [], [], []
            Thetas_list_LGi_V, Thetas_LGammadVG_V = [], []
            for v_Theta in Theta_vertices:
                theta0 = torch.Tensor(v_Theta)
                theta_corner = theta0.repeat((batch_size, 1))

                # Compute Lie Derivatives here
                Lf_V, LF_V, LFGammadV_V, Lg_V, list_LGi_V, LGammadVG_V = self.V_lie_derivatives(x, theta_corner)

                Thetas_Lf_V.append(Lf_V)
                Thetas_LF_V.append(LF_V)
                Thetas_Lg_V.append(Lg_V)
                Thetas_LFGammadV_V.append(LFGammadV_V)
                Thetas_list_LGi_V.append(list_LGi_V)
                Thetas_LGammadVG_V.append(LGammadVG_V)

            # Save all of these per scenario
            Thetas_Lf_V_scenarios.append(Thetas_Lf_V)
            Thetas_LF_V_scenarios.append(Thetas_LF_V)
            Thetas_LFGammadV_V_scenarios.append(Thetas_LFGammadV_V)
            Thetas_Lg_V_scenarios.append(Thetas_Lg_V)
            Thetas_list_LGi_V_scenarios.append(Thetas_list_LGi_V)
            Thetas_LGammadVG_V_scenarios.append(Thetas_LGammadVG_V)

        # Solve a QP for each row in x
        bs = x.shape[0]
        u_result = torch.zeros(bs, n_controls)
        r_result = torch.zeros(bs, n_scenarios, n_Theta_vertices)
        # r_result = torch.zeros(bs, n_scenarios)
        for batch_idx in range(bs):
            # Skip any bad points
            if (
                torch.isnan(x[batch_idx]).any()
                or torch.isinf(x[batch_idx]).any()
                or torch.isnan(Lg_V[batch_idx]).any()
                or torch.isinf(Lg_V[batch_idx]).any()
                or torch.isnan(Lf_V[batch_idx]).any()
                or torch.isinf(Lf_V[batch_idx]).any()
            ):
                continue

            # Instantiate the model
            model = gp.Model("aclf_qp")
            # Create variables for control input and (optionally) the relaxations
            upper_lim, lower_lim = self.dynamics_model.control_limits
            upper_lim = upper_lim.cpu().numpy()
            lower_lim = lower_lim.cpu().numpy()
            u = model.addMVar(n_controls, lb=lower_lim, ub=upper_lim)
            if allow_relaxation:
                r_set = []
                for corner_idx in range(n_Theta_vertices):
                    r_set.append(model.addMVar(n_scenarios, lb=0, ub=GRB.INFINITY))
                # r = model.addMVar(n_scenarios, lb=0, ub=GRB.INFINITY)

            # Define the cost
            Q = np.eye(n_controls)
            u_ref_np = u_ref[batch_idx, :].detach().cpu().numpy()
            objective = u @ Q @ u - 2 * u_ref_np @ Q @ u + u_ref_np @ Q @ u_ref_np
            if allow_relaxation:
                for corner_idx in range(self.dynamics_model.Theta.dim):
                    r = r_set[corner_idx]
                    relax_penalties = relaxation_penalty * np.ones(n_scenarios)
                    objective += relax_penalties @ r
                # relax_penalties = relaxation_penalty * np.ones(n_scenarios)
                # objective += relax_penalties @ r

            # Now build the CLF constraints
            for i in range(n_scenarios):

                # Lg_V_np = Lg_V[batch_idx, i, :].detach().cpu().numpy()
                # Lf_V_np = Lf_V[batch_idx, i, :].detach().cpu().numpy()
                # LF_V_np = LF_V[batch_idx, i, :].detach().cpu().numpy()
                # LFGammadV_V_np = LFGammadV_V[batch_idx, i, :].detach().cpu().numpy()
                # list_LGi_V_np = []
                # for theta_dim in range(n_params):
                #     list_LGi_V_np.append(list_LGi_V[theta_dim][batch_idx, i, :].detach().cpu().numpy())
                # LGammadVG_V_np = LGammadVG_V[batch_idx, i, :].detach().cpu().numpy()

                V_np = V[batch_idx].detach().cpu().numpy()

                for v_idx in range(Theta_vertices.shape[0]):
                    v_Theta = Theta_vertices[v_idx]

                    # Retrieve lie derivatives
                    Lf_V_torch = Thetas_Lf_V_scenarios[i][v_idx]
                    Lf_V_np = Lf_V_torch[batch_idx, i, :].detach().cpu().numpy()

                    LF_V_torch = Thetas_LF_V_scenarios[i][v_idx]
                    LF_V_np = LF_V_torch[batch_idx, i, :].detach().cpu().numpy()

                    LFGammadV_V_torch = Thetas_LFGammadV_V_scenarios[i][v_idx]
                    LFGammadV_V_np = LFGammadV_V_torch[batch_idx, i, :].detach().cpu().numpy()

                    Lg_V_torch = Thetas_Lg_V_scenarios[i][v_idx]
                    Lg_V_np = Lg_V_torch[batch_idx, i, :].detach().cpu().numpy()

                    list_LGi_V_np = []
                    list_LGi_V_torch = Thetas_list_LGi_V_scenarios[i][v_idx]
                    for theta_dim in range(n_params):
                        list_LGi_V_np.append(
                            list_LGi_V_torch[theta_dim][batch_idx, i, :].detach().cpu().numpy()
                        )

                    LGammadVG_V_torch = Thetas_LGammadVG_V_scenarios[i][v_idx]
                    LGammadVG_V_np = LGammadVG_V_torch[batch_idx, i, :].detach().cpu().numpy()

                    # Create term from the sum of dv/dx * Gamma * dV_dth.T * G_i
                    sum_LG_V_np = np.zeros(n_controls)
                    for theta_dim in range(self.dynamics_model.n_params):
                        sum_LG_V_np += v_Theta[theta_dim] * list_LGi_V_np[theta_dim]
                    clf_constraint = Lf_V_np + LF_V_np @ v_Theta + LFGammadV_V_np + \
                        (Lg_V_np + sum_LG_V_np + LGammadVG_V_np) @ u + self.clf_lambda * V_np
                    #clf_constraint = Lf_V_np + Lg_V_np @ u + self.clf_lambda * V_np
                    if allow_relaxation:
                        r = r_set[v_idx]
                        clf_constraint -= r[i]

                    model.addConstr(clf_constraint <= 0.0, name=f"Scenario {i}, Corner {v_idx} Decrease")

            # Optimize!
            model.setParam("DualReductions", 0)
            model.setObjective(objective, GRB.MINIMIZE)
            model.optimize()

            if model.status != GRB.OPTIMAL:
                # Make the relaxations nan if the problem was infeasible, as a signal
                # that something has gone wrong
                if allow_relaxation:
                    for i in range(n_scenarios):
                        for v_Theta_idx in range(n_Theta_vertices):
                            r_result[batch_idx, i, v_Theta_idx] = torch.tensor(float("nan"))
                continue

            # Extract the results
            for i in range(n_controls):
                u_result[batch_idx, i] = torch.tensor(u[i].x)
            if allow_relaxation:
                for i in range(n_scenarios):
                    for th_idx in range(n_Theta_vertices):
                        r_result[batch_idx, i, th_idx] = torch.tensor(r_set[th_idx][i].x)
                    # r_result[batch_idx, i] = torch.tensor(r[i].x)

        return u_result.type_as(x), r_result.type_as(x)

    def _solve_CLF_QP_cvxpylayers(
        self,
        x: torch.Tensor,
        u_ref: torch.Tensor,
        V: torch.Tensor,
        relaxation_penalty: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Determine the control input for a given state using a QP. Solves the QP using
        CVXPyLayers, which does allow for backpropagation, but is slower and less
        accurate than Gurobi.

        args:
            x: bs x self.dynamics_model.n_dims tensor of state
            u_ref: bs x self.dynamics_model.n_controls tensor of reference controls
            V: bs x 1 tensor of CLF values,
            Lf_V: bs x 1 tensor of CLF Lie derivatives,
            Lg_V: bs x self.dynamics_model.n_controls tensor of CLF Lie derivatives,
            LF_V: bs x n_scenarios x self.dynamics_model.n_params tensor of CLF Lie derivatives,
            LFGammadV_V: bs x n_scenarios x 1 tensor of CLF Lie derivatives, dV/dx * F(x) * Gamma * dV/dtheta.T
            list_LGi_V: list of self.dynamics_model.n_params tensors with shape bs x n_scenarios x self.dynamics_model.n_controls. Each tensor is a CLF Lie derivatives,
            LGammadVG_V: list of bs x self.dynamics_model.n_controls tensor of CLF Lie derivatives,
            relaxation_penalty: the penalty to use for CLF relaxation.
        returns:
            u: bs x self.dynamics_model.n_controls tensor of control inputs
            relaxation: bs x 1 tensor of how much the CLF had to be relaxed in each
                        case
        """
        # The differentiable solver must allow relaxation
        relaxation_penalty = min(relaxation_penalty, 1e6)
        Theta = self.dynamics_model.Theta
        V_Theta = pc.extreme(Theta)
        n_V_Theta = V_Theta.shape[0]
        bs = x.shape[0]

        # Assemble list of params
        # =======================
        params = []

        # Following the order of the paramter vector creation
        params.append(V.reshape(-1, 1))
        params.append(u_ref)
        params.append(torch.tensor([relaxation_penalty]).type_as(x))
        for s_idx in range(len(self.scenarios)):

            # Create lie derivatives ONCE
            Thetas_Lf_V, Thetas_Lg_V, Thetas_LF_V, Thetas_LFGammadV_V = [], [], [], []
            Thetas_list_LGi_V, Thetas_LGammadVG_V = [], []
            for v_Theta in V_Theta:
                theta0 = torch.Tensor(v_Theta)
                theta_corner = theta0.repeat((bs, 1))

                # Compute Lie Derivatives here
                Lf_V, LF_V, LFGammadV_V, Lg_V, list_LGi_V, LGammadVG_V = self.V_lie_derivatives(x, theta_corner)

                Thetas_Lf_V.append(Lf_V)
                Thetas_LF_V.append(LF_V)
                Thetas_Lg_V.append(Lg_V)
                Thetas_LFGammadV_V.append(LFGammadV_V)
                Thetas_list_LGi_V.append(list_LGi_V)
                Thetas_LGammadVG_V.append(LGammadVG_V)

            # Save all entries for Lf_V
            for v_Theta_idx in range(n_V_Theta):
                Lf_V_corner = Thetas_Lf_V[v_Theta_idx]

                params.append(Lf_V_corner[:, s_idx, :])

                #print("Lf_V = ", Lf_V_corner[:, s_idx, :])

            # Save all entries for Lg_V
            for v_Theta_idx in range(n_V_Theta):
                Lg_V_corner = Thetas_Lg_V[v_Theta_idx]

                params.append(Lg_V_corner[:, s_idx, :])

                #print("Lg_V = ", Lg_V_corner[:, s_idx, :])

            # Save all entries for LF_V
            for v_Theta_idx in range(n_V_Theta):
                LF_V_corner = Thetas_LF_V[v_Theta_idx]

                params.append(LF_V_corner[:, s_idx, :])

            # Save all entries for LFGammadV_V
            for v_Theta_idx in range(n_V_Theta):
                LFGammadV_V_corner = Thetas_LFGammadV_V[v_Theta_idx]

                params.append(LFGammadV_V_corner[:, s_idx, :])

            # Save all entries for list_LGi_V
            for v_Theta_idx in range(n_V_Theta):
                list_LGi_V_corner = Thetas_list_LGi_V[v_Theta_idx]

                for th_dim in range(self.dynamics_model.n_params):
                    params.append(list_LGi_V_corner[th_dim][:, s_idx, :])

            # Save all entries for LGammadVG_V
            for v_Theta_idx in range(n_V_Theta):
                LGammadVG_V_corner = Thetas_LGammadVG_V[v_Theta_idx]

                params.append(LGammadVG_V_corner[:, s_idx, :])

        # for i in range(self.n_scenarios):
        #     params.append(Lf_V[:, i, :])
        # for i in range(self.n_scenarios):
        #     params.append(Lg_V[:, i, :])
        # for i in range(self.n_scenarios):
        #     params.append(LF_V[:, i, :])
        #
        # for i in range(self.n_scenarios):
        #     # LFGammadV_V
        #     params.append(LFGammadV_V[:, i, :])
        # for i in range(self.n_scenarios):
        #     # list_LGi_V
        #     for theta_dim_index in range(self.dynamics_model.n_params):
        #         params.append(
        #             list_LGi_V[theta_dim_index][:, i, :]
        #         )
        # for i in range(self.n_scenarios):
        #     # LGammadVG_V
        #     params.append(LGammadVG_V[:, i, :])

        # print("params", params)

        # We've already created a parameterized QP solver, so we can use that
        result = self.differentiable_qp_solver(
            *params,
            solver_args={"max_iters": 50000000},
        )

        # Extract the results
        u_result = result[0]
        r_result = torch.hstack(result[1:])

        return u_result.type_as(x), r_result.type_as(x)

    def solve_CLF_QP(
        self,
        x,
        theta_hat: torch.Tensor,
        relaxation_penalty: Optional[float] = None,
        u_ref: Optional[torch.Tensor] = None,
        requires_grad: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Description
            Determine the control input for a given state using a QP

        args:
            x: bs x self.dynamics_model.n_dims tensor of state
            theta_hat: bs x self.dynamics_model.n_dims tensor of state
            relaxation_penalty: the penalty to use for CLF relaxation, defaults to
                                self.clf_relaxation_penalty
            u_ref: allows the user to supply a custom reference input, which will
                   bypass the self.u_reference function. If provided, must have
                   dimensions bs x self.dynamics_model.n_controls. If not provided,
                   default to calling self.u_reference.
            requires_grad: if True, use a differentiable layer
        returns:
            u: bs x self.dynamics_model.n_controls tensor of control inputs
            relaxation: bs x 1 tensor of how much the CLF had to be relaxed in each
                        case
        """
        # Get the value of the CLF and its Lie derivatives
        V = self.V(x, theta_hat)
        Lf_V, LF_V, LFGammadV_V, Lg_V, list_LGi_V, LGammadVG_V = self.V_lie_derivatives(x, theta_hat)

        # Get the reference control input as well
        if u_ref is not None:
            err_message = f"u_ref must have {x.shape[0]} rows, but got {u_ref.shape[0]}"
            assert u_ref.shape[0] == x.shape[0], err_message
            err_message = f"u_ref must have {self.dynamics_model.n_controls} cols,"
            err_message += f" but got {u_ref.shape[1]}"
            assert u_ref.shape[1] == self.dynamics_model.n_controls, err_message
        else:
            u_ref = self.u_reference(x, theta_hat)

        # Apply default penalty if needed
        if relaxation_penalty is None:
            relaxation_penalty = self.clf_relaxation_penalty

        # Figure out if we need to use a differentiable solver (determined by whether
        # the input x requires a gradient or not)
        if requires_grad:
            return self._solve_CLF_QP_cvxpylayers(
                x, u_ref, V, relaxation_penalty
            )
        else:
            return self._solve_CLF_QP_gurobi(
                x, u_ref, V, Lf_V, Lg_V, LF_V, LFGammadV_V, list_LGi_V, LGammadVG_V, relaxation_penalty
            )

    def u(self, x, theta_hat: torch.Tensor) -> torch.Tensor:
        """Get the control input for a given state"""
        u, _ = self.solve_CLF_QP(x, theta_hat)
        return u

    def tau(self, x: torch.Tensor, theta_hat: torch.Tensor, u: torch.Tensor, scenario: Scenario) -> torch.Tensor:
        """
        Description
            Compute the function which changes the estimate of the parameters
        Output
            tau: bs x n_params tensor defining the value of the function at each of these points
        """
        # Constants
        bs = x.shape[0]
        n_scenarios = self.n_scenarios
        n_dims = self.dynamics_model.n_dims
        n_controls = self.dynamics_model.n_controls
        n_params = self.dynamics_model.n_params

        # Get the gradient of V w.r.t. x
        _, dVdx, dVdth = self.V_with_jacobian(x, theta_hat)

        # Get the lie derivatives
        Lf_V, LF_V, LFGammadV_V, Lg_V, list_LGi_V, LGammadVG_V = self.V_lie_derivatives(x, theta_hat, [scenario])

        # Compute tau (tau = LF_V + LGalpha_V)

        LF_V_scenario = torch.zeros((bs, 1, self.dynamics_model.n_params)).type_as(x)
        LF_V_scenario[:, :, :] = LF_V[:, 0, :].unsqueeze(dim=1)

        Galpha_scenario = torch.zeros((bs, n_dims, n_params))
        G = self.dynamics_model._G(x, scenario)
        for th_idx in range(n_params):
            G_th = G[:, :, :, th_idx].reshape((bs, n_dims, n_controls))
            Galpha_scenario[:, :, th_idx] = torch.bmm(G_th, u.reshape((bs, n_controls, 1))).squeeze(dim=2)

        LGalpha_V_scenario = torch.zeros((bs, 1, self.dynamics_model.n_params))
        LGalpha_V_scenario[:, :, :] = torch.bmm(dVdx, Galpha_scenario)

        tau = torch.zeros((bs, self.dynamics_model.n_params)).type_as(x)
        tau[:, :] = (LF_V_scenario + LGalpha_V_scenario).squeeze(dim=1)

        return tau

    def closed_loop_estimator_dynamics(self, x: torch.Tensor, theta_hat: torch.Tensor, u: torch.Tensor, scenario: Scenario):
        """
        closed_loop_estimator_dynamics
        Description
            Computes the dynamics of the estimator (how it changes over time) when given
            the current aCLF controller.
        Inputs

        Outputs
            thetadot: bs x self.dynamics_model.n_params tensor of time derivatives of x
        """

        # Constants
        batch_size = x.shape[0]
        n_params = self.dynamics_model.n_params
        Gamma = self.Gamma

        # Algorithm

        tau = self.tau(x, theta_hat, u, scenario=scenario)

        Gamma_copied = Gamma.repeat(batch_size, 1, 1)

        Gamma_tau = torch.bmm(
            Gamma_copied,
            tau.reshape((batch_size, n_params, 1))
        )

        thetadot = torch.zeros((batch_size, n_params))
        thetadot[:, :] = Gamma_tau.squeeze(dim=2)

        return thetadot

    def V_oracle(
            self,
            x: torch.Tensor,
            theta_hat: torch.Tensor,
            theta: torch.Tensor,
    ):
        """
        V_oracle
        Description:
            Determines the value of the "oracle" CLF which is the clf defined over the
            x, theta-hat, theta space.
            This function should generally be decreasing.
        args:
            x: bs x self.dynamics_model.n_dims tensor of state
            theta_hat: bs x self.dynamics_model.n_params tensor of parameter estimate
            theta: bs x self.dynamics_model.n_params tensor of true parameter values
        """
        # Constants
        bs = x.shape[0]
        Gamma = self.Gamma

        # Get Va
        Va = self.V(x, theta_hat)

        # Compute weighted theta_err term.
        theta_err = theta - theta_hat
        Gamma_copied = Gamma.repeat(bs, 1, 1)
        theta_err_Gamma = torch.bmm(
            theta_err.unsqueeze(1),
            Gamma_copied
        )
        err_term = 0.5 * torch.bmm(
            theta_err_Gamma,
            theta_err.unsqueeze(2)
        ).squeeze()

        return Va + err_term