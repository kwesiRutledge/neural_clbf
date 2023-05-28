import torch

from neural_clbf.systems import ControlAffineSystem
from neural_clbf.systems.adaptive_w_scenarios import ControlAffineParameterAffineSystem2
import polytope as pc
import numpy as np

from typing import List


def normalize(
    dynamics_model: ControlAffineSystem, x: torch.Tensor, k: float = 1.0
) -> torch.Tensor:
    """Normalize the state input to [-k, k]

    args:
        dynamics_model: the dynamics model matching the provided states
        x: bs x self.dynamics_model.n_dims the points to normalize
        k: normalize non-angle dimensions to [-k, k]
    """
    x_max, x_min = dynamics_model.state_limits
    x_center = (x_max + x_min) / 2.0
    x_range = (x_max - x_min) / 2.0
    # Scale to get the input between (-k, k), centered at 0
    x_range = x_range / k
    # We shouldn't scale or offset any angle dimensions
    x_center[dynamics_model.angle_dims] = 0.0
    x_range[dynamics_model.angle_dims] = 1.0

    # Do the normalization
    return (x - x_center.type_as(x)) / x_range.type_as(x)


def normalize_with_angles(
    dynamics_model: ControlAffineSystem, x: torch.Tensor, k: float = 1.0, angle_dims: List[int] = None
) -> torch.Tensor:
    """Normalize the input using the stored center point and range, and replace all
    angles with the sine and cosine of the angles

    args:
        dynamics_model: the dynamics model matching the provided states
        x: bs x self.dynamics_model.n_dims the points to normalize
        k: normalize non-angle dimensions to [-k, k]
    """
    # Scale and offset based on the center and range
    x = normalize(dynamics_model, x, k)

    # Replace all angles with their sine, and append cosine
    if angle_dims is None:
        angle_dims = dynamics_model.angle_dims

    if len(angle_dims) > 0:
        angles = x[:, angle_dims]
        x[:, angle_dims] = torch.sin(angles)
        x = torch.cat((x, torch.cos(angles)), dim=-1)

    return x


def normalize_theta(
    dynamics_model: ControlAffineSystem, theta: torch.Tensor, k: float = 1.0
) -> torch.Tensor:
    """
    Description
        Normalize the unknown parameter input to [-k, k]

    args:
        dynamics_model: the dynamics model matching the provided states
        theta: bs x self.dynamics_model.n_params the points to normalize
        k: normalize non-angle dimensions to [-k, k]
    """
    # Constants
    Theta = dynamics_model.Theta

    # Get minimum and maximum possible values for theta
    V_Theta = pc.extreme(Theta)
    theta_max = torch.tensor(
        np.max(V_Theta, axis=0),
        dtype=torch.get_default_dtype(),
    )
    theta_min = torch.tensor(
        np.min(V_Theta, axis=0),
        dtype=torch.get_default_dtype(),
    )

    theta_center = (theta_max + theta_min) / 2.0
    theta_range = (theta_max - theta_min) / 2.0
    # Scale to get the input between (-k, k), centered at 0
    theta_range = theta_range / k
    # We shouldn't scale or offset any angle dimensions
    theta_center[dynamics_model.parameter_angle_dims] = 0.0
    theta_range[dynamics_model.parameter_angle_dims] = 1.0

    # Do the normalization
    return (theta - theta_center.type_as(theta)) / theta_range.type_as(theta)

def normalize_theta_with_angles(
    dynamics_model: ControlAffineSystem, theta: torch.Tensor, k: float = 1.0, angle_dims: List[int] = None
) -> torch.Tensor:
    """
    Description
        Normalize the input set of parameter vectors using the stored center point and range, and replace all
        angles with the sine and cosine of the angles

    args:
        dynamics_model: the dynamics model matching the provided states
        x: bs x self.dynamics_model.n_dims the points to normalize
        k: normalize non-angle dimensions to [-k, k]
    """
    # Scale and offset based on the center and range
    theta = normalize_theta(dynamics_model, theta, k)

    # Replace all angles with their sine, and append cosine
    if angle_dims is None:
        angle_dims = dynamics_model.parameter_angle_dims

    if len(angle_dims) > 0:
        theta_angles = theta[:, angle_dims]
        theta[:, angle_dims] = torch.sin(theta_angles)
        theta = torch.cat((theta, torch.cos(theta_angles)), dim=-1)

    return theta

def normalize_scenario(
    dynamics_model: ControlAffineParameterAffineSystem2, scen: torch.Tensor, k: float = 1.0
) -> torch.Tensor:
    """
    Description
        Normalize the unknown parameter input to [-k, k]

    args:
        dynamics_model: the dynamics model matching the provided states
        scen: bs x self.dynamics_model.n_scenario the points to normalize
        k: normalize non-angle dimensions to [-k, k]
    """
    # Constants
    P_scenario = dynamics_model.scenario_set

    # Get minimum and maximum possible values for theta
    V_Theta = pc.extreme(P_scenario)
    scen_max = torch.tensor(
        np.max(V_Theta, axis=0),
        dtype=torch.get_default_dtype(),
    )
    scen_min = torch.tensor(
        np.min(V_Theta, axis=0),
        dtype=torch.get_default_dtype(),
    )

    scen_center = (scen_max + scen_min) / 2.0
    scen_range = (scen_max - scen_min) / 2.0
    # Scale to get the input between (-k, k), centered at 0
    scen_range = scen_range / k
    # We shouldn't scale or offset any angle dimensions
    scen_center[dynamics_model.parameter_angle_dims] = 0.0
    scen_range[dynamics_model.parameter_angle_dims] = 1.0

    # Do the normalization
    return (scen - scen_center.type_as(scen)) / scen_range.type_as(scen)

def normalize_scenario_with_angles(
    dynamics_model: ControlAffineParameterAffineSystem2, scen: torch.Tensor, k: float = 1.0, angle_dims: List[int] = None
) -> torch.Tensor:
    """
    Description
        Normalize the input set of parameter vectors using the stored center point and range, and replace all
        angles with the sine and cosine of the angles

    args:
        dynamics_model: the dynamics model matching the provided states
        scen: bs x self.dynamics_model.n_scenario the points to normalize
        k: normalize non-angle dimensions to [-k, k]
    """
    # Scale and offset based on the center and range
    scen_out = normalize_scenario(dynamics_model, scen, k)

    # Replace all angles with their sine, and append cosine
    if angle_dims is None:
        # TODO: Define angle_dims for scenario
        raise NotImplementedError
        # angle_dims = dynamics_model.scenario_angle_dims

    if len(angle_dims) > 0:
        theta_angles = scen_out[:, angle_dims]
        scen_out[:, angle_dims] = torch.sin(theta_angles)
        scen_out = torch.cat((scen_out, torch.cos(theta_angles)), dim=-1)

    return scen_out
