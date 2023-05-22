from warnings import warn

from .control_affine_parameter_affine_system import ControlAffineParameterAffineSystem
from .load_sharing_manipulator import LoadSharingManipulator
from .scalar_demo_capa2_system import ScalarCAPA2Demo
from .tumbling_target import TumblingTarget
from .tumbling_target2 import TumblingTarget2
from .adaptive_pusher_slider_force_input import AdaptivePusherSliderStickingForceInput
from .adaptive_pusher_slider_force_input_more_obstacles import AdaptivePusherSliderStickingForceInput_MoreObstacles

__all__ = [
    "ControlAffineParameterAffineSystem",
    "LoadSharingManipulator"
    "ScalarCAPA2Demo",
    "TumblingTarget",
    "TumblingTarget2",
    "AdaptivePusherSliderStickingForceInput",
    "AdaptivePusherSliderStickingForceInput_MoreObstacles",
]

# try:
#     from .f16 import F16  # noqa
#
#     __all__.append("F16")
# except ImportError:
#     warn("Could not import F16 module; is AeroBench installed")
