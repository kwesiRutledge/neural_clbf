from warnings import warn

from .control_affine_parameter_affine_system import ControlAffineParameterAffineSystem
from .load_sharing_manipulator import LoadSharingManipulator

__all__ = [
    "ControlAffineParameterAffineSystem",
    "LoadSharingManipulator"
]

# try:
#     from .f16 import F16  # noqa
#
#     __all__.append("F16")
# except ImportError:
#     warn("Could not import F16 module; is AeroBench installed")
