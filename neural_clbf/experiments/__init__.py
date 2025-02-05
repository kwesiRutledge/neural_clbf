from warnings import warn

from .experiment import Experiment
from .experiment_suite import ExperimentSuite

from .clf_contour_experiment import CLFContourExperiment
from .clf_verification_experiment import CLFVerificationExperiment
from .bf_contour_experiment import BFContourExperiment
from .lf_contour_experiment import LFContourExperiment
from .rollout_time_series_experiment import RolloutTimeSeriesExperiment
from .rollout_norm_experiment import RolloutNormExperiment
from .rollout_state_space_experiment import RolloutStateSpaceExperiment
from .rollout_success_rate_experiment import RolloutSuccessRateExperiment
from .car_s_curve_experiment import CarSCurveExperiment
from .obs_bf_verification_experiment import ObsBFVerificationExperiment
# from .obstacle_avoidance_pushing_experiment import PusherObstacleAvoidanceExperiment
from neural_clbf.experiments.adaptive.aclf_contour_experiment import AdaptiveCLFContourExperiment
from .rollout_state_parameter_space_experiment import RolloutStateParameterSpaceExperiment
from .rollout_timing_comparison_experiment import ACLFRolloutTimingExperiment
from .rollout_state_parameter_space_experiment_multipleslices import RolloutStateParameterSpaceExperimentMultiple
from neural_clbf.experiments.adaptive.rollout_manipulator_convergence_experiment import RolloutManipulatorConvergenceExperiment

__all__ = [
    "Experiment",
    "ExperimentSuite",
    "CLFContourExperiment",
    "CLFVerificationExperiment",
    "BFContourExperiment",
    "LFContourExperiment",
    "RolloutTimeSeriesExperiment",
    "RolloutStateSpaceExperiment",
    "RolloutSuccessRateExperiment",
    "RolloutNormExperiment",
    "CarSCurveExperiment",
    "CarSCurveExperiment2",
    "ObsBFVerificationExperiment",
    # "PusherObstacleAvoidanceExperiment",
    "AdaptiveCLFContourExperiment",
    "RolloutStateParameterSpaceExperiment",
    "ACLFRolloutTimingExperiment",
    "RolloutStateParameterSpaceExperimentMultiple",
    "RolloutManipulatorConvergenceExperiment",
]

try:
    from .turtlebot_hw_state_feedback_experiment import (  # noqa: F401
        TurtlebotHWStateFeedbackExperiment,
    )
    from .turtlebot_hw_obs_feedback_experiment import (  # noqa: F401
        TurtlebotHWObsFeedbackExperiment,
    )

    __all__.append("TurtlebotHWStateFeedbackExperiment")
    __all__.append("TurtlebotHWObsFeedbackExperiment")
except ImportError:
    warn("Could not import HW module; is ROS installed?")
