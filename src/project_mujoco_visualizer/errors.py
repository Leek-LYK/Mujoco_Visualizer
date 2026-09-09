"""Exceptions raised by the MuJoCo motion visualizer."""


class VisualizerError(RuntimeError):
    """Base class for user-facing visualizer errors."""


class MotionFormatError(VisualizerError, ValueError):
    """The motion CSV cannot be interpreted safely."""


class ModelLoadError(VisualizerError):
    """A MuJoCo model could not be discovered or compiled."""


class JointMappingError(VisualizerError, ValueError):
    """CSV joint fields do not match the MuJoCo model."""
