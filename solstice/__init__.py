"""Solstice, a library for creating and scaling experiments in JAX."""

from solstice.experiment import Experiment
from solstice.metrics import ClassificationMetrics, Metrics
from solstice.trainer import (
    train,
    test,
    Callback,
    CheckpointingCallback,
    EarlyStoppingCallback,
    LoggingCallback,
    ProfilingCallback,
)
from solstice.utils import replace, EarlyStoppingException

__all__ = (
    "Experiment",
    "Metrics",
    "ClassificationMetrics",
    "Callback",
    "CheckpointingCallback",
    "EarlyStoppingCallback",
    "LoggingCallback",
    "ProfilingCallback",
    "train",
    "test",
    "replace",
    "EarlyStoppingException",
)
