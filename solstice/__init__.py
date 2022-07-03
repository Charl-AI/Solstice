"""Solstice, a library for creating and scaling experiments in JAX."""

from solstice.experiment import Experiment
from solstice.metrics import ClassificationMetrics, Metrics
from solstice.trainer import (
    train,
    test,
    Callback,
    CheckpointingCallback,
    EarlyStoppingCallback,
    ProfilingCallback,
)
from solstice.utils import replace

__all__ = (
    "Experiment",
    "Metrics",
    "ClassificationMetrics",
    "Callback",
    "CheckpointingCallback",
    "EarlyStoppingCallback",
    "ProfilingCallback",
    "train",
    "test",
    "replace",
)
