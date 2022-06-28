"""Solstice, a library for creating and scaling experiments in JAX."""

from solstice.experiment import ClassificationExperiment, Experiment
from solstice.metrics import ClassificationMetrics, Metrics
from solstice.utils import replace

__all__ = (
    "Experiment",
    "ClassificationExperiment",
    "Metrics",
    "ClassificationMetrics",
    "replace",
)

from solstice import compat as compat
