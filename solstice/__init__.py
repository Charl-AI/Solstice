"""Solstice, a library for creating and scaling experiments in JAX."""

from solstice.experiment import ClassificationExperiment, Experiment
from solstice.logger import Logger, TensorBoardLogger, WandbLogger
from solstice.metrics import ClassificationMetrics, Metrics
from solstice import compat as compat

__all__ = (
    "ClassificationExperiment",
    "ClassificationMetrics",
    "Experiment",
    "Logger",
    "Metrics",
    "TensorBoardLogger",
    "WandbLogger",
)
