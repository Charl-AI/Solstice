"""Solstice, a library for scaling experiments with Equinox."""

from solstice.experiment import ClassificationExperiment, Experiment
from solstice.logger import Logger, TensorBoardLogger, WandbLogger
from solstice.metrics import ClassificationMetrics, Metrics
from solstice.optimizer import OptaxOptimizer, Optimizer, SGDOptimizer

__all__ = (
    "ClassificationExperiment",
    "ClassificationMetrics",
    "Experiment",
    "Logger",
    "Metrics",
    "OptaxOptimizer",
    "Optimizer",
    "SGDOptimizer",
    "TensorBoardLogger",
    "WandbLogger",
)
