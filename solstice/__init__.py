"""Solstice, a library for scaling experiments with Equinox."""

from experiment import ClassificationExperiment, Experiment
from logger import Logger, TensorBoardLogger, WandbLogger
from metrics import ClassificationMetrics, Metrics
from optimizer import OptaxOptimizer, Optimizer, SGDOptimizer

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
