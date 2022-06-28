"""Solstice.compat is a set of optional compatibility layers for giving common
interfaces to popular JAX libraries. This allows you to write training code in
Pure JAX+Equinox without depending on neural network or optimization libraries."""

from solstice.compat.optimizer import Optimizer, OptaxOptimizer, SGDOptimizer

__all__ = ["Optimizer", "OptaxOptimizer", "SGDOptimizer"]
