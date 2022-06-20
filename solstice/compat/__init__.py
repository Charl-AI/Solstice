"""solstice.compat is a set of optional compatibility layers for giving common
interfaces to popular JAX libraries. This allows you to write training code in
Solstice that is not dependent on any neural network/optimisation/dataset libraries."""

from solstice.compat.optimizer import Optimizer, OptaxOptimizer, SGDOptimizer

__all__ = ["Optimizer", "OptaxOptimizer", "SGDOptimizer"]
