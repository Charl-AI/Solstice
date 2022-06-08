"""Solstice optimizers. Includes a basic interface for defining optimizers, a helper
class for wrapping Optax optimizers, and an example implementation of SGD."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import dataclasses
import equinox as eqx
import optax


class Optimizer(eqx.Module, ABC):
    """Base class for optimizers. Create a Solstice optimizer by subclassing and
    implementing this interface."""

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        """Constructor for initialising the optimizer."""
        raise NotImplementedError()

    @abstractmethod
    def __call__(
        self, grads: optax.Params, params: Optional[optax.Params] = None
    ) -> Tuple[optax.Updates, Optimizer]:
        """Calling the optimizer converts a PyTree of gradients to a tree of updates
        with the same shape, returning them and the updated optimizer state. The
        current params can optionally be passed in if the optimizer needs them."""
        raise NotImplementedError()


class OptaxOptimizer(Optimizer):
    """Wraps any Optax optimizer for use with Solstice."""

    update_fn: optax.TransformUpdateFn
    opt_state: optax.OptState

    def __init__(
        self, optimizer: optax.GradientTransformation, params: optax.Params
    ) -> None:
        """
        Args:
            optimizer (optax.GradientTransformation): Optax optimizer to wrap.
            params (optax.Params): Model params for initialising optimizer.
        """
        self.update_fn = optimizer.update
        self.opt_state = optimizer.init(params)

    def __call__(
        self, grads: optax.Params, params: Optional[optax.Params] = None
    ) -> Tuple[optax.Updates, OptaxOptimizer]:
        """
        Args:
            grads (optax.Params): PyTree of gradients.
            params (Optional[optax.Params], optional): PyTree of current parameters
            with same shape as grads. Only needed for optimizers which use the current
            params to determine the updates. Defaults to None.

        Returns:
            Tuple[optax.Updates, OptaxOptimizer]: Tuple of updates with the same shape
            as the grads and params, and an updated Optimizer object with the new state.
        """

        updates, new_state = self.update_fn(self.opt_state, grads, params)
        return updates, dataclasses.replace(self, opt_state=new_state)


class SGDOptimizer(Optimizer):
    """Solstice implementation of Stochastic Gradient Descent optimizer."""

    def __init__(self) -> None:
        ...

    def __call__(self, grads) -> Tuple[optax.Updates, SGDOptimizer]:
        ...
