from __future__ import annotations

from abc import ABC, abstractmethod
import dataclasses
from typing import Any, Protocol, Tuple
import equinox as eqx
from optax import GradientTransformation
import jax
import jax.numpy as jnp
from optimizer import Optimizer
import optax

Metrics = Any


class Experiment(eqx.Module, ABC):
    """Abstract base class for experiments.

    An Experiment holds all necessary models, optimizers, etc... for a run and
    implements this interface. To make your own experiments, inherit from this class and
    implement the logic for initialisation, training, evaluating, and predicting.
    This is a subclass of `equinox.Module`, so you are free to use pure JAX
    transformations such as `jax.jit` and `jax.pmap`, as long as you remember to filter
    out static PyTree fields.

    Since Experiment is an  `equinox.Module`, it is actually a frozen dataclass,
    this means it is immutable, so training steps do not update parameters in 'self'
    but instead return a new instance of the Experiment with the params replaced. This
    is a common pattern in functional programming and JAX."""

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        """Constructor for initialising the experiment."""
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        """Perform inference."""
        raise NotImplementedError()

    @abstractmethod
    def train_step(self, batch, rng) -> Tuple[Metrics, Experiment]:
        """A training step takes a batch of data and returns the updated experiment and
        metrics over the batch."""
        raise NotImplementedError()

    @abstractmethod
    def eval_step(self, batch, rng) -> Metrics:
        """An evaluation step (e.g. for validation or testing) takes a batch of data
        and returns metrics over the batch."""
        raise NotImplementedError()


class ClassificationExperiment(Experiment):
    """Pre-made experiment class for basic classification tasks.

    For multiclass classification, softmax cross entropy is used as a loss function. In
    the binary class case, sigmoid binary cross entropy is used.
    """

    model: eqx.Module
    opt: Optimizer
    num_classes: int = eqx.static_field()

    def __init__(
        self, model: eqx.Module, optimizer: Optimizer, num_classes: int
    ) -> None:

        self.model = model  # not sure whether this should be already inited or not
        self.opt = optimizer
        self.num_classes = num_classes

        # add assertion about model output shape

    @eqx.filter_jit
    def train_step(self, batch):
        x, y = batch

        def loss_fn(model, x, y):
            logits = model(x)
            loss = jnp.mean(
                optax.softmax_cross_entropy(logits, jax.nn.one_hot(y, self.num_classes))
            )
            return loss, logits

        (loss, logits), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            self.model, x, y
        )

        updates, new_opt = self.opt(grads)
        new_model = eqx.apply_updates(self.model, updates)  # type: ignore

        metrics = None

        return metrics, dataclasses.replace(self, model=new_model, opt=new_opt)

    @eqx.filter_jit
    def eval_step(self, batch):
        x, y = batch
