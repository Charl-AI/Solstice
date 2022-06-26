"""The `Experiment` is at the heart of Solstice. The API is similar to the one loved by
[PyTorch-Lightning](https://pytorch-lightning.readthedocs.io/en/latest/) users, but we
do less 'magic' to keep it as transparent as possible. If in doubt, just read the source
code - it's really short!"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import Any, Mapping, Protocol, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
from tqdm import tqdm
import chex
from solstice.compat.optimizer import Optimizer
from solstice.metrics import ClassificationMetrics, Metrics


class Experiment(eqx.Module, ABC):
    """Base class for Solstice experiments.

    An Experiment holds all stateful models, optimizers, etc... for a run and
    implements this interface. To make your own experiments, subclass this class and
    implement the logic for initialisation, training, evaluating, and predicting.

    !!! tip
        This is a subclass of `equinox.Module`, so you are free to use pure JAX
        transformations such as `jax.jit` and `jax.pmap`, as long as you remember to
        filter out static PyTree fields (e.g. with `eqx.filter_jit`).

    !!! example
        Pseudo code for typical `Experiment` usage:
        ```python

        exp = MyExperiment(...)  # initialise experiment state

        for step in range(num_steps):
            exp, outs = exp.train_step(batch)
            #do anything with the outputs here

        exp.save_checkpoint(...)  # save the trained state of the experiment

        ```

    Since Experiment is an  `equinox.Module`, it is actually a frozen dataclass,
    this means it is immutable, so training steps do not update parameters in 'self'
    but instead return a new instance of the Experiment with the params replaced. This
    is a common pattern in functional programming and JAX. Solstice includes the
    `solstice.replace` utility, which you will find useful for this pattern.
    """

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        """Initialising the experiment. This will likely involve creating models,
        optimizers, etc..."""
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        """Perform inference."""
        raise NotImplementedError()

    @abstractmethod
    def train_step(
        self, batch: Tuple[jnp.ndarray, ...] | Mapping[str, jnp.ndarray]
    ) -> Tuple[Experiment, Any]:
        """A training step takes a batch of data and returns the updated experiment and
        any auxiliary outputs. Usually, this will be a `solstice.Metrics` object."""
        raise NotImplementedError()

    @abstractmethod
    def eval_step(
        self, batch: Tuple[jnp.ndarray, ...] | Mapping[str, jnp.ndarray]
    ) -> Tuple[Experiment, Any]:
        """An evaluation step (e.g. for validation or testing) takes a batch of data and
        returns the updated experiment and any auxiliary outputs. Usually, this will be
        a `solstice.Metrics` object. In most evaluation cases, the experiment returned
        will be unchanged, but in some cases, you may want to update the PRNG etc..."""
        raise NotImplementedError()

    def save_checkpoint(self, checkpoint_dir: str, step: int) -> None:
        """Save the current state of the experiment to a checkpoint."""
        raise NotImplementedError()

    def restore_checkpoint(self, checkpoint_dir: str) -> Experiment:
        """Restore an experiment from a checkpoint."""
        raise NotImplementedError()


class CallablePyTree(Protocol):
    """A callable PyTree is a JAX PyTree which implements the `__call__` method,
    accepting an input array and optional PRNGKey. Just used internally for
    type-hinting models. All models in `equinox.nn` follow this signature."""

    def __call__(self, x: Any, *, key: chex.PRNGKey | None = None) -> Any:
        raise NotImplementedError()


class ClassificationExperiment(Experiment):
    """Pre-made experiment class for basic classification tasks. Performs multiclass
    classification with softmax cross entropy loss. You can use this class for binary
    classification if you treat it as a multi-class classification task with two
    classes and threshold set at 0.5.
    """

    _model: CallablePyTree
    _opt: Optimizer
    _num_classes: int

    def __init__(
        self, model: CallablePyTree, optimizer: Optimizer, num_classes: int
    ) -> None:
        """
        Args:
            model (CallablePyTree): Model to train. Must take an (unbatched) input
                array and optional PRNGKey as inputs to it's `__call__` method,
                returning unbatched, unnormalized vector of logits, shape
                (num_classes,). An example of a model like this is `equinox.nn.MLP`.
            optimizer (Optimizer): Solstice Optimizer to use.
            num_classes (int): Number of classes in the dataset.
        """

        self._model = model
        self._opt = optimizer
        self._num_classes = num_classes

    @eqx.filter_jit
    def __call__(self, *args, **kwargs) -> Any:
        logits = self._model(*args, **kwargs)
        return jnp.argmax(logits, axis=-1)

    @eqx.filter_jit
    def train_step(
        self, batch: Tuple[jnp.ndarray, jnp.ndarray]
    ) -> Tuple[Metrics, Experiment]:
        x, y = batch

        def loss_fn(model, x, y):
            logits = model(x)
            loss = jnp.mean(
                optax.softmax_cross_entropy(
                    logits, jax.nn.one_hot(y, self._num_classes)
                )
            )
            return loss, logits

        (loss, logits), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            self._model, x, y
        )

        updates, new_opt = self._opt(grads)
        new_model = eqx.apply_updates(self._model, updates)  # type: ignore

        preds = jnp.argmax(logits, axis=-1)
        metrics = ClassificationMetrics(preds, y, loss, self._num_classes)

        return metrics, dataclasses.replace(self, model=new_model, opt=new_opt)

    @eqx.filter_jit
    def eval_step(self, batch: Tuple[jnp.ndarray, jnp.ndarray]) -> Metrics:
        x, y = batch

        logits = self._model(x)
        loss = jnp.mean(
            optax.softmax_cross_entropy(logits, jax.nn.one_hot(y, self._num_classes))
        )
        preds = jnp.argmax(logits, axis=-1)
        metrics = ClassificationMetrics(preds, y, loss, self._num_classes)
        return metrics
