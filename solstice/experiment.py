"""The `Experiment` is at the heart of Solstice. The API is similar to the one loved by
[PyTorch-Lightning](https://pytorch-lightning.readthedocs.io/en/latest/) users, but we
do less 'magic' to keep it as transparent as possible. If in doubt, just read the source
code - it's really short!"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Mapping, Protocol, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

from solstice.compat.optimizer import Optimizer
from solstice.metrics import ClassificationMetrics, Metrics

if TYPE_CHECKING:
    import numpy as np


class Experiment(eqx.Module, ABC):
    """Base class for Solstice experiments.

    An Experiment holds all stateful models, optimizers, etc... for a run and
    implements this interface. To make your own experiments, subclass this class and
    implement the logic for initialisation, training, and evaluating.

    !!! tip
        This is a subclass of `equinox.Module`, so you are free to use pure JAX
        transformations such as `jax.jit` and `jax.pmap`, as long as you remember to
        filter out static PyTree fields (e.g. with `eqx.filter_jit`).

    !!! example
        Pseudocode for typical `Experiment` usage:
        ```python

        exp = MyExperiment(...)  # initialise experiment state

        for step in range(num_steps):
            exp, outs = exp.train_step(batch)
            #do anything with the outputs here

        exp.save_checkpoint(...)  # save the trained state of the experiment

        ```

    This class just spefifies a recommended interface for experiment code. You can
    always create or override methods as you wish. For example it is common to define
    a `__call__` method to perform inference on a batch of data.
    """

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        """Initialise the experiment.
        !!! example
            Pseudocode implementation for initialising an MNIST classifier with flax
            and optax:
            ```python
            class MNISTExperiment(Experiment):
                params: Any
                opt_state: Any
                opt_apply: Callable
                model_apply: Callable
                num_classes: int

                def __init__(self, rng: int, model: flax.nn.Module,
                    optimizer = optax.GradientTransformation
                ) -> None:
                    key = jax.random.PRNGKey(rng)
                    dummy_batch = jnp.zeros((32, 784))
                    self.params = model.init(key, dummy_batch)
                    self.model_apply = model.apply
                    self.opt = optax.adam(learning_rate=1e-3)
                    self.opt_state = optimizer.init(self.params)
                    self.num_classes = 10
            ```
        """
        raise NotImplementedError()

    @abstractmethod
    def train_step(
        self, batch: Tuple[np.ndarray, ...] | Mapping[str, np.ndarray]
    ) -> Tuple[Experiment, Any]:
        """A training step takes a batch of data and returns the updated experiment and
        any auxiliary outputs (usually a `solstice.Metrics` object).

        !!! tip
            You will typically want to use `jax.jit` or `eqx.filter_jit` on this method.
            See the [solstice primer](https://charl-ai.github.io/Solstice/primer/)
            for more info on filtered transformations.

        !!! example
            Pseudocode implementation for training a MNIST classifier with flax and
            optax:
            ```python
            class MNISTExperiment(Experiment):
                @eqx.filter_jit(kwargs=dict(batch=True))
                def train_step(self, batch: Tuple[np.ndarray, ...]
                ) -> Tuple[Experiment, Any]:

                imgs, labels = batch

                def loss_fn(params, x, y):
                    logits = self.model_apply(params, x)
                    loss = jnp.mean(
                        optax.softmax_cross_entropy(logits, jax.nn.one_hot(y, self.num_classes))
                    )
                    return loss, logits

                (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                    self.params, imgs, labels
                )

                updates, new_opt_state = self.opt_apply(grads, self.opt_state, self.params)
                new_params = optax.apply_updates(self.params, updates)

                preds = jnp.argmax(logits, axis=-1)
                metrics = MyMetrics(preds, labels, loss)

                return (
                    solstice.replace(self, params=new_params, opt_state=new_opt_state),
                    metrics,
                )
            ```

        !!! tip
            You can use the `solstice.replace` function as a way of returning an
            experiment instance with modified state.

        Args:
            batch (Tuple[np.ndarray, ...] | Mapping[str, np.ndarray]): Batch of data.
                Usually, this will be either a tuple of (input, target) arrays or a
                dictionary, mapping keys to arrays.

        Returns:
            Tuple[Experiment, Any]: A new instance of the Experiment with the updated
                state and any auxiliary outputs, such as metrics.
        """
        raise NotImplementedError()

    @abstractmethod
    def eval_step(
        self, batch: Tuple[np.ndarray, ...] | Mapping[str, np.ndarray]
    ) -> Tuple[Experiment, Any]:
        """An evaluation step (e.g. for validation or testing) takes a batch of data and
        returns the updated experiment and any auxiliary outputs. Usually, this will be
        a `solstice.Metrics` object. Like `train_step()`, you should probably JIT this
        method.

        !!! tip
            In most evaluation cases, the experiment returned will be unchanged,
            the main reason why you would want to modify it is to advance PRNG state.

        !!! example
            Pseudocode implementation for evaluating a MNIST classifier with flax and
            optax:
            ```python
            class MNISTExperiment(Experiment):
                @eqx.filter_jit(kwargs=dict(batch=True))
                def eval_step(self, batch: Tuple[np.ndarray, ...]
                ) -> Tuple[Experiment, Any]:
                imgs, labels = batch

                logits = self.model_apply(self.params, imgs)
                loss = jnp.mean(
                    optax.softmax_cross_entropy(
                        logits, jax.nn.one_hot(labels, self.num_classes)
                    )
                )
                preds = jnp.argmax(logits, axis=-1)
                metrics = MyMetrics(preds, labels, loss)
                return self, metrics
            ```

        Args:
            batch (Tuple[np.ndarray, ...] | Mapping[str, np.ndarray]): Batch of data.
                Usually, this will be either a tuple of (input, target) arrays or a
                dictionary, mapping keys to arrays.

        Returns:
            Tuple[Experiment, Any]: A new instance of the Experiment with the updated
                state and any auxiliary outputs, such as metrics.

        """
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

    def __call__(self, x: Any, *, key: Any | None = None) -> Any:
        raise NotImplementedError()


# helper function, based on optax implementation, but in pure JAX to avoid the import
_softmax_cross_entropy = lambda logits, labels: -jnp.sum(
    labels * jax.nn.log_softmax(logits, axis=-1), axis=-1
)


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
                _softmax_cross_entropy(logits, jax.nn.one_hot(y, self._num_classes))
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
            _softmax_cross_entropy(logits, jax.nn.one_hot(y, self._num_classes))
        )
        preds = jnp.argmax(logits, axis=-1)
        metrics = ClassificationMetrics(preds, y, loss, self._num_classes)
        return metrics
