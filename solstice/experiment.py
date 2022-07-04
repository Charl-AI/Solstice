"""The `Experiment` is at the heart of Solstice. The API is similar to the
`pl.LightningModule` loved by
[PyTorch-Lightning](https://pytorch-lightning.readthedocs.io/en/latest/) users, but we
do less 'magic' to keep it as transparent as possible. If in doubt, just read the source
code - it's really short!"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Mapping, Tuple

import equinox as eqx

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

        # exp is just a pytree, so we can save and restore checkpoints like so...
        equinox.tree_serialise_leaves("checkpoint_0.eqx", exp)


        ```

    This class just specifies a recommended interface for experiment code. Experiments
    implementing this interface will automatically work with the Solstice training
    loops. You can always create or override methods as you wish and no methods are
    special-cased. For example it is common to define a `__call__` method to perform
    inference on a batch of data.
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
    def train_step(self, batch: Any) -> Tuple[Experiment, Any]:
        """A training step takes a batch of data and returns the updated experiment and
        any auxiliary outputs (usually a `solstice.Metrics` object).

        !!! tip
            You will typically want to use `jax.jit`, `jax.pmap`, `eqx.filter_jit`, or
            `eqx.filter_pmap` on this method. See the
            [solstice primer](https://charl-ai.github.io/Solstice/primer/)
            for more info on filtered transformations. You can also read the tutorial on
            different [parallelism strategies](https://charl-ai.github.io/Solstice/parallelism_strategies/).

        !!! example
            Pseudocode implementation of a training step:
            ```python
            class MNISTExperiment(Experiment):
                @eqx.filter_jit(kwargs=dict(batch=True))
                def train_step(self, batch: Tuple[np.ndarray, ...]
                ) -> Tuple[Experiment, solstice.Metrics]:

                imgs, labels = batch

                def loss_fn(params, x, y):
                    ... # compute loss
                    return loss, logits

                (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                    self.params, imgs, labels
                )

                new_params, new_opt_state = ... # calculate grads and update params
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
            batch (Any): Batch of data. Usually, this will be either a tuple of
                (input, target) arrays or a dictionary mapping keys to arrays.

        Returns:
            Tuple[Experiment, Any]: A new instance of the Experiment with the updated
                state and any auxiliary outputs, such as metrics.
        """
        raise NotImplementedError()

    @abstractmethod
    def eval_step(self, batch: Any) -> Tuple[Experiment, Any]:
        """An evaluation step (e.g. for validation or testing) takes a batch of data and
        returns the updated experiment and any auxiliary outputs. Usually, this will be
        a `solstice.Metrics` object. Like `train_step()`, you should probably JIT this
        method.

        !!! tip
            In most evaluation cases, the experiment returned will be unchanged,
            the main reason why you would want to modify it is to advance PRNG state.

        !!! example
            Pseudocode implementation of an evaluation step:
            ```python
            class MNISTExperiment(Experiment):
                @eqx.filter_jit(kwargs=dict(batch=True))
                def eval_step(self, batch: Tuple[np.ndarray, ...]
                ) -> Tuple[Experiment, Any]:
                imgs, labels = batch

                logits = ... # apply the model e.g. self.apply_fn(imgs)
                loss = ... # compute loss
                preds = jnp.argmax(logits, axis=-1)
                metrics = MyMetrics(preds, labels, loss)
                return self, metrics
            ```

        Args:
            batch (Any): Batch of data. Usually, this will be either a tuple of
                (input, target) arrays or a dictionary mapping keys to arrays.

        Returns:
            Tuple[Experiment, Any]: A new instance of the Experiment with the updated
                state and any auxiliary outputs, such as metrics.

        """
        raise NotImplementedError()
