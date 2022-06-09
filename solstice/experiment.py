from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import Any, Mapping, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
from tqdm import tqdm

from optimizer import Optimizer
from solstice import ClassificationMetrics

Metrics = Any


class Experiment(eqx.Module, ABC):
    """Base class for Solstice experiments.

    An Experiment holds all stateful models, optimizers, etc... for a run and
    implements this interface. To make your own experiments, inherit from this class and
    implement the logic for initialisation, training, evaluating, and predicting.
    This is a subclass of `equinox.Module`, so you are free to use pure JAX
    transformations such as `jax.jit` and `jax.pmap`, as long as you remember to filter
    out static PyTree fields.

    We provide basic training and testing loops which should suit most use cases. If
    you want more control, you can override these methods or even ignore them entirely.

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
    def train_step(
        self, batch: Tuple[jnp.ndarray, ...] | Mapping[str, jnp.ndarray]
    ) -> Tuple[Metrics, Experiment]:
        """A training step takes a batch of data and returns the updated experiment and
        metrics over the batch."""
        raise NotImplementedError()

    @abstractmethod
    def eval_step(
        self, batch: Tuple[jnp.ndarray, ...] | Mapping[str, jnp.ndarray]
    ) -> Metrics:
        """An evaluation step (e.g. for validation or testing) takes a batch of data
        and returns metrics over the batch."""
        raise NotImplementedError()

    def save_checkpoint(self, checkpoint_dir: str, step: int) -> None:
        """Save the current state of the experiment to a checkpoint."""
        raise NotImplementedError()

    @classmethod
    def restore_checkpoint(cls, checkpoint_dir: str) -> Experiment:
        """Restore an experiment from a checkpoint."""
        raise NotImplementedError()

    @staticmethod
    def train(
        *,
        experiment: Experiment,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset | None = None,
        num_epochs: int,
        logger=None,
        log_every_n_steps: int = 100,
        ckpt_dir: str | None = None,
    ) -> Experiment:
        """Basic training loop for Solstice experiments. Includes optional logging and
        checkpointing. Notice that this is a staticmethod, so it does not update `self`,
        instead you pass in an instance of the experiment to be trained, e.g.

            - Initialise the experiment: `exp = Experiment(...)`
            - Train: `trained_exp = Experiment.train(exp, train_ds, ...)`

        Args:
            - experiment (Experiment): Solstice Experiment to train.
            - train_ds (tf.data.Dataset): Train dataset.
            - val_ds (tf.data.Dataset | None, optional): Validation dataset. Skips
                validation if None is given. Defaults to None.
            - num_epochs (int): Number of epochs to train for.
            - logger (_type_, optional): _description_. Defaults to None.
            - log_every_n_steps (int, optional): Metrics are accumulated over n steps,
                then computed and logged. Defaults to 100.
            - ckpt_dir (str | None, optional): Checkpoints are saved every epoch to this
                directory. If None, no checkpoints are saved. Defaults to None.

        Returns:
            - Experiment: New state of the experiment after training.
        """
        min_loader_len = (
            min(len(train_ds), len(val_ds)) if val_ds is not None else len(train_ds)
        )
        assert log_every_n_steps < min_loader_len or logger is None, (
            "log_every_n_steps must be less than the length of the shortest"
            f" dataloader, got {log_every_n_steps=} >= {min_loader_len=}"
        )

        for epoch in tqdm(range(num_epochs), desc="Training", unit="epoch"):
            for mode, ds in zip(["train", "val"], [train_ds, val_ds]):
                if ds is None:
                    continue
                global_step = epoch * len(ds)
                metrics = None
                for step, batch in enumerate(
                    tqdm(
                        ds.as_numpy_iterator(),
                        total=len(ds),
                        desc=f"{mode}",
                        leave=False,
                        unit="step",
                    )
                ):
                    if mode == "train":
                        batch_metrics, experiment = experiment.train_step(batch)
                    else:
                        batch_metrics = experiment.eval_step(batch)

                    metrics = metrics.merge(batch_metrics) if metrics else batch_metrics

                    if logger is not None and (step + 1) % log_every_n_steps == 0:
                        metrics_dict = metrics.compute()
                        logger.write_scalars(
                            global_step + step,
                            {f"{mode}/{key}": val for key, val in metrics_dict.items()},
                        )
                        metrics = None  # reset metrics object
                logger.flush() if logger is not None else None
            experiment.save_checkpoint(ckpt_dir, step=epoch) if ckpt_dir else None
        return experiment

    @staticmethod
    def test(
        experiment: Experiment, test_ds: tf.data.Dataset, logger=None
    ) -> Mapping[str, float]:
        """Basic testing loop for Solstice experiments. Includes optional logging.
        This is a  staticmethod, so pass in an instance of an Experiment to be tested.

        Args:
            - experiment (Experiment): Experiment to test.
            - test_ds (tf.data.Dataset): Test dataset.
            - logger (_type_, optional): _description_. Defaults to None.

        Returns:
            - Mapping[str, float]: Dictionary of scalar result metrics.
        """
        metrics = None

        for batch in tqdm(
            test_ds.as_numpy_iterator(), total=len(test_ds), desc="Testing", unit="step"
        ):
            batch_metrics = experiment.eval_step(batch)
            metrics = metrics.merge(batch_metrics) if metrics else batch_metrics

        assert metrics is not None  # type guard to check a metrics object was created
        metrics_dict = metrics.compute()

        if logger is not None:
            logger.write_scalars(
                0,
                {f"test/{key}": val for key, val in metrics_dict.items()},
            )
            logger.flush()

        return metrics_dict


class CallableModule(eqx.Module, ABC):
    """Equinox model which implements the `__call__` method. Just used internally for
    type hinting models."""

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        raise NotImplementedError()


class ClassificationExperiment(Experiment):
    """Pre-made experiment class for basic classification tasks. Performs multiclass
    classification with softmax cross entropy loss. You can use this class for binary
    classification if you treat it as a multi-class classification task with two
    classes and threshold set at 0.5.
    """

    model: CallableModule
    opt: Optimizer
    num_classes: int = eqx.static_field()

    def __init__(
        self, model: CallableModule, optimizer: Optimizer, num_classes: int
    ) -> None:
        """
        Args:
            - model (CallableModule): Model to train (instance of a callable
                `eqx.Module`), should return unnormalised logits shape
                (batch_size, num_classes) from its `__call__` method.
            - optimizer (Optimizer): Optimizer to use (instance of a Solstice
                optimizer).
            - num_classes (int): Number of classes in the dataset.
        """

        self.model = model
        self.opt = optimizer
        self.num_classes = num_classes

    @eqx.filter_jit
    def __call__(self, *args, **kwargs) -> Any:
        logits = self.model(*args, **kwargs)
        return jnp.argmax(logits, axis=-1)

    @eqx.filter_jit
    def train_step(
        self, batch: Tuple[jnp.ndarray, jnp.ndarray]
    ) -> Tuple[Metrics, Experiment]:
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

        preds = jnp.argmax(logits, axis=-1)
        metrics = ClassificationMetrics(preds, y, loss, self.num_classes)

        return metrics, dataclasses.replace(self, model=new_model, opt=new_opt)

    @eqx.filter_jit
    def eval_step(self, batch: Tuple[jnp.ndarray, jnp.ndarray]) -> Metrics:
        x, y = batch

        logits = self.model(x)
        loss = jnp.mean(
            optax.softmax_cross_entropy(logits, jax.nn.one_hot(y, self.num_classes))
        )
        preds = jnp.argmax(logits, axis=-1)
        metrics = ClassificationMetrics(preds, y, loss, self.num_classes)
        return metrics
