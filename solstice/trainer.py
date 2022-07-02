"""Training loops are usually boilerplate code that has little to do with your research.
We provide a training loop which integrates with a simple and flexible callback system.
Any `solstice.Experiment` can be passed to the trainer, but you can always write your
own training loops if necessary. We provide a handful of pre-implemented callbacks, but
leave logging callbacks to the user to implement because there are many different
libraries that can be used for logging which we do not want Solstice to depend on (e.g.
wandb, TensorBoard(X), CLU, ...)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, TypeVar

import jax
from tqdm import tqdm

from solstice.experiment import Experiment

if TYPE_CHECKING:
    import tensorflow as tf


class Callback(ABC):
    """Base class for callbacks to `solstice.train()` and `solstice.test(). Subclass
    and implement this interface to inject arbitrary functionality into the training
    and testing loops.

    !!! tip
        All callback hooks return `None`, so they cannot affect the training itself.
        Use callbacks to execute side effects like logging, checkpointing or profiling.

    !!! example
        Pseudocode callback implementation for logging with `solstice.Metrics`:
        ```python

        class LoggingCallback(Callback):
            def __init__(self, log_every_n_steps, ...):
                self.metrics = None
                self.log_every_n_steps = log_every_n_steps
                ... # set up logging, e.g. wandb.init(...)

            def on_step_end(self, exp, global_step, training, batch, outs):
                assert isinstance(outs, solstice.Metrics)
                self.metrics = outs.merge(self.metrics) if self.metrics else outs
                if (global_step + 1) % self.log_every_n_steps == 0:
                    metrics_dict = self.metrics.compute()
                    ... # do logging e.g. wandb.log(metrics_dict)
                    self.metrics = None
        ```
    """

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        """Initialize the callback."""
        raise NotImplementedError

    def on_epoch_start(self, exp: Experiment, epoch: int) -> None:
        """Called at the start of each epoch, i.e. before the model has seen any data
        for that epoch.

        Args:
            exp (Experiment): Current Experiment state.
            epoch (int): Current epoch number.
        """
        pass

    def on_epoch_end(self, exp: Experiment, epoch: int) -> None:
        """Called at the end of each epoch, i.e. after the model has seen the full
        training and validation sets for that epoch.

        Args:
            exp (Experiment): Current Experiment state.
            epoch (int): Current epoch number.
        """
        pass

    def on_step_start(
        self, exp: Experiment, global_step: int, training: bool, batch
    ) -> None:
        """Called at the start of each training and validation step, i.e. before the
        batch has been seen.

        Args:
            exp (Experiment): Current Experiment state.
            global_step (int): Current step number. This is the global step, i.e. the
                total number of training or validation or testing steps seen so far.
                Note that wekeep separate step counts for training and validation, so
                it might not be unique.
            training (bool): Whether this is a training or evaluation step. In `test()`
                loop, this is always False.
            batch (_type_): Current batch of data for this step.
        """
        pass

    def on_step_end(
        self, exp: Experiment, global_step: int, training: bool, batch, outs: Any
    ) -> None:
        """Called at the end of each training and validation step, i.e. after the batch
        has been seen.

        Args:
            exp (Experiment): Current Experiment state.
            global_step (int): Current step number. This is the global step, i.e. the
                total number of training or validation or testing steps seen so far.
                Note that wekeep separate step counts for training and validation, so
                it might not be unique.
            training (bool): Whether this is a training or evaluation step. In `test()`
                loop, this is always False.
            batch (_type_): Current batch of data for this step.
            outs (Any): Auxiliary outputs from the experiment train/eval step. Usually,
                this should be a `solstice.Metrics` object.
        """
        pass


class CheckpointingCallback(Callback):
    """Checkpoint the experiment state at the end of each epoch.

    !!! todo
        Implement this. Consider adding asynchronous checkpointing."""

    pass


class ProfilingCallback(Callback):
    """Use the built-in JAX (TensorBoard) profiler to profile training and evaluation
    steps.

    !!! note
        To view the traces, ensure TensorBoard is installed. Then run
        `tensorboard --logdir=<log_dir>`. See
        https://jax.readthedocs.io/en/latest/profiling.html for more information."""

    def __init__(self, log_dir: str, steps_to_profile: list[int] | None = None) -> None:
        """Initialize the Profiler callback.

        !!! tip
            You can use the `steps_to_profile` argument to profile only a subset of the
            steps. Usually, step 0 will be slowest due to JIT compilation, so you might
            want to profile steps 0 and 1.

        Args:
            log_dir (str): Directory to write the profiler trace files to.
            steps_to_profile (list[int] | None, optional): If given, only profile these
                steps, else profile every step. Defaults to None.
        """
        self.log_dir = log_dir
        self.steps_to_profile = steps_to_profile

    def on_step_start(
        self, exp: Experiment, global_step: int, training: bool, batch
    ) -> None:
        del exp, training, batch
        if self.steps_to_profile is None or global_step in self.steps_to_profile:
            jax.profiler.start_trace(self.log_dir)

    def on_step_end(
        self, exp: Experiment, global_step: int, training: bool, batch, outs: Any
    ) -> None:
        del exp, training, batch, outs
        if self.steps_to_profile is None or global_step in self.steps_to_profile:
            jax.profiler.stop_trace()


# Currently not exposed in the public API. Use the EarlyStoppingCallback to raise this.
class _EarlyStoppingException(Exception):
    """A callback can raise this exception `on_epoch_end` to break the training loop
    early."""

    pass


class EarlyStoppingCallback(Callback):
    """Stop training early if a criterion is met. Checks once per epoch (at the end).
    Internally, this accumulates the auxiliary outputs from each validation step into a
    list and passes it to the criterion function."""

    def __init__(
        self,
        criterion_fn: Callable[[list[Any]], bool],
        accumulate_every_n_steps: int | None,
    ) -> None:
        """Initialize the EarlyStoppingCallback.

        !!! tip
            It may not be desirable to accumulate all the validation step outputs due
            to memory constraints (especially if you are outputting images etc.). You
            can use the `accumulate_every_n_steps` argument to only pass a subset of
            the validation step outputs to the criterion function.

        Args:
            criterion_fn (Callable[[list[Any]], bool]): Function that takes a list of
                auxiliary outputs from each step and returns a boolean indicating
                whether to stop training.
            accumulate_every_n_steps (int | None, optional): If given, only accumulate
                auxiliary outputs every nth step. I.e. set to 2 to only keep half, 3 for
                keeping 1/3, etc. If None is given, accumulate all outputs.
                Defaults to None.
        """
        self.outs = []
        self.criterion_fn = criterion_fn
        self.accumulate_every_n_steps = accumulate_every_n_steps

    def on_step_end(
        self, exp: Experiment, global_step: int, training: bool, batch, outs: Any
    ) -> None:
        del exp, global_step, batch

        if not training:
            self.outs.append(outs)

    def on_epoch_end(self, exp: Experiment, epoch: int) -> None:
        del exp, epoch
        if self.criterion_fn(self.outs):
            raise _EarlyStoppingException()
        self.outs = []  # reset for next epoch


# type variable for experiment, this is needed because we want the train loop to
# accept an Experiment subclass and return the *same* type of Experiment subclass.
ExperimentType = TypeVar("ExperimentType", bound=Experiment)


def train(
    exp: ExperimentType,
    num_epochs: int,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset | None = None,
    callbacks: list[Callback] | None = None,
) -> ExperimentType:
    """Train a `solstice.Experiment`, using `tf.data.Dataset` for data loading.
    Supply `solstice.Callback`s to add any additional functionality.

    Args:
        exp (Experiment): Solstice experiment to train.
        num_epochs (int): Number of epochs to train for.
        train_ds (tf.data.Dataset): TensorFlow dataset of training data.
        val_ds (tf.data.Dataset | None, optional): TensorFlow dataset of validation
            data. If none is given, validation is skipped. Defaults to None.
        callbacks (list[Callback] | None, optional): List of Solstice callbacks. These
            can execute arbitrary code on certain events, usually for side effects like
            logging and checkpointing. See `solstice.Callback`. Defaults to None.

    Returns:
        Experiment: Trained experiment.
    """

    for epoch in tqdm(range(num_epochs), desc="Training", unit="epoch"):
        assert isinstance(epoch, int)  # just for mypy for type narrowing
        [
            cb.on_epoch_start(exp, epoch) for cb in callbacks
        ] if callbacks is not None else None

        for training, ds in zip([True, False], [train_ds, val_ds]):
            if ds is None:
                continue

            global_step = epoch * len(ds)  # nb: separate step counts for train and val
            for batch in tqdm(
                ds.as_numpy_iterator(),
                total=len(ds),
                desc=f"{'Training' if training else 'Validation'}",
                leave=False,
                unit="step",
            ):
                global_step += 1

                [
                    cb.on_step_start(exp, global_step, training, batch)
                    for cb in callbacks
                ] if callbacks is not None else None

                exp, outs = exp.train_step(batch) if training else exp.eval_step(batch)

                [
                    cb.on_step_end(exp, global_step, training, batch, outs)
                    for cb in callbacks
                ] if callbacks is not None else None

        try:
            [
                cb.on_epoch_end(exp, epoch) for cb in callbacks
            ] if callbacks is not None else None
        except _EarlyStoppingException:
            print(f"Early stopping at epoch {epoch}")
            return exp
    return exp


def test(
    exp: Experiment,
    test_ds: tf.data.Dataset,
    callbacks: list[Callback] | None = None,
    return_outs: bool = False,
) -> list[Any] | None:
    """Test a `solstice.Experiment`, using `tf.data.Dataset` for data loading. Supply
    `solstice.Callback`s to add any additional functionality.

    Args:
        exp (Experiment): Experiment to test.
        test_ds (tf.data.Dataset): TensorFlow dataset of test data.
        callbacks (list[Callback] | None, optional): List of Solstice callbacks. These
            can execute arbitrary code on certain events, usually for side effects like
            logging. See `solstice.Callback`. Defaults to None.
        return_outs (bool, optional): If True, the auxiliary outputs from
            `exp.eval_step()` are accumulated into a list and returned, else this
            function returns nothing. Defaults to False.

    !!! tip
        Testing simply involves running through the test_ds for a single epoch. Thus
        the `on_epoch_start()` and `on_epoch_end()` callback hooks are executed once
        each, before testing starts and after testing ends.

    Returns:
        list[Any] | None: List of auxiliary outputs from `exp.eval_step()` if
            return_outs is True, else None.
    """

    [cb.on_epoch_start(exp, 0) for cb in callbacks] if callbacks is not None else None

    global_step = 0
    outputs_list = []

    for batch in tqdm(
        test_ds.as_numpy_iterator(), total=len(test_ds), desc="Testing", unit="step"
    ):
        global_step += 1

        [
            cb.on_step_start(exp, global_step, False, batch) for cb in callbacks
        ] if callbacks is not None else None

        exp, outs = exp.eval_step(batch)
        outputs_list.append(outs) if return_outs else None

        [
            cb.on_step_end(exp, global_step, False, batch, outs) for cb in callbacks
        ] if callbacks is not None else None

    [cb.on_epoch_end(exp, 0) for cb in callbacks] if callbacks is not None else None

    return outputs_list if return_outs else None
