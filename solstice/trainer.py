"""Training loops are usually boilerplate code that has little to do with your research.
We provide training and testing loops which integrate with a simple and flexible
callback system. Any `solstice.Experiment` can be passed to the loops, but you can
always write your own if necessary. We provide a handful of pre-implemented callbacks,
but if they do not suit your needs, you can use them as inspiration to write your own.
"""

from __future__ import annotations

import functools
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, TypeVar

import jax
from tqdm import tqdm
from typing_extensions import TypeGuard, Literal

from solstice.experiment import Experiment
from solstice.metrics import Metrics
from solstice.utils import EarlyStoppingException

if TYPE_CHECKING:
    import tensorflow as tf

# use "solstice" logger for all output logging
logger = logging.getLogger("solstice")


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

        class MyLoggingCallback(Callback):
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

    def on_epoch_start(
        self, exp: Experiment, epoch: int, mode: Literal["train", "val", "test"]
    ) -> None:
        """Called at the start of each epoch, i.e. before the model has seen any data
        for that epoch.

        Args:
            exp (Experiment): Current Experiment state.
            epoch (int): Current epoch number.
            mode (Literal["train", "val", "test"]): String representing whether this is
                a training, validation or testing epoch.
        """
        pass

    def on_epoch_end(
        self, exp: Experiment, epoch: int, mode: Literal["train", "val", "test"]
    ) -> None:
        """Called at the end of each epoch, i.e. after the model has seen the full
        dataset for that epoch.

        Args:
            exp (Experiment): Current Experiment state.
            epoch (int): Current epoch number.
            mode (Literal["train", "val", "test"]): String representing whether this is
                a training, validation or testing step.
        """
        pass

    def on_step_start(
        self,
        exp: Experiment,
        global_step: int,
        mode: Literal["train", "val", "test"],
        batch: Any,
    ) -> None:
        """Called at the start of each training and validation step, i.e. before the
        batch has been seen.

        Args:
            exp (Experiment): Current Experiment state.
            global_step (int): Current step number. This is the global step, i.e. the
                total number of training or validation or testing steps seen so far.
                Note that we keep separate step counts for training and validation, so
                it might not be unique.
            mode (Literal["train", "val", "test"]): String representing whether this is
                a training, validation or testing step.
            batch (Any): Current batch of data for this step.
        """
        pass

    def on_step_end(
        self,
        outs: Any,
        exp: Experiment,
        global_step: int,
        mode: Literal["train", "val", "test"],
        batch: Any,
    ) -> None:
        """Called at the end of each training and validation step, i.e. after the batch
        has been seen.

        Args:
            exp (Experiment): Current Experiment state.
            global_step (int): Current step number. This is the global step, i.e. the
                total number of training or validation or testing steps seen so far.
                Note that we keep separate step counts for training and validation, so
                it might not be unique.
            mode (Literal["train", "val", "test"]): String representing whether this is
                a training, validation or testing step.
            batch (Any): Current batch of data for this step.
            outs (Any): Auxiliary outputs from the experiment train/eval step. Usually,
                this should be a `solstice.Metrics` object.
        """
        pass


class LoggingCallback(Callback):
    """Logs auxiliary outputs from training or evaulation steps (either periodically
    every n steps, or at the end of the epoch). Internally, this accumulates metrics
    with `metrics.merge()`, computes them with `metrics.compute()`, and then passes
    the final results to the given logging function.

    !!! warning
        Auxiliary outputs from the train and eval steps must be a `solstice.Metrics`
        instance for this callback to work properly. We raise an AssertionError if this
        is not the case.

    !!! note
        There are many different libraries you can use for writing logs (e.g. wandb,
        TensorBoard(X), ...). We offer no opinion on which one you should use. Pass in
        a logging function to use any arbitrary logger.
    """

    def __init__(
        self,
        log_every_n_steps: int | None = None,
        logging_fn: Callable[[Any, int, Literal["train", "val", "test"]], None]
        | None = None,
    ) -> None:
        """Initialize the logging callback.

        Args:
            log_every_n_steps (int | None, optional): If given, accumulate metrics over
                n steps before logging. If None, log at end of epoch. Defaults to None.
            logging_fn (Callable[[Any, int, Literal['train', 'val', 'test']], None] | None, optional):
                Logging function. Takes the outputs of `metrics.compute()`, the current
                step or epoch number, and a string representing whether training,
                validating, or testing. The function should return nothing. If no
                logging_fn is given, the default behaviour is to log with the built in
                Python logger (INFO level). Defaults to None.

        !!! example
            The default logging function (used if None is given) logs using the built
            in Python logger, with name "solstice" and INFO level
            (notice that the output of `metrics.compute()` must be printable):
            ```python
            logger = logging.getLogger("solstice")

            default_logger = lambda metrics, step, mode: logging.info(
                f"{mode} step {step}: {metrics}"
            )
            ```

            If the logs aren't showing, you might need to put this line at the top of
            your script:
            ```python
            import logging
            logging.getLogger("solstice").setLevel(logging.INFO)
            ```
        """
        default_logger = lambda metrics, step, mode: logger.info(
            f"{mode} step {step}: {metrics}"
        )
        self.logging_fn = logging_fn if logging_fn else default_logger
        self.log_every_n_steps = log_every_n_steps
        self.metrics = None

    def on_step_end(
        self,
        outs: Any,
        exp: Experiment,
        global_step: int,
        mode: Literal["train", "val", "test"],
        batch: Any,
    ) -> None:
        del exp, batch
        assert isinstance(outs, Metrics)
        self.metrics = outs.merge(self.metrics) if self.metrics else outs

        if self.log_every_n_steps and (global_step + 1) % self.log_every_n_steps == 0:
            final_metrics = self.metrics.compute()
            self.logging_fn(final_metrics, global_step, mode)
            self.metrics = None

    def on_epoch_end(
        self, exp: Experiment, epoch: int, mode: Literal["train", "val", "test"]
    ) -> None:
        del exp
        # if not logging every n steps, we just log at the end of the epoch
        if not self.log_every_n_steps:
            assert self.metrics is not None
            final_metrics = self.metrics.compute()
            self.logging_fn(final_metrics, epoch, mode)
        # reset the metrics object to prevent train/val metrics from being mixed
        self.metrics = None


class CheckpointingCallback(Callback):
    """Checkpoint the experiment state at the end of each epoch.

    !!! todo
        Implement this. Consider adding asynchronous checkpointing."""

    pass


class ProfilingCallback(Callback):
    """Uses the built-in JAX (TensorBoard) profiler to profile training and evaluation
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
        self,
        exp: Experiment,
        global_step: int,
        mode: Literal["train", "val", "test"],
        batch,
    ) -> None:
        del exp, mode, batch
        if self.steps_to_profile is None or global_step in self.steps_to_profile:
            jax.profiler.start_trace(self.log_dir)

    def on_step_end(
        self,
        outs: Any,
        exp: Experiment,
        global_step: int,
        mode: Literal["train", "val", "test"],
        batch,
    ) -> None:
        del exp, mode, batch, outs
        if self.steps_to_profile is None or global_step in self.steps_to_profile:
            jax.profiler.stop_trace()


class EarlyStoppingCallback(Callback):
    """Stops training early if a criterion is met. Checks once per validation epoch
    (at the end). This callback accumulates auxiliary outputs from each validation step
    into a list and passes them to the criterion function which determines whether to
    stop training.

    !!! tip
        If this callback doesn't suit your needs, you can implement your own early
        stopping callback by raising an `EarlyStoppingException` in the `on_step_end`
        hook.
    """

    def __init__(
        self,
        criterion_fn: Callable[[list[Any]], bool],
        accumulate_every_n_steps: int = 1,
    ) -> None:
        """Initialize the EarlyStoppingCallback.

        Args:
            criterion_fn (Callable[[list[Any]], bool]): Function that takes a list of
                the accumulated auxiliary outputs from each step and returns a boolean
                indicating whether to stop training.
            accumulate_every_n_steps (int, optional): Accumulate auxiliary outputs every
                nth step. Set to 2 to only keep half, 3 for keeping 1/3, etc. This
                effectively downsamples the signal (so beware it is losing information).
                Defaults to 1.

        !!! example
            Example criterion function takes the final metrics object, calls .compute()
            on it to return a dictionary, and stops training if accuracy is > 0.9:
            TODO: update example when `solstice.reduce`  is implemented
            ```python
            criterion fn = lambda metrics: metrics.compute()["accuracy"] > 0.9
            ```
        """
        self.accumulated_outs = []
        self.criterion_fn = criterion_fn
        self.accumulate_every_n_steps = accumulate_every_n_steps

    def on_step_end(
        self,
        outs: Any,
        exp: Experiment,
        global_step: int,
        mode: Literal["train", "val", "test"],
        batch,
    ) -> None:
        del exp, batch

        if mode == "val" and global_step % self.accumulate_every_n_steps == 0:
            self.accumulated_outs.append(outs)

    def on_epoch_end(
        self, exp: Experiment, epoch: int, mode: Literal["train", "val", "test"]
    ) -> None:
        del exp, epoch
        if mode == "val":
            if self.criterion_fn(self.accumulated_outs):
                raise EarlyStoppingException()
        self.accumulated_outs = []  # reset for next epoch


# type variable for experiment, this is needed because we want the train loop to
# accept an Experiment subclass and return the *same* type of Experiment subclass.
ExperimentType = TypeVar("ExperimentType", bound=Experiment)


# Type guard for mypy type narrowing
def _is_valid_mode(mode: str) -> TypeGuard[Literal["train", "val", "test"]]:
    return mode in ["train", "val", "test"]


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

        for mode, ds in zip(["train", "val"], [train_ds, val_ds]):
            assert _is_valid_mode(mode)  # type narrowing
            if ds is None:
                continue

            [
                cb.on_epoch_start(exp, epoch, mode) for cb in callbacks
            ] if callbacks is not None else None

            global_step = epoch * len(ds)  # nb: separate step counts for train and val
            for batch in tqdm(
                ds.as_numpy_iterator(),
                total=len(ds),
                desc=f"{mode}",
                leave=False,
                unit="step",
            ):
                global_step += 1

                [
                    cb.on_step_start(exp, global_step, mode, batch) for cb in callbacks
                ] if callbacks is not None else None

                exp, outs = (
                    exp.train_step(batch) if mode == "train" else exp.eval_step(batch)
                )

                [
                    cb.on_step_end(outs, exp, global_step, mode, batch)
                    for cb in callbacks
                ] if callbacks is not None else None

            try:
                [
                    cb.on_epoch_end(exp, epoch, mode) for cb in callbacks
                ] if callbacks is not None else None
            except EarlyStoppingException:
                logging.info(f"Early stopping at epoch {epoch}")
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
    assert callbacks is not None or return_outs is True, (
        "No callbacks were provided and return_outs is False. This function thus has no"
        " return vaules or side effects. All it does is heat up the planet :("
    )

    mode: Literal["test"] = "test"
    [
        cb.on_epoch_start(exp, 0, mode) for cb in callbacks
    ] if callbacks is not None else None

    global_step = 0
    outputs_list = []

    for batch in tqdm(
        test_ds.as_numpy_iterator(), total=len(test_ds), desc="Testing", unit="step"
    ):
        global_step += 1

        [
            cb.on_step_start(exp, global_step, mode, batch) for cb in callbacks
        ] if callbacks is not None else None

        exp, outs = exp.eval_step(batch)
        outputs_list.append(outs) if return_outs else None

        [
            cb.on_step_end(outs, exp, global_step, mode, batch) for cb in callbacks
        ] if callbacks is not None else None

    [
        cb.on_epoch_end(exp, 0, mode) for cb in callbacks
    ] if callbacks is not None else None

    return outputs_list if return_outs else None
