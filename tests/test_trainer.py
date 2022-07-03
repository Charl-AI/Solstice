from typing import Tuple

import pytest
import solstice
import tensorflow_datasets as tfds
from tensorflow_datasets import testing as tfds_testing


class _DummyCounterExperiment(solstice.Experiment):
    """Dummy experiment used for testing the training loops. Simply implements a counter
    which increments every time it sees a training batch."""

    count: int

    def __init__(self, initial_count: int) -> None:
        self.count = initial_count

    def train_step(self, batch) -> Tuple[solstice.Experiment, int]:
        del batch
        new_count = self.count + 1
        return solstice.replace(self, count=new_count), new_count

    def eval_step(self, batch) -> Tuple[solstice.Experiment, int]:
        del batch
        return self, self.count


def test_training_loop():
    exp = _DummyCounterExperiment(0)
    num_epochs = 2
    num_examples = 50
    batch_size = 5

    with tfds_testing.mock_data(num_examples=num_examples):
        train_ds = tfds.load("mnist", split="train")
        preprocess_data = lambda ds: ds.batch(batch_size).prefetch(1)
        train_ds = preprocess_data(train_ds)

        exp = solstice.train(exp, num_epochs, train_ds, None)  # test w/o val_ds

        # the counter has trained properly if it is showing the number of batches seen
        assert exp.count == num_epochs * num_examples // batch_size


def test_testing_loop():
    exp = _DummyCounterExperiment(1)
    num_examples = 50
    batch_size = 5

    with tfds_testing.mock_data(num_examples=num_examples):
        test_ds = tfds.load("mnist", split="test")
        preprocess_data = lambda ds: ds.batch(batch_size).prefetch(1)
        test_ds = preprocess_data(test_ds)

        outs = solstice.test(exp, test_ds, None, return_outs=True)  # test w/o val_ds
        # the test loop returns a list of auxiliary outputs, in this case 1s
        assert outs == [1 for _ in range(num_examples // batch_size)]

        # ensure the test loop raises an error if no callbacks or returns
        with pytest.raises(AssertionError):
            _ = solstice.test(exp, test_ds, None, return_outs=False)


def test_callbacks():
    """Test all Solstice callbacks in one go."""
    exp = _DummyCounterExperiment(0)
    num_epochs = 2
    num_examples = 50
    batch_size = 5

    count_after_one_epoch = 1 * num_examples // batch_size

    prof_callback = solstice.ProfilingCallback(
        log_dir="/tmp/solstice_test/profiles", steps_to_profile=[0, 1, 2]
    )
    es_criterion = lambda counts: (sum(counts) / len(counts)) >= count_after_one_epoch
    early_stop_callback = solstice.EarlyStoppingCallback(es_criterion, 2)

    with tfds_testing.mock_data(num_examples=num_examples):
        train_ds = tfds.load("mnist", split="train")
        preprocess_data = lambda ds: ds.batch(batch_size).prefetch(1)
        train_ds = preprocess_data(train_ds)
        val_ds = tfds.load("mnist", split="test")
        val_ds = preprocess_data(val_ds)

        exp = solstice.train(
            exp,
            num_epochs,
            train_ds,
            val_ds,
            callbacks=[prof_callback, early_stop_callback],
        )
        # can't check profiling without actually looking at it, mainly just relying on
        # not throwing an exception to check it works

        # check that the early stopping worked
        assert exp.count == count_after_one_epoch
