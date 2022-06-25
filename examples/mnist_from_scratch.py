"""The MNIST example demonstrates how to implement everything yourself using just the
Solstice base classes. In practice, you can do this in far fewer lines by using the
`solstice.compat` API and the pre-made `solstice.ClassificationExperiment` etc."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Tuple

import equinox as eqx
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm

import solstice

# Ensure TF does not see GPU and grab all GPU memory.
tf.config.set_visible_devices([], device_type="GPU")


class MyMetrics(solstice.Metrics):
    """Custom Metrics class for calculating accuracy and average loss. Included for
    didactic purposes, in practice `solstice.ClassificationMetrics` is better."""

    average_loss: float
    count: int
    num_correct: int

    def __init__(self, preds: jnp.ndarray, targets: jnp.ndarray, loss: float) -> None:
        self.average_loss = loss
        self.count = preds.shape[0]  # assumes batch is first dim
        self.num_correct = jnp.sum(preds == targets)

    def merge(self, other: MyMetrics) -> MyMetrics:
        # can simply sum num_correct and count
        new_num_correct = self.num_correct + other.num_correct
        new_count = self.count + other.count

        # average loss is weighted by count from each object
        new_loss = (
            self.average_loss * self.count + other.average_loss * other.count
        ) / (self.count + other.count)

        return eqx.tree_at(lambda m: [m.num_correct, m.count, m.average_loss], self, [new_num_correct, new_count, new_loss])  # type: ignore

    def compute(self) -> Mapping[str, float]:
        return {
            "accuracy": self.num_correct / self.count,
            "average_loss": self.average_loss,
        }


class MNISTClassifier(solstice.Experiment):
    """MNIST Classification Experiment, implemented with Haiku and Optax."""

    params: Any
    opt_state: Any
    model_apply: Callable
    opt: optax.GradientTransformation
    num_classes: int

    def __init__(self, rng: int):
        key = jax.random.PRNGKey(rng)
        dummy_batch = jnp.zeros((32, 784))
        init_fn, apply_fn = hk.transform(lambda x: hk.nets.MLP([300, 100, 10])(x))
        self.params = init_fn(key, dummy_batch)
        self.model_apply = apply_fn

        self.opt = optax.adam(learning_rate=1e-3)
        self.opt_state = self.opt.init(self.params)
        self.num_classes = 10

    @eqx.filter_jit
    def __call__(self, imgs: jnp.ndarray) -> Any:
        logits = self.model_apply(self.params, None, imgs)
        return jnp.argmax(logits, axis=-1)

    # always trace batch pytree, but only trace jnp.arrays in self
    @eqx.filter_jit(kwargs=dict(batch=True))
    def train_step(
        self, batch: Tuple[jnp.ndarray, jnp.ndarray]
    ) -> Tuple[solstice.Metrics, solstice.Experiment]:
        imgs, labels = batch

        def loss_fn(params, x, y):
            logits = self.model_apply(params, None, x)
            loss = jnp.mean(
                optax.softmax_cross_entropy(logits, jax.nn.one_hot(y, self.num_classes))
            )
            return loss, logits

        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            self.params, imgs, labels
        )

        updates, new_opt_state = self.opt.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        preds = jnp.argmax(logits, axis=-1)
        metrics = MyMetrics(preds, labels, loss)

        return metrics, self.replace(params=new_params, opt_state=new_opt_state)

    @eqx.filter_jit(kwargs=dict(batch=True))
    def eval_step(self, batch: Tuple[jnp.ndarray, jnp.ndarray]) -> solstice.Metrics:
        imgs, labels = batch

        logits = self.model_apply(self.params, None, imgs)
        loss = jnp.mean(
            optax.softmax_cross_entropy(
                logits, jax.nn.one_hot(labels, self.num_classes)
            )
        )
        preds = jnp.argmax(logits, axis=-1)
        metrics = MyMetrics(preds, labels, loss)
        return metrics

    # def replace(self, **changes):
    #     keys, vals = zip(*changes.items())
    #     return eqx.tree_at(lambda c: [c.__dict__[key] for key in keys], self, vals)


def basic_train(
    experiment: solstice.Experiment, ds: tf.data.Dataset, num_epochs: int
) -> solstice.Experiment:
    """Rudimental training loop for didactic purposes. Use one of the included solstice
    loops or write your own like this one."""

    for epoch in tqdm(range(num_epochs)):
        metrics = None
        for batch in ds.as_numpy_iterator():
            batch_metrics, experiment = experiment.train_step(batch)
            metrics = batch_metrics.merge(metrics) if metrics else batch_metrics
        assert metrics is not None
        metrics_dict = metrics.compute()
        metrics = None
        print(f"Train {epoch=}: {metrics_dict=}")

    return experiment


def basic_test(
    experiment: solstice.Experiment, ds: tf.data.Dataset
) -> Mapping[str, float]:
    """Rudimental test loop for didactic purposes. Use one of the included solstice
    loops or write your own like this one."""

    metrics = None
    for batch in tqdm(ds.as_numpy_iterator()):
        batch_metrics = experiment.eval_step(batch)
        metrics = batch_metrics.merge(metrics) if metrics else batch_metrics
    assert metrics is not None
    metrics_dict = metrics.compute()
    print(f"Test: {metrics_dict=}")
    return metrics_dict


def main():
    train_ds = tfds.load(name="mnist", split="train", as_supervised=True)
    assert isinstance(train_ds, tf.data.Dataset)
    preprocess_mnist = lambda x, y: (
        tf.reshape(tf.cast(x, tf.float32) / 255, (784,)),
        tf.cast(y, tf.float32),
    )
    train_ds = train_ds.map(preprocess_mnist).batch(32).prefetch(1)

    exp = MNISTClassifier(rng=0)
    trained_exp = basic_train(exp, train_ds, num_epochs=5)

    test_ds = tfds.load(name="mnist", split="test", as_supervised=True)
    assert isinstance(test_ds, tf.data.Dataset)
    test_ds = test_ds.map(preprocess_mnist).batch(32).prefetch(1)

    _ = basic_test(trained_exp, test_ds)


if __name__ == "__main__":
    main()
