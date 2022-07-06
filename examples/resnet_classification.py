"""The ResNet example demonstrates how to perform image classification on CIFAR10 with
data-parallel multi-gpu support. Uses Haiku to define the base neural net and
[Hydra](https://hydra.cc/docs/intro/) for config management.

!!! faq
    Run `python examples/resnet_classification.py -h` to see the available configuration
    options.

    Logging is done with TensorBoard. Run `tensorboard --logdir=outputs` to view the
    logs.

!!! summary
    This example demonstrates:

    - How to implement `solstice.Experiment` for training a ResNet50 with multi-gpu
        support.

    - Usage of `solstice.ClassificationMetrics` for tracking metrics.

    - Usage of `solstice.LoggingCallback` with TensorBoard (using the
        [CLU](https://github.com/google/CommonLoopUtils) SummaryWriter interface).

    - Usage of `solstice.ProfilingCallback` for profiling with TensorBoard.

!!! warning
    Multi-GPU support is not yet implemented. Currently only working with single GPU.

"""

from __future__ import annotations

import dataclasses
import functools
import logging
import os
from typing import Literal, Mapping, Tuple

import equinox as eqx
import haiku as hk
import hydra
import jax
import jax.numpy as jnp
import optax
import solstice
import tensorflow as tf
import tensorflow_datasets as tfds
from clu import metric_writers
from hydra.core.config_store import ConfigStore

# Ensure TF does not see GPU and grab all GPU memory.
tf.config.set_visible_devices([], device_type="GPU")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # JAX memory preallocation
logging.getLogger("absl").setLevel(logging.WARNING)  # Keep TF/JAX quiet


#######################################################################################
# STEP 1: Create the Solstice Experiment
#######################################################################################


class ResNetClassifier(solstice.Experiment):
    """A ResNet50 image classifier for multiclass problems. You can use it with binary
    problems by treating it as a multiclass case with 2 classes and a threshold fixed
    at 0.5."""

    params: optax.Params
    model_state: optax.OptState
    opt_state: optax.OptState
    model_fns: hk.TransformedWithState
    opt: optax.GradientTransformation
    num_classes: int

    def __init__(
        self,
        optimizer: optax.GradientTransformation,
        num_classes: int,
        input_shape: Tuple[int, int, int],
        *,
        rng: int,
    ):
        assert num_classes > 1, f"Only multiclass (>=2) supported, got {num_classes=}"
        key = jax.random.PRNGKey(rng)
        dummy_batch = jnp.zeros((1, *input_shape))
        self.model_fns = hk.without_apply_rng(
            hk.transform_with_state(
                lambda x, is_training: hk.nets.ResNet50(num_classes, resnet_v2=True)(
                    x, is_training
                ),
            )
        )
        self.params, self.model_state = self.model_fns.init(
            key, dummy_batch, is_training=True
        )

        self.opt = optimizer
        self.opt_state = self.opt.init(self.params)
        self.num_classes = num_classes

    @eqx.filter_jit(kwargs=dict(batch=True))
    def train_step(
        self, batch: Tuple[jnp.ndarray, jnp.ndarray]
    ) -> Tuple[solstice.Experiment, solstice.Metrics]:
        imgs, labels = batch

        def loss_fn(params):
            logits, new_model_state = self.model_fns.apply(
                params, self.model_state, imgs, is_training=True
            )
            loss = jnp.mean(
                optax.softmax_cross_entropy(
                    logits, jax.nn.one_hot(labels, self.num_classes)
                )
            )
            return loss, (logits, new_model_state)

        (loss, (logits, new_model_state)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(self.params)

        updates, new_opt_state = self.opt.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        preds = jnp.argmax(logits, axis=-1)
        metrics = solstice.ClassificationMetrics(preds, labels, loss, self.num_classes)

        return (
            solstice.replace(
                self,
                params=new_params,
                opt_state=new_opt_state,
                model_state=new_model_state,
            ),
            metrics,
        )

    @eqx.filter_jit(kwargs=dict(batch=True))
    def eval_step(
        self, batch: Tuple[jnp.ndarray, jnp.ndarray]
    ) -> Tuple[solstice.Experiment, solstice.Metrics]:
        imgs, labels = batch

        logits, _ = self.model_fns.apply(
            self.params, self.model_state, imgs, is_training=False
        )
        loss = jnp.mean(
            optax.softmax_cross_entropy(
                logits, jax.nn.one_hot(labels, self.num_classes)
            )
        )
        preds = jnp.argmax(logits, axis=-1)
        metrics = solstice.ClassificationMetrics(preds, labels, loss, self.num_classes)

        return self, metrics


#######################################################################################
# STEP 2: Set up datasets with tfds (not massively relevant to solstice, you can do it
# however you like)
#######################################################################################


def get_cifar_datasets(
    data_dir: str, batch_size: int, prefetch_size: int
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    load_ds = functools.partial(
        tfds.load, data_dir=data_dir, as_supervised=True, name="cifar10"
    )

    train_ds = load_ds(split="train[:80%]")
    val_ds = load_ds(split="train[80%:]")
    test_ds = load_ds(split="test")
    assert isinstance(train_ds, tf.data.Dataset)  # just for type narrowing
    assert isinstance(val_ds, tf.data.Dataset)
    assert isinstance(test_ds, tf.data.Dataset)

    preprocess_data = lambda x, y: (
        tf.cast(x, tf.float32) / 255.0,
        tf.cast(y, tf.float32),
    )
    prepare_data = (
        lambda ds: ds.map(
            preprocess_data, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        .batch(batch_size)
        .prefetch(prefetch_size)
        .cache()
    )
    train_ds = prepare_data(
        train_ds.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
    )

    val_ds = prepare_data(val_ds)
    test_ds = prepare_data(test_ds)

    return train_ds, val_ds, test_ds


#######################################################################################
# STEP 3: Set up config with hydra (again, not that relevant to solstice).
#######################################################################################


@dataclasses.dataclass
class Config:
    rng: int = 0
    num_epochs: int = 50
    lr: float = 4e-2

    data_dir: str = "/tmp/data"
    batch_size: int = 128
    prefetch_size: int = 5

    log_dir: str = "logs/"
    log_every_n_steps: int = 50


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


#######################################################################################
# STEP 4: Train!
#######################################################################################


@hydra.main(config_path=None, config_name="config", version_base="1.1")
def main(cfg: Config):
    train_ds, val_ds, test_ds = get_cifar_datasets(
        cfg.data_dir, cfg.batch_size, cfg.prefetch_size
    )
    optimizer: optax.GradientTransformation = optax.adamw(learning_rate=cfg.lr)

    exp = ResNetClassifier(
        optimizer=optimizer,
        num_classes=10,  # hard coded to cifar10 classes and shape
        input_shape=(32, 32, 3),
        rng=cfg.rng,
    )

    logger = metric_writers.create_default_writer(cfg.log_dir)
    # this gets passed to `solstice.LoggingCallback` and tells it what to do with the
    # output of `metrics.compute()`
    def logging_fn(
        metrics_dict: Mapping[str, float],
        step: int,
        mode: Literal["train", "val", "test"],
    ):
        logger.write_scalars(
            step,
            {f"{mode}/{key}": val for key, val in metrics_dict.items()},
        )

    exp = solstice.train(
        exp,
        num_epochs=cfg.num_epochs,
        train_ds=train_ds,
        val_ds=val_ds,
        callbacks=[
            solstice.ProfilingCallback(cfg.log_dir, steps_to_profile=[1, 2, 3]),
            solstice.LoggingCallback(
                log_every_n_steps=cfg.log_every_n_steps, logging_fn=logging_fn
            ),
        ],
    )

    _ = solstice.test(
        exp,
        test_ds=test_ds,
        callbacks=[solstice.LoggingCallback(logging_fn=logging_fn)],
    )


if __name__ == "__main__":
    main()
