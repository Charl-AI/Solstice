"""The adversarial training example demonstrates how to implement custom training logic
to remove bias from Colour MNIST (based on https://arxiv.org/abs/1812.10352). We use
Equinox to define the base network.

!!! summary
    This example implements:

    - a `solstice.Metrics` class for tracking fairness-related metrics

    - a `solstice.Experiment` class for adversarially training the model

    - `solstice.Callback`s for custom logging with `solstice.train()`

!!! warning
    Not Implemented Yet

"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import solstice
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm

# Ensure TF does not see GPU and grab all GPU memory.
tf.config.set_visible_devices([], device_type="GPU")
