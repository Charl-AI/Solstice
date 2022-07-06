"""The adversarial training example demonstrates how to implement custom training logic
to remove bias from Colour MNIST (based on https://arxiv.org/abs/1812.10352). We use
Equinox to define the base network.

!!! summary
    This example demonstrates:

    - How to implement `solstice.Experiment` for adversarially training a fair
        classifier.

    - How to implement a custom `solstice.Metrics` class for tracking fairness-related
        metrics.

    - How to implement a custom `solstice.Callback` for conditional early stopping.

    - Usage of `solstice.LoggingCallback` with
        [Weights and Biases](https://docs.wandb.ai/) integration.

!!! warning
    Aspirational, not implemented yet.

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
