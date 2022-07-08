# Solstice

Solstice is a library for constructing modular and structured deep learning experiments in JAX. Built with [Equinox](https://docs.kidger.site/equinox/), but designed for full interoparability with JAX neural network libraries e.g. Stax, Haiku, Flax, Optax etc...

**Why use Solstice in a world with Flax/Haiku/Objax/...?** Solstice is *not* a neural network framework. It is a system for **organising** JAX code, with a small library of sane defaults for common use cases (think [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/), but for JAX). The library itself is simple and flexible, leaving most important decisions to the user - we aim to provide high-quality examples to demonstrate the different ways you can use this flexibility.

> Solstice is in the pre-alpha stage of development, you can expect it to be broken until I get round to releasing version 1. It has not yet been uploaded to PyPI, the installation wont work.

### Installation

First, [install JAX](https://github.com/google/jax#installation), then:

```bash
pip install <not yet in PyPI>
```

### [Docs](https://charl-ai.github.io/Solstice/)

Solstice is fully documented, including a full API Reference, as well as tutorials and examples. Below, we provide a bare minimum example for how to get started.


## Getting Started

The central abstraction in Solstice is the `solstice.Experiment`. An Experiment is a container for all functions and stateful objects that are relevant to a run. You can create an Experiment by subclassing `solstice.Experiment` and implementing the abstractmethods for initialisation, training, and evaluation. Experiments are best used with `solstice.Metrics` for tracking metrics and `solstice.train()` so you can stop writing boilerplate training loops.


```python
from typing import Any, Tuple
import logging
import jax
import jax.numpy as jnp
import solstice
import tensorflow_datasets as tfds

logging.basicConfig(level=logging.INFO)


class RandomClassifier(solstice.Experiment):
    """A terrible, terrible classifier for binary class problems :("""

    rng_state: Any

    def __init__(self, rng: int):
        self.rng_state = jax.random.PRNGKey(rng)

    def __call__(self, x):
        del x
        return jax.random.bernoulli(self.rng_state, p=0.5).astype(jnp.float32)

    @jax.jit
    def train_step(
        self, batch: Tuple[jnp.ndarray, ...]
    ) -> Tuple["RandomClassifier", solstice.Metrics]:
        x, y = batch
        preds = jax.vmap(self)(x)
        # use solstice Metrics API for convenient metrics calculation
        metrics = solstice.ClassificationMetrics(preds, y, loss=jnp.nan, num_classes=2)
        new_rng_state = jax.random.split(self.rng_state)[0]

        return solstice.replace(self, rng_state=new_rng_state), metrics

    @jax.jit
    def eval_step(
        self, batch: Tuple[jnp.ndarray, ...]
    ) -> Tuple["RandomClassifier", solstice.Metrics]:
        x, y = batch
        preds = jax.vmap(self)(x)
        metrics = solstice.ClassificationMetrics(preds, y, loss=jnp.nan, num_classes=2)
        return self, metrics


train_ds = tfds.load(name="mnist", split="train", as_supervised=True)  # type: Any
train_ds = train_ds.batch(32).prefetch(1)
exp = RandomClassifier(42)
# use solstice.train() with callbacks to remove boilerplate code
trained_exp = solstice.train(
    exp, num_epochs=1, train_ds=train_ds, callbacks=[solstice.LoggingCallback()]
)

```

Notice that we were able to use pure JAX transformations such as `jax.jit` and `jax.vmap` within the class. This is because `solstice.Experiment` is just a subclass of `Equinox.Module`. We explain this further in the [Solstice Primer](https://charl-ai.github.io/Solstice/primer/), but in general, if you understand JAX/Equinox, you will understand Solstice.


## Incrementally buying-in

Solstice is a library, not a framework, and it is important to us that you have the freedom to use as little or as much of it as you like. If are interested in starting using Solstice, but don't know where to begin, here are three steps towards Solstice-ification.

### Stage 1: organise your training code with `solstice.Experiment`

The `Experiment` object contains stateful objects such as model and optimizer parameters and also encapsulates the steps for training and evaluation. In Flax, this would replace the `TrainState` object and serve to better organise your code. At this stage, the main advantage is that your code is more readable and scalable because you can define different `Experiment`s for different use cases.

### Stage 2: implement `solstice.Metrics` for tracking metrics

A `solstice.Metrics` object knows how to calculate and accumulate intermediate results, before computing final metrics. The main advantage is the ability to scalably track lots of metrics with a common interface. By tracking intermediate results and computing at the end, it is easier to handle metrics which are not 'averageable' over batches (e.g. precision).

### Stage 3: use the premade `solstice.train()` loop with `solstice.Callback`s

Training loops are usually boilerplate code. We provide premade training and testing loops which integrate with a simple and flexible callback system. This allows you to separate the basic logic of training from customisable side effects such as logging and checkpointing. We provide some useful pre-made callbacks and give examples for how to write your own.

## Our Logos


We have two Solstice logos: the Summer Solstice :sun_with_face: and the Winter Solstice :first_quarter_moon_with_face:. Both were created with [Dall-E mini](https://huggingface.co/spaces/dalle-mini/dalle-mini) (free license) with the following prompt:
> a logo featuring stonehenge during a solstice

![Solstice Logos](https://github.com/Charl-AI/Solstice/blob/main/docs/both_logos.png?raw=true)
