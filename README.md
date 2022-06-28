# Solstice


Solstice is a library for constructing modular and structured deep learning experiments in JAX. Built with [Equinox](https://docs.kidger.site/equinox/), but designed for full interoparability with JAX neural network libraries e.g. Stax, Haiku, Flax, Optax etc...

**Why use Solstice in a world with Flax/Haiku/Objax/...?** Solstice is *not* a neural network framework. It is a system for **organising** JAX code, with a small library of sane defaults for common use cases (think [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/), but for JAX). The library itself is simple and flexible, leaving most important decisions to the user - we aim to provide high-quality examples to demonstrate the different ways you can use this flexibility.

> Solstice is in the pre-alpha stage of development, you can expect it to be broken until I get round to releasing version 1. It has not yet been uploaded to PyPI, the installation wont work.

### Installation

```bash
pip install solstice
```

### [Docs](https://charl-ai.github.io/Solstice/)


## Getting Started with `solstice.Experiment`

The central abstraction in Solstice is the `solstice.Experiment`. An Experiment is a container for all functions and stateful objects that are relevant to a run. You can create an Experiment by subclassing `solstice.Experiment` and implementing the abstractmethods for initialisation/training/evaluation/inference. Solstice Experiments come with a pre-made training loop which will fit most use cases (you can always overrwrite it with your own).


```python

import dataclasses
import jax
import jax.numpy as jnp
import solstice

class RandomClassifier(solstice.Experiment):
    """A terrible, terrible classifier for binary class problems :("""
    rng_state: Any

    def __init__(self, rng: int):
        self.rng_state = jax.random.PRNGKey(rng)

    def __call__(self, x):
        del x
        return jax.random.bernoulli(self.rng_state, p=0.5)

    @jax.jit
    def train_step(self, batch: Tuple[jnp.ndarray, ...]) -> Tuple[solstice.Metrics, "MNISTClassifier"]:
        x, y = batch
        preds = jax.vmap(self)(x)
        metrics = solstice.ClassificationMetrics(preds, y, loss=jnp.nan, num_classes=2)
        new_rng_state = jax.random.split(self.rng_state)[0]

        return metrics, dataclasses.replace(self, rng_state=new_rng_state)


    @jax.jit
    def eval_step(self, batch: Tuple[jnp.ndarray, ...]) -> solstice.Metrics:
        x, y = batch
        preds = jax.vmap(self)(x)
        metrics = solstice.ClassificationMetrics(preds, y, loss=jnp.nan, num_classes=2)
        return metrics

exp = MNISTClassifier(42)
trained_exp = exp.train(...)

```

Notice that we were able to use pure JAX transformations such as `jax.jit` within the class. This is because `solstice.Experiment` is just a subclass of `Equinox.Module`. We explain this further in the [Solstice Primer](https://charl-ai.github.io/Solstice/primer/), but in general, if you understand JAX/Equinox, you will understand Solstice.

## The `solstice.compat` API

Using `solstice.Experiment` and the related utilities (such as `solstice.Metrics`) is likely enough for many projects, but where Solstice really shines is its ability to tie together different libraries in the JAX ecosystem. We provide `solstice.compat`, a library of compatibility layers which give a common interface for neural network and optimization libraries in JAX. Using this API allows users to write `solstice.Experiment`s in pure JAX and reuse them with different frameworks. We use this power to provide a set of plug-and-play Experiments for common use cases.

Here, we show how `solstice.ClassificationExperiment` can be used with the `solstice.compat` API to classify MNIST with any neural network framework in just a few lines:

TODO: complete example

<table>
<tr>
<td> Solstice+Stax </td> <td> Solstice+Flax </td> <td> Solstice+Haiku </td>
</tr>
<tr>
<td>

```python

import solstice
import optax

opt = solstice.compat.OptaxOptimizer(optax.adam(3e-4))

exp = solstice.ClassificationExperiment(model, opt)
exp.train(...)

```

</td>
<td>

```python

import solstice
import optax

opt = solstice.compat.OptaxOptimizer(optax.adam(3e-4))

exp = solstice.ClassificationExperiment(model, opt)
exp.train(...)

```
</td>
<td>

```python

import solstice
import optax

opt = solstice.compat.OptaxOptimizer(optax.adam(3e-4))

exp = solstice.ClassificationExperiment(model, opt)
exp.train(...)
```

</td>
</tr>
</table>

## Incrementally buying-in

Solstice is a library, not a framework, and it is important to us that you have the freedom to use as little or as much of it as you like. Heres an example of a Flax project in 5 different stages of Solstice-ification:

1 -> 2: Organise training code with `solstice.Experiment`. This replaces the `TrainState` object and serves to better organise your code.

2 -> 3: Implement metrics calculation with the `solstice.Metrics` interface. This simplifies logging and shifts complexity away from the training loop.

3 -> 4: Use the premade `solstice.train` function with callbacks to specify custom behaviour. This gives a balance of flexibility, whilst removing boilerplate code.

4 -> 5: Wrap the model with the `solstice.compat.ClassificationModel` API. This fully decouples the model from the training logic - it is now trivial to mix-and-match models built with different neural network libraries and reuse your `Experiment` code.

TODO: complete example

<table>
<tr>
<td> 1. Pure Flax </td> <td> 2. Introduce `solstice.Experiment` </td> <td> 3. Introduce `solstice.Metrics` </td> <td> 4. Introduce `solstice.train` </td> <td> 5. Introduce `solstice.compat` </td>
</tr>
<tr>
<td>

```python

x

```

</td>
<td>

```python

x

```
</td>
<td>

```python

x

```

</td>
<td>

```python

x

```
</td>
<td>

```python

x

```
</td>
</tr>
</table>
