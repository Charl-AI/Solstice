# Solstice


Solstice is a library for constructing modular and structured deep learning experiments in JAX. Built with [Equinox](https://docs.kidger.site/equinox/), but designed for full interoparability with JAX neural network libraries e.g. Stax, Haiku, Flax, Optax etc...

**Why use Solstice in a world with Flax/Haiku/Objax/...?** Solstice is *not* a neural network framework. It is a system for **organising** JAX code, with a small library of sane defaults for common use cases (think [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/), but for JAX). The library itself is simple and flexible, leaving most important decisions to the user - we aim to provide high-quality examples to demonstrate the different ways you can use this flexibility.

> Solstice is in the pre-alpha stage of development, you can expect it to be broken until I get round to releasing version 1. It has not yet been uploaded to PyPI, the installation wont work.

### Installation

First, [install JAX](https://github.com/google/jax#installation), then:

```bash
pip install solstice
```

### [Docs](https://charl-ai.github.io/Solstice/)


## Getting Started with `solstice.Experiment`

The central abstraction in Solstice is the `solstice.Experiment`. An Experiment is a container for all functions and stateful objects that are relevant to a run. You can create an Experiment by subclassing `solstice.Experiment` and implementing the abstractmethods for initialisation, training, and evaluation. Experiments integrate with the `solstice.Trainer` so you can stop writing boilerplate training loops.


```python
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import solstice
import tensorflow_datasets as tfds


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
trained_exp = solstice.train(exp, num_epochs=1, train_ds=train_ds)


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

Solstice is a library, not a framework, and it is important to us that you have the freedom to use as little or as much of it as you like. We give examples of Flax projects in 5 different stages of Solstice-ification:

### Stage 1: organise your training code with `solstice.Experiment`

The `Experiment` object replaces the `TrainState` object and serves to better organise your code. At this stage, the main advantage is that your code is more readable and scalable because you can define different `Experiment`s for different use cases.

Below is an example of an `Experiment` for training a neural network on MNIST.

<table>
<tr>
<td> Pure Flax </td> <td> Using solstice.Experiment </td>
</tr>
<tr>
<td>

```python

@struct.dataclass
class TrainState:
    apply_fn: Callable = struct.field(pytree_node=False)
    params: Any
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState

    @classmethod
    def create(cls, params, apply_fn, tx):
        opt_state = tx.init(params)
        return cls(apply_fn, params, tx, opt_state)

# assumes we defined MLP as a flax model elsewhere
def create_initial_state(rng: int) -> train_state.TrainState:
    key = jax.random.PRNGKey(rng)
    dummy_x = jnp.zeros((32, 784))
    net = MLP(features=[200, 200, 10])
    params = jax.jit(net.init)(key, dummy_x)
    return train_state.TrainState.create(
        params=params, apply_fn=net.apply, tx=optax.sgd(4e-3)
    )

@jax.jit
def train_step(
    state: TrainState, batch: Tuple[jnp.ndarray, jnp.ndarray]
) -> Tuple[TrainState, Mapping[str, Any]]:
    imgs, labels = batch

    def loss_fn(params):
        logits = state.apply_fn(params, imgs)
        loss = jnp.mean(optax.softmax_cross_entropy(logits, jax.nn.one_hot(labels, 10)))
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    updates, new_opt_state = state.tx.update(grads, self.opt_state, self.params)
    new_params = optax.apply_updates(self.params, updates)

    metrics = {
        "loss": loss,
        "accuracy": jnp.mean(jnp.argmax(logits, axis=-1) == labels),
    }
    return new_state, metrics

```

</td>
<td>

```python


class MNISTClassifier(solstice.Experiment):
    apply_fn: Callable
    params: Any
    tx: optax.GradientTransformation
    opt_state: Any

    def __init__(self, rng: int):
        key = jax.random.PRNGKey(rng)
        dummy_x = jnp.zeros((32, 784))
        net = MLP(features=[200, 200, 10])
        self.apply_fn = net.apply
        self.params = jax.jit(net.init)(key, dummy_x)
        self.tx = optax.sgd(4e-3)
        self.opt_state = self.tx.init(self.params)

    @eqx.filter_jit(kwargs=dict(batch=True))
    def train_step(
        self, batch: Tuple[jnp.ndarray, jnp.ndarray]
    ) -> Tuple[solstice.Experiment, Mapping[str, Any]]:
        imgs, labels = batch

        def loss_fn(params):
            logits = self.apply_fn(params, imgs)
            loss = jnp.mean(
                optax.softmax_cross_entropy(logits, jax.nn.one_hot(labels, 10))
            )
            return loss, logits

        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(self.params)
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        metrics = {
            "loss": loss,
            "accuracy": jnp.mean(jnp.argmax(logits, axis=-1) == labels),
        }
        return (
            solstice.replace(self, params=new_params, opt_state=new_opt_state),
            metrics,
        )

    def eval_step(self, batch):
        raise NotImplementedError("not bothering with evaluation in this demo")

```

</tr>
</table>

### Stage 2: implement `solstice.Metrics` for tracking metrics

A `solstice.Metrics` object knows how to calculate and accumulate intermediate results, before computing final metrics. The main advantage is the ability to scalably track lots of metrics with a common interface. By tracking intermediate results and computing at the end, it is easier to handle metrics which are not 'averageable' over batches (e.g. precision).

Below, we show how you might implement a class for handling loss, accuracy, and precision for a binary classification task. Notice that the Solstice solution moves complexity away from the training step and loop. If you wanted to add more metrics, you would only have to change the `Metrics` class.

<table>
<tr>
<td> 1. Metrics in pure JAX </td> <td> 2. Using solstice.Metrics </td>
</tr>
<tr>
<td>

```python

@dataclass
class MyMetrics:
    tp: int
    tn: int
    pp: int
    loss: float
    count: int


def compute_final_metrics(metrics_list: List[MyMetrics]) -> Mapping[str, Any]:
    total_tp = sum([m.tp for m in metrics_list])
    total_tn = sum([m.tn for m in metrics_list])
    total_pp = sum([m.pp for m in metrics_list])
    total_loss = sum([m.loss for m in metrics_list])
    total_count = sum([m.count for m in metrics_list])

    return {
        "loss": total_loss / total_count,
        "accuracy": (total_tp + total_tn) / total_count,
        "precision": total_tp / total_pp
    }


class BinaryClassifier(solstice.Experiment):
... # other code for initialisation etc...

    @eqx.filter_jit(kwargs=dict(batch=True))
    def train_step(self, batch):
        imgs, labels = batch

        def loss_fn(params):
            ... # do something
            return loss, logits

        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(self.params)
        new_params, new_opt_state = ... # do some optimization

        preds = jnp.argmax(logits, axis=-1)
        metrics = MyMetrics(
            tp = jnp.sum(preds == 1 & labels == 1),
            tn = jnp.sum(preds == 0 & labels == 0),
            pp = jnp.sum(preds == 1),
            loss = loss,
            count = preds.shape[0]
        )

        return (
            solstice.replace(self, params=new_params, opt_state=new_opt_state),
            metrics,
        )


def train(experiment: BinaryClassifier, ds: tf.data.Dataset, num_epochs: int):
    for epoch in tqdm(range(num_epochs)):
        metrics = []
        for batch in ds.as_numpy_iterator():
            experiment, batch_metrics = train_step(experiment, batch)
            metrics.append(batch_metrics)

        metrics = compute_final_metrics(metrics)
        print(f"Train {epoch=}: {metrics=}")
        metrics = []
    return experiment

```

</td>
<td>

```python

class MyMetrics(solstice.Metrics):
    tp: int
    tn: int
    pp: int
    loss: float
    count: int

    def __init__(self, preds, labels, loss):
        self.tp = jnp.sum(preds == 1 & labels == 1)
        self.tn = jnp.sum(preds == 0 & labels == 0)
        self.pp = jnp.sum(preds == 1)
        self.loss = loss
        self.count = preds.shape[0] # preds is shape [num_examples,]

    def merge(self, other):
        new_count = self.count + other.count
        return solstice.replace(self,
            tp=self.tp + other.tp,
            tn=self.tn + other.tn,
            pp=self.pp + other.pp,
            loss=self.loss + other.loss,
            count=self.count+other.count
            )

    def compute(self):
        return {
            "loss": self.loss / self.count,
            "accuracy": (self.tp + self.tn) / self.count
            "precision": self.tp / self.pp
        }

class BinaryClassifier(solstice.Experiment):
    ... # other code for initialisation etc...

    @eqx.filter_jit(kwargs=dict(batch=True))
    def train_step(self, batch):
        imgs, labels = batch

        def loss_fn(params):
            ... # do something
            return loss, logits

        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(self.params)
        new_params, new_opt_state = ... # do some optimization

        preds = jnp.argmax(logits, axis=-1)
        metrics = solstice.ClassificationMetrics(preds, labels, loss, num_classes=10)
        return (
            solstice.replace(self, params=new_params, opt_state=new_opt_state),
            metrics,
        )

def train(
    experiment: BinaryClassifier, ds: tf.data.Dataset, num_epochs: int):
    for epoch in tqdm(range(num_epochs)):
        metrics = None
        for batch in ds.as_numpy_iterator():
            experiment, batch_metrics = experiment.train_step(batch)
            metrics = batch_metrics.merge(metrics) if metrics is not None else batch_metrics
        assert metrics is not None
        print(f"Train {epoch=}: {metrics=}")
        metrics = None
    return experiment


```

</tr>
</table>

### Stage 3: use the premade `solstice.Trainer` with `solstice.Callback`s

Training loops are usually boilerplate code. We provide a premade trainer which integrates with a simple and flexible callback system. Any `solstice.Experiment` can be used with the trainer.

### Stage 4: wrap other frameworks with `solstice.compat`

The `solstice.compat` API allows you to write your `Experiment`s in pure JAX, without depending on external libraries such as flax, optax, or haiku. Here, by wrapping your model in the flax compatibility layer, you are able to remove most of your code by using the premade `solstice.ClassificationExperiment`.
