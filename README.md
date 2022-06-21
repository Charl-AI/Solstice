# Solstice

Solstice is a library for constructing modular and structured deep learning experiments in JAX. Built with [Equinox](https://docs.kidger.site/equinox/), but designed for full interoparability with JAX neural network libraries e.g. Stax, Haiku, Flax, Optax etc...

**Why use Solstice in a world with Flax/Haiku/Objax/...?** Solstice is *not* a neural network framework. It is a system for **organising** JAX code, with a small library of sane defaults for common use cases (think [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/), but for JAX). The library itself is simple and flexible, leaving most important decisions to the user - we aim to provide high-quality examples to demonstrate the different ways you can use this flexibility.

Solstice is in the pre-alpha stage of development, you can expect it to be broken until I get round to releasing version 0. TODO:

[ ] figure out a way to use `dataclasses.replace` with custom `__init__()` constructors.
[ ] finalise scope and API of package
[ ] create readthedocs
[ ] finish implementations
[ ] finish docs + tutorials
[ ] create tests
[ ] release version 0 on PyPI

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

Notice that we were able to use pure JAX transformations such as `jax.jit` within the class. This is because `solstice.Experiment` is just a subclass of `Equinox.Module`. We explain this further in the NOTE:TODO tutorial, but in general, if you understand JAX/Equinox, you will understand Solstice.

## The `solstice.compat` API

Using `solstice.Experiment` and the related utilities (such as `solstice.Metrics`) is likely enough for many projects, but where Solstice really shines is its ability to tie together different libraries in the JAX ecosystem. We provide `solstice.compat`, a library of compatibility layers which give a common interface for models and optimizers (and datasets???) in JAX. Using this API allows users to write `solstice.Experiment`s that are independent of the libraries used for neural network, optimization etc... We use this power to provide a set of plug-and-play Experiments for common use cases.

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

## Whole API

TODO: move away from readme and into docs (when created)

Solstice has 4 abstractions that help you organise your code. We provide some implementations which can be used for basic cases, but if you need more flexibility, it is trivial to write your own. Solstice is just a small library on top of Equinox, so you can feel free to write pure JAX code and mix-and-match with other libraries. If you understand JAX/Equinox, you understand Solstice.

### Abstractions (these are `ABC`s, so they specify an interface you should implement by subclassing the abstract class)

- `solstice.Experiment`: A Solstice Experiment holds all stateful objects that are part of a training run, e.g. model, optimizer. It also encapsulates useful functions such as training/evalution steps.

- `solstice.Metrics`: A Metrics object is used to collect and aggregate intermediate values, before computing the final metrics during training.

- `solstice.Optimizer`: A Solstice optimizer knows how to initialise optimizer state (with `__init__()`), and apply the optimizer to convert a tree of gradients to a tree of updates (with `__call__()`).

- `solstice.Logger`: A Solstice logger is an interface for logging dictionaries of scalars.

### Implementations (pre-made for basic use cases, implement your own for more niche/advanced cases)

- `solstice.ClassificationExperiment`: Pre-made Experiment for basic multi-class classification problems. Includes pre-made steps and training loop. Just plug in a model, optimizer, and dataset and go!

- `solstice.ClassificationMetrics`: Pre-made Metrics class for basic classification experiments. Keeps track of the overall confusion matrix and uses it to compute a battery of metrics.

- `solstice.OptaxOptimizer`: Adaptor which wraps any [Optax](https://optax.readthedocs.io/en/latest/) optimizer to become a Solstice optimizer.

- `solstice.SGDOptimizer`: Implementation of classic SGD optimizer.

- `solstice.PythonLogger`: Log asynchronously to the terminal with inbuilt [Python logger](https://docs.python.org/3/library/logging.html).

- `solstice.TensorBoardLogger`: Logger for asynchronously writing scalars to [TensorBoard](https://www.tensorflow.org/tensorboard).

- `solstice.WandBLogger`: Adaptor to use [Weights and Biases](https://docs.wandb.ai/) with Solstice logger interface.

## Examples

TODO: move into separate example readme

We provide 6 full examples of Solstice usage in different settings (this is aspirational, not all implemented yet!) *TODO: ensure at least one demonstrates multi-GPU data-parallel and multi-host/TPU, consider building this into default `ClassificationExperiment`*:

- **MNIST MLP:** Implement an MNIST classifier with built-in Solstice modules.

- **CIFAR Convnext:** Implement a [ConvNext](https://arxiv.org/abs/2201.03545) model in Equinox, using the built-in `ClassificationExperiment` and training loop to classify CIFAR-10. Supports single CPU/GPU, data-parallel multi-GPU, and multi-host/TPU. Uses [Hydra](https://hydra.cc/docs/intro/) for config management.

- **Adversarial Training:** Write custom training steps to train a model adversarially for removing bias from Colour-MNIST (based on https://arxiv.org/abs/1812.10352). Uses Haiku to define the base network.

- **Vmap Ensemble:** Train an ensemble of small neural networks simultaneously on one GPU (inspired by https://willwhitney.com/parallel-training-jax.html).

## Tutorials

TODO: move into docs (when created)

We provide 4 tutorial notebooks, for longer form writing and explanations:

- **Design Decisions:** Not really a tutorial, more of an explanation/justification for why Solstice is the way it is.

- **Library Compatibility:** Solstice is remarkably simple and is trivial to use alongside other JAX libraries. We provide a guide for common cases and also demonstrate how to write adaptor layers to make different libraries interchangable.

- **Configuration Management:** The Solstice library is not opinionated about config management and doesn't lock you in to any solution. Nonetheless, it is important to get right because good config management can accelerate your research by allowing faster iteration of ideas. We show how to integrate Solstice into Hydra.


## A note on philosophy

*Doesn't this library go against the spirit of using JAX?*

Maybe... I haven't fully made up my mind yet.

If the reason you like JAX is that it gets you away from object-oriented style code and towards a pure functional paradigm, you might not like the inclusion of the OO design patterns in this codebase. Pragmatically though, Python is not a functional language - without a strong, compile-time-checked type system, it can be difficult to reach the functional paradise. Dependency inversion with abstract base classes seems the most Pythonic way to implement interfaces and they are hard to avoid if you are trying to write scalable code with good types. Importantly, not everyone who uses JAX is a functional programming evangelist or allergic to encapsulation. Some people simply want to use a familiar framework with a well though out set of user-facing abstractions - they might not care about the details of how pure functions make JAX beautiful (although I do). Equinox is probably the closest JAX project to pytorch-style design patterns, so it is a natural choice to use as the base of this project.

As I said. I am undecided as to whether this is a great idea. This is highly experimental. It's inevitable that someone is going to come along and try to force JAX back into an OO mould, so it might as well be me :p.

## Installation

**The codebase is small, so for now this is not set up as a PyPI package. Copy-paste any useful code into your projects.** If you want to play with the library or the examples, the supported method for installation is through VSCode devcontainers. Simply run `Remote containers: Clone Repository in Container Volume` from the VScode command palette to clone the repository in an isolated volume, build the image, spin up a container, and mount a vscode window. Alternatively, you can clone the repository normally and run `Remote containers: Open Folder in Container` to do the same thing.

You will need [Docker with the nvidia container runtime installed](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker), as well as VScode with the `remote development - containers` extension.

*For the image to build, you will need a GPU+driver capable of running CUDA 11.5 e.g. RTX 3090, you can check this by running nvidia-smi and seeing if 'CUDA Version' is 11.5 or greater*

TODO: create alternate build for CPU only.
