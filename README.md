# Solstice

Solstice is a library for constructing modular and structured deep learning experiments in JAX. Built with [Equinox](https://docs.kidger.site/equinox/). Designed for researchers to flexibly create and scale experiments, filling the same niche for Equinox as [Jaxline](https://github.com/deepmind/jaxline) for Haiku, [Scenic](https://github.com/google-research/scenic) for Flax, and [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/) for Pytorch.

The fundamental idea behind this project is to represent the init/apply pattern typically used in JAX and functional programming within a familiar Pythonic class structure (i.e. `__init__()` and `__call__()`), this enables us to use design principles such as dependency inversion to loosen the coupling of our program while retaining immutability and other functional programming advantages. The library itself is simple and flexible, leaving most important decisions to the user - we aim to provide high-quality examples to demonstrate the different ways you can use this flexibility.

## API

Solstice is essentially a library of 4 abstractions that help you organise your code. We provide some implementations which can be used for basic cases, but if you need more flexibility, it is trivial to write your own. Solstice is just a small library on top of Equinox, so you can feel free to write pure JAX code and mix-and-match with other libraries. If you understand JAX/Equinox, you understand Solstice.

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

We provide 6 full examples of Solstice usage in different settings (this is aspirational, not all implemented yet!) *TODO: ensure at least one demonstrates multi-GPU data-parallel, consider building this into default `ClassificationExperiment`*:

- **MNIST MLP:** Implement an MNIST classifier with built-in Solstice modules.

- **CIFAR Convnext:** Implement a [ConvNext](https://arxiv.org/abs/2201.03545) model in Equinox, using the built-in `ClassificationExperiment` and training loop to classify CIFAR-10 on multi-GPU(?). Additionally demonstrates using [Hydra](https://hydra.cc/docs/intro/) for config management.

- **Adversarial Training:** Write your own `Experiment` class to train a model adversarially for removing bias from Colour-MNIST (based on https://arxiv.org/abs/1812.10352). Additionally demonstrates compatibility with [Flax]()

- **Vmap Ensemble:** Train an ensemble of small neural networks simultaneously on one GPU (inspired by https://willwhitney.com/parallel-training-jax.html).

## Tutorials

We provide 3 tutorial notebooks, for longer form writing and explanations:

- **Library Compatibility:** Solstice is remarkably simple and is actually trivial to use alongside other JAX libraries. We provide a guide for common cases.

- **Configuration Management:** The Solstice library is not opinionated about config management and doesn't lock you in to any solution. Nonetheless, it is important to get right because good config management can accelerate your research by allowing faster iteration of ideas. We show how easy it is to integrate Solstice into solutions such as Hydra, ml_collections, argparse, etc...

- **Design Decisions:** Not really a tutorial, more of an explanation/justification for why Solstice is the way it is.

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
