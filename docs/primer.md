# Solstice Primer

Here, we write about the concepts behind Solstice. It is not necessary to understand all of it before getting started, but it might help to understand our design.

## The Problem: coupling in ML research code

In machine learning projects, you generally have at least four main parts of your code:

- model
- dataset
- training strategy (i.e. optimization, train/eval step logic, metrics calculation)
- training/testing loops (including logging, checkpointing etc...)

In many research projects, it is helpful to be able to swap out any of these parts on a whim (ideally, just by changing your config files). In practice, it follows that **the better you can decouple these four parts of your code, the faster you can iterate your experiments.**

Much attention is paid to the four components individually, but many researchers then just throw everything together haphazardly. It can require quite a significant engineering effort to properly decouple all the components, so most people don't bother. Wouldn't it be great if there was a standard way of organising your code to rapidly scale and iterate experiments...

## Related Work / Why Solstice?

We are not the first people to notice the usefulness of a well-engineered experiment library for deep learning. PyTorch Lightning has filled this niche for PyTorch and Sonnet/Keras have done the same for TensorFlow. In JAX, however, there is currently a cambrian explosion of libraries, e.g. Jaxline, Scenic, Objax, Elegy, CLU. All of these works are excellent, but they each have pros and cons.

Libraries such as Jaxline, Objax, and Scenic are more 'framework-ish' - they ask you to do things their way and in return they do a lot for you. The natural tradeoff with this style of library is that you are locked in to their entire ecosystem which can result in a lack of flexibility. Of these, Elegy is the most closely related to Solstice due to its everything-is-a-pytree approach, however, where Elegy aims to be Keras-like, we aim to be lighter-weight. In Solstice, we leave nearly everything to the user and have incremental buy-in as a top design priority.

Equinox is the foundation of this project and is an elegent library for 'callable pytrees' in JAX (we explain this further below). Where Equinox strives to be as general as possible, we offer opinions about how to structure deep learning experiments. Solstice is a small library, implemented entirely in pure JAX+Equinox. Like Equinox, the simplicity of Solstice is its selling point.

CLU (for Flax) is also closely related, with Solstice having a similar Metrics API, as well as a similar philosophy of providing flexible utilities for common use cases. Surprisingly, Flax itself comes quite close to Solstice with its `flax.training.TrainState` concept. This is an object which holds experiment state with related functions and, taken to the extreme, can end up looking very similar to a `solstice.Experiment`.

## The key idea: immutable classes in Python

In object-oriented code, classes encapsulate both state and transformations on that state. Crucially, the state is mutable, so calling a method might return nothing but alter the object's state. This is a *side effect*. In functional programming, side effects are avoided by making state immutable, so instead of calling a method to update an object's state, you could pass the object to a pure function, returning a new object with the altered state. This is generally an easier paradigm to reason about and immutability is also needed in JAX for XLA to work its magic.

This is all great, but Python is not *really* designed for the functional paradigm so it is difficult to fully decouple all the parts of your code. Type hinting functions with `Protocols` can get you surprisingly far, but at some point you will probably want to achieve some level of encapsulation and use abstract base classes to get dependency inversion.

The approach we take in Solstice is to use immutable dataclasses to try to get the best of both worlds, the code below shows how you would implement a simple counter in each of the paradigms.

<table>
<tr>
<td> OO-Style </td> <td> Functional-Style </td> <td> Solstice/Equinox-Style </td>
</tr>
<tr>
<td>

```python
class OOPCounter:
    def __init__(self, initial_value: int = 0) -> None:
        self.count = initial_value

    def increment(self) -> None:
        self.count += 1

# 'initialise' the OO counter
counter = OOPCounter()
assert counter.count == 0
start_id = id(counter)


# 'apply' the increment method
counter.increment()
assert counter.count == 1
end_id = id(counter)
assert start_id == end_id

```

</td>
<td>

```python

def increment_fn(current_value: int) -> int:
    return current_value + 1

# 'initialise' the functional counter
count = 0
assert count == 0
start_id = id(count)


# 'apply' the increment func
count = increment_fn(count)
assert count == 1
end_id = id(count)
assert start_id != end_id

```
</td>
<td>

```python

import dataclasses

@dataclasses.dataclass(frozen=True)
class SolsticeStyleCounter:
    count: int = 0

    def increment(self) -> "SolsticeStyleCounter":
        return dataclasses.replace(self, count=self.count + 1)

# 'initialise' the SolsticeStyleCounter
counter = SolsticeStyleCounter()
assert counter.count == 0
start_id = id(counter)

# 'apply' the increment method, returning a new state object
counter = counter.increment()
assert counter.count == 1
end_id = id(counter)
assert start_id != end_id
```

</td>
</tr>
</table>


Notice that the Solstice style counter did not mutate its state, it returned a new instance of itself with the new state. The great thing about this pattern is that by keeping our data structures immutable, we get to keep the readability and XLA optimization advantages that come with it, however, we also get all the power of Python classes and OO-ish design patterns.

In practice, in machine learning, this means we can replace the common init/apply pure functions with methods in a frozen dataclass (usually `__init__()`, and `__call__()`). There is also one final matter to take care of... JAX only operates on PyTrees and doesn't know how to deal with dataclasses. This is why we build Solstice on top of Equinox, because an `equinox.Module` is just a dataclass which is registered as a PyTree.

This is a powerful paradigm, and it allows us to trivially do things which are considerably more difficult in Flax/Haiku, like specifying common interfaces for models using abstract base classes:

```python

from abc import ABC, abstractmethod
import equinox as eqx
import jax.numpy as jnp

class EncoderDecoderModel(eqx.Module, ABC):
    """Encoder-Decoder models (e.g. VAEs) implement this interface."""

    @abstractmethod
    def __init__(self):
        """Initialise model parameters."""
        pass

    @abstractmethod
    def encode(self, x: jnp.ndarray) -> jnp.ndarray:
        """Encode the input data into a latent representation."""
        pass

    @abstractmethod
    def decode(self, z: jnp.ndarray) -> jnp.ndarray:
        """Decode the latent representation."""
        pass

```

All classes in Solstice (except the training loop ones) inherit from `equinox.Module`. To get started, your mental model can simply be that an `equinox.Module` is a frozen dataclass that JAX can interpret, although it helps to understand the ideas behind Equinox a bit.

## Equinox and filtered transformations

*Filtered transformations are explained fully in the [Equinox Docs](https://docs.kidger.site/equinox/), but we include a quick primer here for completeness.*

When we use Equinox to create class-style PyTrees, we are able to attach things to the PyTree that JAX function transformations don't know how to interpret. For example, you might store a callable, like your model's `apply()` function in a `solstice.Experiment`. Now, if you try to `jax.jit` a method in the experiment which takes `self` as a parameter, JAX with throw an error because it doesn't know how to JIT with callables as inputs to the function (usually, you would have to specify the argument as static). Equinox gets around this problem with filtered transformations, which involve splitting the PyTree into a 'static' part and a 'parameter' part and then reassembling the PyTree after the transformation. This is an elegant way of solving the problem and is surprisingly powerful, because PyTrees can be filtered with arbitrary criteria (by default, `jnp.array`s are treated as parameters and everything else is treated as static). Here are some examples to get a feel for it:

TODO: make examples... make a few, some that break some that don't
