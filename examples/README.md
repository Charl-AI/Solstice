## Examples

We provide X full examples of Solstice usage in different settings (this is aspirational, not all implemented yet!) *TODO: ensure at least one demonstrates multi-GPU data-parallel and multi-host/TPU, consider building this into default `ClassificationExperiment`*:

- **MNIST from scratch:** Implement an MNIST classifier using just the basic solstice abstractions. Uses Haiku to define the model (without the `solstice.compat` API).

- **CIFAR Convnext:** Implement a [ConvNext](https://arxiv.org/abs/2201.03545) model in Equinox, using the built-in `ClassificationExperiment` and training loop to classify CIFAR-10. Supports single CPU/GPU, data-parallel multi-GPU, and multi-host/TPU. Uses [Hydra](https://hydra.cc/docs/intro/) for config management.

- **Adversarial Training:** Write custom training steps to train a model adversarially for removing bias from Colour-MNIST (based on https://arxiv.org/abs/1812.10352). Uses Haiku to define the base network.

- **Vmap Ensemble:** Learn how to implement parallelism strategies by training an ensemble of small neural networks simultaneously on one GPU (inspired by https://willwhitney.com/parallel-training-jax.html).

- **X Validation:**
