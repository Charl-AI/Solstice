# Examples

We provide full examples of Solstice usage in different settings. Each example is runnable as a standalone script. You can set up an environment for running the examples in just a few clicks by using the provided devcontainer; see the [Contributing](https://charl-ai.github.io/Solstice/contributing/) page for more information.

---

::: examples.mnist_from_scratch
    selection:
            members: false
            show_root_full_path: false
            show_root_toc_entry: true
            show_root_heading: true

---

!!! todo
        - **CIFAR Convnext:** Implement a [ConvNext](https://arxiv.org/abs/2201.03545) model in Equinox, using the built-in `ClassificationExperiment` and training loop to classify CIFAR-10. Supports single CPU/GPU, data-parallel multi-GPU, and multi-host/TPU. Uses [Hydra](https://hydra.cc/docs/intro/) for config management.

    - **Adversarial Training:** Write custom training steps to train a model adversarially for removing bias from Colour-MNIST (based on https://arxiv.org/abs/1812.10352). Uses Haiku to define the base network.

    - **Vmap Ensemble:** Learn how to implement parallelism strategies by training an ensemble of small neural networks simultaneously on one GPU (inspired by https://willwhitney.com/parallel-training-jax.html).

    - **X Validation:**
