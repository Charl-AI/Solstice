# Examples

We provide X full examples of Solstice usage in different settings (this is aspirational, not all implemented yet!) *TODO: ensure at least one demonstrates multi-GPU data-parallel and multi-host/TPU, consider building this into default `ClassificationExperiment`*:

::: examples.mnist_from_scratch
    selection:
            members: false
            show_root_full_path: false
            show_root_toc_entry: true
            show_root_heading: true

!!! todo
        - **CIFAR Convnext:** Implement a [ConvNext](https://arxiv.org/abs/2201.03545) model in Equinox, using the built-in `ClassificationExperiment` and training loop to classify CIFAR-10. Supports single CPU/GPU, data-parallel multi-GPU, and multi-host/TPU. Uses [Hydra](https://hydra.cc/docs/intro/) for config management.

    - **Adversarial Training:** Write custom training steps to train a model adversarially for removing bias from Colour-MNIST (based on https://arxiv.org/abs/1812.10352). Uses Haiku to define the base network.

    - **Vmap Ensemble:** Learn how to implement parallelism strategies by training an ensemble of small neural networks simultaneously on one GPU (inspired by https://willwhitney.com/parallel-training-jax.html).

    - **X Validation:**


The examples are all runnable as standalone scripts. We provide a dockerfile to set up a GPU-accelerated environment with Solstice and the other dependencies necessary. This allows you to run the examples easily like so:

```
docker build ...
docker run ...
```

!!! info
    You will need to have Docker with the nvidia container runtime installed. The Docker image will not build if you do not have a GPU capable of running CUDA 11.3.