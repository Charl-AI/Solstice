"""The vmap ensemble example demonstrates how to implement different parallelism
strategies by training an ensemble of small neural networks simultaneously on one GPU
(inspired by https://willwhitney.com/parallel-training-jax.html). Uses Flax to define
the base network.

!!! summary
    This example demonstrates:

    - How to implement `solstice.Experiment` for training an ensemble simultaneously
        on one GPU.


!!! warning
    Aspirational, not implemented yet.

"""
