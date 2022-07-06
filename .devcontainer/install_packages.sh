#! /bin/bash

pip install --upgrade pip

poetry install

if [ -z "$(which nvcc)" ]; then
    echo "CUDA not detected, using CPU version of JAX."
else
    echo "CUDA detected, installing CUDA version of JAX"
    pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
fi
