import chex
import jax
import jax.numpy as jnp
import solstice
import functools

# import os
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
# with chex.fake_pmap
chex.set_n_cpu_devices(5)

# with chex.fake_pmap():


@functools.partial(jax.pmap, axis_name="batch", in_axes=0)
def f(targets):
    key = jax.random.PRNGKey(0)
    preds = jax.random.bernoulli(key, p=0.5, shape=targets.shape).astype(jnp.float32)
    preds = jax.lax.all_gather(preds, axis_name="batch")
    return solstice.ClassificationMetrics(preds, targets, loss=jnp.nan, num_classes=2)


key = jax.random.PRNGKey(1)
targets = jax.random.bernoulli(key, p=0.5, shape=(5,)).astype(jnp.float32)
metrics = f(targets)
