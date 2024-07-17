import numpy as np
import jax
import jax.numpy as jnp
import flax

print('jax.devices()',jax.devices())
print("NumPy version:", np.__version__)
print("JAX version:", jax.__version__)
print("Flax version:", flax.__version__)

# 进行简单的JAX操作
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (10,))
print("JAX array:", x)
