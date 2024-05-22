from typing import Callable
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
from jaxtyping import Array, Float, PyTree

def split_and_reduce(true_func: Callable[[PyTree], PyTree], false_func: Callable[[PyTree], PyTree], n_splits: int):
	"""
	Params:
		func (Callable): 
		n_splits (int):
	"""
	def cond_call(x, cond):
		return jax.lax.cond(
			cond,
			lambda x: jax.vmap(true_func)(x),
			lambda x: jax.vmap(false_func)(x),
			x
		)

	def dec_func(x: PyTree[Float[Array, "N ..."]], mask: Float[Array, "N"]):
		x = jax.tree_map(lambda a: a.reshape((n_splits, -1, *a.shape[1:])), x) # S x N/S x ...
		mask = jnp.any(mask.reshape((n_splits, -1)), axis=-1) # S
		
		y = jax.vmap(cond_call)(x, mask)
		y = jax.tree_map(lambda a: a.reshape((-1, *a.shape[2:])), y)
		return y

	return dec_func


if __name__ == '__main__':
	import timeit
	import equinox.nn as nn
	mlp = nn.MLP(10, 10, 128, 4, key=jr.PRNGKey(101))
	
	def f_true(x):
		return mlp(x)

	def f_false(x):
		return jnp.zeros(x.shape)

	f_sr = split_and_reduce(f_true, f_false, 4)

	N = 10_000
	NTrue = 7_000
	x = jr.normal(jr.PRNGKey(1), (N, 10))
	m = jnp.ones((N,)).at[NTrue:].set(0.).astype(bool)
	print(timeit.timeit(lambda : f_sr(x, m), number=1_000))


	N = 10_000
	NTrue = 0
	x = jr.normal(jr.PRNGKey(1), (N, 10))
	m = jnp.ones((N,)).at[NTrue:].set(0.).astype(bool)
	print(timeit.timeit(lambda : f_sr(x, m), number=1_000))