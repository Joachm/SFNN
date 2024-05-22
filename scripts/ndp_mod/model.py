from functools import partial
from typing import Callable, NamedTuple, Optional, Tuple
from jaxtyping import Float, Array, PyTree
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx
import equinox.nn as nn



def fully_connected_topology(key: jax.Array, N: int):
	return jnp.ones((N, N)) - jnp.identity(N)

def erdos_renyi_topology(key: jax.Array, N: int, p: float, self_loops: bool=False):
	A = (jr.uniform(key, (N,N)) < p).astype(float)
	if not self_loops:
		A = jnp.where(jnp.identity(N), 0., A)
	return A

def reservoir(key: jax.Array, N: int, in_dims: int, out_dims: int, p_hh: float=1., p_ih: float=.3, p_ho: float=.5, p_oh:float=0.1 , p_io:float=0.1):
	key_ih, key_hh, key_ho, key_io, key_oi = jr.split(key, 5)
	A = jnp.zeros((N, N))
	I = jnp.arange(in_dims)
	O = jnp.arange(out_dims) + (N-out_dims)
	H = jnp.arange(N-out_dims-in_dims) + in_dims

	A = A.at[jnp.ix_(I, H)].set((jr.uniform(key_ih, (in_dims, len(H)))<p_ih).astype(float))
	A = A.at[jnp.ix_(H, H)].set(erdos_renyi_topology(key_hh, N=len(H), p=p_hh))
	A = A.at[jnp.ix_(H, O)].set((jr.uniform(key_ho, (len(H), out_dims))<p_ho).astype(float))
	A = A.at[jnp.ix_(O, H)].set((jr.uniform(key_oi, (out_dims, len(H)))<p_oh).astype(float))
	A = A.at[jnp.ix_(I, O)].set((jr.uniform(key_io, (in_dims, out_dims))<p_io).astype(float))
	#A = A.at[jnp.ix_(O, I)].set((jr.uniform(key_oi, (out_dims, in_dims))<p_oi).astype(float))

	return A


