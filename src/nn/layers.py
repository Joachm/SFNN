import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx
import equinox.nn as nn




class Neuron(eqx.Module):
	
	#-------------------------------------------------------------------
	#gru: nn.GRUCell
	out: nn.Linear
	#-------------------------------------------------------------------

	def __init__(self, input_dims: int, output_dims: int, *, key: jax.Array):

		_, out = jr.split(key)
		self.out = nn.Linear(input_dims, output_dims, use_bias=False, key=out)

	#-------------------------------------------------------------------

	def __call__(self, x: jax.Array):
		
		y = jnn.tanh(self.out(x))
		return y

