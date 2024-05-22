from typing import Callable, Tuple
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx
from jaxtyping import PyTree

State = PyTree[...]
Params = PyTree[...]
Statics = PyTree[...]

class BaseModel(eqx.Module):
	
	"""
	"""
	#-------------------------------------------------------------------
	#-------------------------------------------------------------------

	def dna_partition(self):
		raise NotImplementedError("This model has no dna partition")

	#-------------------------------------------------------------------
	
	def partition(self):
		return eqx.partition(self, eqx.is_array)

	#-------------------------------------------------------------------

	def save(self, filename: str):
		eqx.tree_serialise_leaves(filename, self)

	#-------------------------------------------------------------------

	def load(self, filename: str):
		return eqx.tree_deserialise_leaves(filename, self)


class DevelopmentalModel(BaseModel):
	
	"""
	Base structure for iterative developmental models
	"""
	#-------------------------------------------------------------------
	#-------------------------------------------------------------------

	def __init__(self):
		pass

	#-------------------------------------------------------------------

	def __call__(self, state: State, key: jr.PRNGKeyArray)->State:
		
		raise NotImplementedError("__call__ method not implemented")

	#-------------------------------------------------------------------

	def initialize(self, key: jr.PRNGKeyArray)->State:
		raise NotImplementedError("initialize method not implemented")

	#-------------------------------------------------------------------

	def init_and_rollout(self, key: jr.PRNGKeyArray, steps: int)->State:

		key_init, key_rollout = jr.split(key)
		state = self.initialize(key_init)
		return self.rollout(state, key_rollout, steps)

	#-------------------------------------------------------------------

	def rollout(self, state: State, key: jr.PRNGKeyArray, steps: int)->Tuple[State, State]:

		def _step(c, x):
			s, k = c
			k, k_ = jr.split(k)
			s = self.__call__(s, k_)
			return [s,k], s

		[state, _], states = jax.lax.scan(
			_step, [state, key], None, steps
		)

		return state, states

	#-------------------------------------------------------------------

	def init_and_rollout_(self, key: jr.PRNGKeyArray, steps: int)->State:

		key_init, key_rollout = jr.split(key)
		state = self.initialize(key_init)
		return self.rollout_(state, key_rollout, steps)

	#-------------------------------------------------------------------

	def rollout_(self, state: State, key:jr.PRNGKeyArray, steps: int)->State:

		def _step(i, sk):
			s, k = sk
			k, k_ = jr.split(k)
			s = self.__call__(s, k_)
			return [s, k]

		[state, _] = jax.lax.fori_loop(0, steps, _step, [state, key])
		return state

	#-------------------------------------------------------------------



class DNABasedModel(eqx.Module):
	
	"""
	"""
	#-------------------------------------------------------------------
	#-------------------------------------------------------------------

	def initialize(self, dna: PyTree, key: jr.PRNGKeyArray)->State:
		raise NotImplementedError

		
