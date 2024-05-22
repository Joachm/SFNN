from ast import Call
from typing import Callable, NamedTuple, Optional, Tuple, TypeAlias, Union
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

import gymnax

from brax import envs
from brax.envs import Env
from jaxtyping import Float, PyTree

Params: TypeAlias = PyTree
TaskParams: TypeAlias = PyTree
EnvState: TypeAlias = PyTree
Action: TypeAlias = jax.Array
PolicyState: TypeAlias = PyTree
BraxEnv: TypeAlias = Env
GymEnv: TypeAlias = gymnax.environments.environment.Environment

class State(NamedTuple):
	env_state: EnvState
	policy_state: PolicyState


#=======================================================================
#=======================================================================
#=======================================================================

class GymnaxTask(eqx.Module):
	
	"""
	"""
	#-------------------------------------------------------------------
	statics: PyTree
	env: GymEnv
	env_params: PyTree
	data_fn: Callable
	#-------------------------------------------------------------------

	def __init__(
		self, 
		statics: PyTree,
		env: str,
		env_params: Optional[PyTree] = None,
		data_fn: Callable=lambda d: d):

		self.statics = statics
		self.env, default_env_params = gymnax.make(env) #type: ignore
		self.env_params = env_params if env_params is not None else default_env_params 
		self.data_fn = data_fn

	#-------------------------------------------------------------------

	def __call__(
		self, 
		params: Params, 
		key: jax.Array, 
		task_params: Optional[TaskParams]=None)->Tuple[Float, PyTree]:

		return self.rollout(params, key, task_params)

	#-------------------------------------------------------------------

	def rollout(self, params: Params, key: jax.Array, task_params: Optional[TaskParams]=None)->Tuple[Float, PyTree]:
		"""
		code adapted from: https://github.com/RobertTLange/gymnax/blob/main/gymnax/experimental/rollout.py
		"""
		
		model = eqx.combine(params, self.statics)
		key_reset, key_episode, key_model = jr.split(key, 3)
		obs, state = self.env.reset(key_reset, self.env_params)

		def policy_step(state_input, tmp):
			"""lax.scan compatible step transition in jax env."""
			policy_state, obs, state, rng, cum_reward, valid_mask = state_input
			rng, rng_step, rng_net = jax.random.split(rng, 3)
			
			action, policy_state = model(obs, policy_state, rng_net)
			next_obs, next_state, reward, done, _ = self.env.step(
				rng_step, state, action, self.env_params
			)
			new_cum_reward = cum_reward + reward * valid_mask
			new_valid_mask = valid_mask * (1 - done)
			carry = [
				policy_state,
				next_obs,
				next_state,
				rng,
				new_cum_reward,
				new_valid_mask,
			]
			y = [policy_state, obs, action, reward, next_obs, done]
			return carry, y
			
		policy_state = model.initialize(key_model)
		# Scan over episode step loop
		carry_out, scan_out = jax.lax.scan(
			policy_step,
			[
				policy_state,
				obs,
				state,
				key_episode,
				jnp.array([0.0]),
				jnp.array([1.0]),
			],
			(),
			self.env.default_params.max_steps_in_episode,
		)
		# Return the sum of rewards accumulated by agent in episode rollout
		policy_state, obs, action, reward, _, _ = scan_out
		cum_return = carry_out[-2][0]
		data = {"policy_states": policy_state, "obs": obs, 
				"action": action, "rewards": reward}
		data = self.data_fn(data)
		return cum_return, data

	#-------------------------------------------------------------------


#=======================================================================
#=======================================================================
#=======================================================================

