from src.tasks.rl import *

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
from jaxtyping import PyTree
from typing import Callable, Optional


class EnvManager:
	
	#envs = list

	def __init__(
		self,
		envs:[],
		):

		self.envs = envs
	
	def __call__(
		self, 
		params: Params, 
		key: jax.Array, 
		task_params: Optional[TaskParams]=None):

		norm_fitness = jnp.zeros(len(self.envs), dtype=float)
		fitnesses = jnp.zeros(len(self.envs), dtype=float)
		for idx, env in enumerate(self.envs):
			norm_score, score, data = env(params=params, key=key)
			
			norm_fitness = norm_fitness.at[idx].set(norm_score)
			fitnesses = fitnesses.at[idx].set(score)

		return norm_fitness.prod(), fitnesses, data

	



class MultiEpisodeTask(GymnaxTask):
	
	"""
	"""
	#-------------------------------------------------------------------
	n_episodes: int
	max_score: float
	min_score: float
	#-------------------------------------------------------------------

	def __init__(
		self, 
		statics: PyTree,
		env: str,
		n_episodes: int=8,
		max_score: int=100,
		min_score: int=0,
		env_params: Optional[PyTree] = None,
		data_fn: Callable=lambda d: d):
		
		super().__init__(statics, env, env_params, data_fn)

		self.n_episodes = n_episodes
		self.max_score = max_score
		self.min_score = min_score

	#-------------------------------------------------------------------

	def __call__(
		self, 
		params: Params, 
		key: jax.Array, 
		task_params: Optional[TaskParams]=None):

		policy_state=None
		full_return = jax.numpy.arange(0,self.n_episodes, dtype=float)
		normalized_return = jax.numpy.arange(0,self.n_episodes, dtype=float)
		weights = jax.numpy.arange(0,self.n_episodes)
		
		for episode in range(self.n_episodes):
			key, key_ = jr.split(key)
			policy_state, episode_return, data = self._rollout(params, key_, policy_state)
			full_return = full_return.at[episode].set(episode_return)
			normed_val = 0 + ((1-0)/(self.max_score-self.min_score))*(episode_return-self.min_score)
			normalized_return = normalized_return.at[episode].set(normed_val)

            

		#normalized_returns = (full_return-self.min_score)/(self.max_score-self.min_score)
		weights = weights / weights.sum(0)
		norm_weighted_mean = jax.numpy.dot(normalized_return, weights)
		not_norm_weighted_mean = jax.numpy.dot(full_return, weights)
		return norm_weighted_mean, not_norm_weighted_mean, data
	#-------------------------------------------------------------------

	def _rollout(self, params: Params, key: jax.Array, policy_state: Optional[PolicyState]=None):
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
			policy_state = policy_state._replace(r=reward)
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
		
		if policy_state is None:
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
		policy_states, obs, action, reward, _, _ = scan_out
		policy_state, *_ = carry_out
		cum_return = carry_out[-2][0]
		data = {"policy_states": policy_states, "obs": obs, 
				"action": action, "rewards": reward}
		data = self.data_fn(data)
		return policy_state, cum_return, data

def make(config, statics):
	data_fn = lambda *_, **kws: None
	return MultiEpisodeTask(statics, n_episodes=config.n_episodes, env=config.env_name, max_score=config.max_score, min_score=config.min_score,  data_fn=data_fn)
