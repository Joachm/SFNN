from src.training.logging import Logger
from src.training.base import BaseMultiTrainer, BaseTrainer

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
from jax.experimental.shard_map import shard_map as shmap
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
import evosax as ex
import equinox as eqx
from typing import Any, Callable, Collection, Dict, Optional, Union, NamedTuple, Tuple
from jaxtyping import Array, Float, PyTree

TrainState = ex.EvoState
Params = PyTree[...]
Data = PyTree[...]
TaskParams = PyTree[...]
Task = Callable


def default_metrics(state, data):
	y = {}
	y["best"] = state.best_fitness
	y["gen_best"] = data["fitness"].min()
	y["gen_mean"] = data["fitness"].mean()
	y["gen_worse"] = data["fitness"].max()
	y["var"] = data["fitness"].var()
	return y


class EvosaxTrainer(BaseTrainer):

	"""
	"""
	#-------------------------------------------------------------------
	strategy: ex.Strategy
	es_params: ex.EvoParams
	params_shaper: ex.ParameterReshaper
	task: Task
	fitness_shaper: ex.FitnessShaper
	n_devices: int
	multi_device_mode: str
	#-------------------------------------------------------------------

	def __init__(
		self, 
		train_steps: int,
		strategy: Union[ex.Strategy, str],
		task: Callable,
		params_shaper: ex.ParameterReshaper,
		popsize: Optional[int]=None,
		fitness_shaper: Optional[ex.FitnessShaper]=None,
		es_kws: Optional[Dict[str, Any]]={},
		es_params: Optional[ex.EvoParams]=None,
		eval_reps: int=1,
		logger: Optional[Logger]=None,
	    progress_bar: Optional[bool]=True,
	    n_devices: int=1,
	    multi_device_mode: str="shmap"):

		super().__init__(train_steps=train_steps, 
						 logger=logger, 
						 progress_bar=progress_bar)
		
		if isinstance(strategy, str):
			assert popsize is not None
			self.strategy = self.create_strategy(strategy, popsize, params_shaper.total_params, **es_kws) # type: ignore
		else:
			self.strategy = strategy

		if es_params is None:
			self.es_params = self.strategy.default_params
		else:
			self.es_params = es_params

		self.params_shaper = params_shaper

		if eval_reps > 1:
			def _eval_fn(p: Params, k: jr.PRNGKeyArray, tp: Optional[PyTree]=None):
				"""
				"""
				fit,ind_env_fit, info = jax.vmap(task, in_axes=(None,0,None))(p, jr.split(k,eval_reps), tp)
				return jnp.mean(fit), jnp.mean(ind_env_fit, axis=0), info
			self.task = _eval_fn
		else :
			self.task = task

		if fitness_shaper is None:
			self.fitness_shaper = ex.FitnessShaper()
		else:
			self.fitness_shaper = fitness_shaper

		self.n_devices = n_devices
		self.multi_device_mode = multi_device_mode

	#-------------------------------------------------------------------

	def eval(self, *args, **kwargs):
		
		if self.n_devices == 1:
			return self._eval(*args, **kwargs)
		if self.multi_device_mode=="shmap":
			return self._eval_shmap(*args, **kwargs)
		elif self.multi_device_mode == "pmap":
			return self._eval_pmap(*args, **kwargs)
		else:
			raise ValueError(f"multi_device_mode {self.multi_device_mode} is not a valid mode")

	#-------------------------------------------------------------------

	def _eval(self, x: jax.Array, key: jax.Array, task_params: PyTree)->Tuple[jax.Array, PyTree]:
		
		#params = self.params_shaper.reshape(x)
		#_eval = jax.vmap(self.task, in_axes=(0, None, None))
		#return _eval(params, key, task_params)

		params = self.params_shaper.reshape(x)
		_eval = jax.vmap(self.task, in_axes=(0, 0, None)) 
		keys = jr.split(key, x.shape[0])
		return _eval(params, keys, task_params)
	#-------------------------------------------------------------------

	def _eval_shmap(self, x: jax.Array, key: jax.Array, task_params: PyTree)->Tuple[jax.Array, PyTree]:
		
		devices = mesh_utils.create_device_mesh((self.n_devices,))
		device_mesh = Mesh(devices, axis_names=("p"))

		_eval = lambda x, k: self.task(self.params_shaper.reshape_single(x), k)
		batch_eval = jax.vmap(_eval, in_axes=(0,None))
		sheval = shmap(batch_eval, 
					   mesh=device_mesh, 
					   in_specs=(P("p",), P()),
					   out_specs=(P("p"), P("p")),
					   check_rep=False)

		return sheval(x, key)

	#-------------------------------------------------------------------

	def _eval_pmap(self, x: jax.Array, key: jax.Array, data: PyTree)->Tuple[jax.Array, PyTree]:
		
		_eval = lambda x, k: self.task(self.params_shaper.reshape_single(x), k)
		batch_eval = jax.vmap(_eval, in_axes=(0,None))
		pop_batch = x.shape[0] // self.n_devices
		x = x.reshape((self.n_devices, pop_batch, -1))
		pmapeval = jax.pmap(batch_eval, in_axes=(0,None)) #type: ignore
		f, eval_data = pmapeval(x, key)
		return f.reshape((-1,)), eval_data

	#-------------------------------------------------------------------

	def train_step(self, state: TrainState, key: jr.PRNGKeyArray, data: Optional[TaskParams]=None) -> Tuple[TrainState, Data]:
		
		ask_key, eval_key = jr.split(key, 2)
		x, state = self.strategy.ask(ask_key, state, self.es_params)
		fitness, ind_env_fit, eval_data = self.eval(x, eval_key, data)
		ind_env_fit = jnp.mean(ind_env_fit, axis=0)
		f = self.fitness_shaper.apply(x, fitness)
		state = self.strategy.tell(x, f, state, self.es_params)
		state = self._update_evo_state(state, x, fitness)
		print('SHAPE',ind_env_fit.shape)
		if ind_env_fit.shape[0] == 3:
			return state, {"fitness": fitness, "data": eval_data, 'cart_fit':ind_env_fit[0], 'acro_fit':ind_env_fit[1], 'mount_fit':ind_env_fit[2]} #TODO def best as >=
		elif ind_env_fit.shape[0] == 2:
			return state, {"fitness": fitness, "data": eval_data, 'cart_fit':ind_env_fit[0], 'mount_fit':ind_env_fit[1]} #TODO def best as >=
		else:
			return state, {"fitness": fitness, "data": eval_data, 'cart_fit':ind_env_fit} #TODO def best as >=



	#-------------------------------------------------------------------

	def _update_evo_state(self, state: TrainState, x: jax.Array, f: jax.Array)->TrainState:
		is_best = f.min() <= state.best_fitness
		gen_best = x[jnp.argmin(f)]
		best_member = jax.lax.select(is_best, gen_best, state.best_member)
		state = state.replace(best_member=best_member) #type:ignore
		return state

	#-------------------------------------------------------------------

	def initialize(self, key: jr.PRNGKeyArray, **kwargs) -> TrainState:
		
		state = self.strategy.initialize(key, self.es_params)
		state = state.replace(**kwargs)
		return state

	#-------------------------------------------------------------------

	def create_strategy(self, name: str, popsize: int, num_dims: int, **kwargs)->ex.Strategy:
		
		ES = getattr(ex, name)
		es = ES(popsize=popsize, num_dims=num_dims, **kwargs)
		return es

	#-------------------------------------------------------------------

	def load_ckpt(self, ckpt_path: str)->Params:
		params = eqx.tree_deserialise_leaves(
			ckpt_path, jnp.zeros((self.params_shaper.total_params,))
		)
		return params

	#-------------------------------------------------------------------

	def train_from_model_ckpt(self, ckpt_path: str, key: jax.Array)->Tuple[TrainState, Data]: #type:ignore
		
		key_init, key_train = jr.split(key)
		params = self.load_ckpt(ckpt_path)
		state = self.initialize(key_init, mean=self.params_shaper.flatten_single(params))
		return self.train(state, key_train)

	#-------------------------------------------------------------------

	def train_from_model_ckpt_(self, ckpt_path: str, key: jax.Array)->TrainState:#type:ignore
		
		key_init, key_train = jr.split(key)
		params = self.load_ckpt(ckpt_path)
		state = self.initialize(key_init, mean=self.params_shaper.flatten_single(params))
		return self.train_(state, key_train)


	def train_from_loaded(self,  key: jax.Array, loaded_params:jax.Array)->TrainState:#type:ignore
		
		key_init, key_train = jr.split(key)
		state = self.initialize(key_init, mean=self.params_shaper.flatten_single(loaded_params))
		return self.train_(state, key_train)




def default_transform(prev_state: TrainState, new_state: TrainState)->TrainState:
	return new_state.replace(mean=prev_state.best_member, best_member=prev_state.best_member,# type:ignore
							 best_fitness=prev_state.best_fitness) 

class MultiESTrainer(BaseMultiTrainer):
	"""
	Implements evolutionary training with multiple possibly different trainers
	Params:
		trainers: list of evolutionary trainers
		transform_fns: function or list of functions mapping the final 
			tarining state of th ith trainer and the initial sampled state of the 
			i+1th trainer to the actual initial state of the i+1th trainer. 
			By default initialize the distribution mean with previous best member 
	"""
	#-------------------------------------------------------------------
	#-------------------------------------------------------------------

	def __init__(self, 
				 trainers: list[EvosaxTrainer], 
				 transform_fns: Union[list[Callable[[TrainState, TrainState], TrainState]], 
				 					  Callable[[TrainState, TrainState], TrainState]]=default_transform):
		super().__init__(trainers, transform_fns) #type:ignore









