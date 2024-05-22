import wandb
import jax
import jax.numpy as jnp
from jaxtyping import PyTree
from typing import Tuple, TypeAlias, Callable, Optional
from jax.experimental import host_callback as hcb
import equinox as eqx
import os

TrainState: TypeAlias = PyTree[...]
Data: TypeAlias = PyTree[...]

class Logger:

	#-------------------------------------------------------------------

	def __init__(
		self, 
		wandb_log: bool,
		metrics_fn: Callable[[TrainState, Data], Tuple[Data, Data, int]], 
		ckpt_file: Optional[str]=None, 
		ckpt_freq: int=100,
		verbose: bool=False):
		
		if ckpt_file is not None and "/" in ckpt_file:
			if not os.path.isdir(ckpt_file[:ckpt_file.rindex("/")]):
				os.makedirs(ckpt_file[:ckpt_file.rindex("/")])
		self.wandb_log = wandb_log
		self.metrics_fn = metrics_fn
		self.ckpt_file = ckpt_file
		self.ckpt_freq = ckpt_freq
		self.epoch = [0]
		self.verbose = verbose

	#-------------------------------------------------------------------

	def log(self, state: TrainState, data: Data):
		
		log_data, ckpt_data, epoch = self.metrics_fn(state, data)
		if self.wandb_log:
			self._log(log_data)
		self.save_chkpt(ckpt_data, epoch)
		return log_data

	#-------------------------------------------------------------------

	def _log(self, data: dict):
		hcb.id_tap(
			lambda d, *_: wandb.log(d), data
		)

	#-------------------------------------------------------------------

	def save_chkpt(self, data: dict, epoch: int):

		def save(data):
			assert self.ckpt_file is not None
			file = f"{self.ckpt_file}.eqx"
			if self.verbose:
				print("saving data at: ", file)
			eqx.tree_serialise_leaves(file, data)

		def tap_save(data):
			hcb.id_tap(lambda d, *_: save(d), data)
			return None

		if self.ckpt_file is not None:
			jax.lax.cond(
				(jnp.mod(epoch, self.ckpt_freq))==0,
				lambda data : tap_save(data),
				lambda data : None,
				data
			)

	#-------------------------------------------------------------------

	def wandb_init(self, project: str, config: dict, **kwargs):
		if self.wandb_log:
			wandb.init(project=project, config=config, **kwargs)

	#-------------------------------------------------------------------

	def wandb_finish(self, *args, **kwargs):
		if self.wandb_log:
			wandb.finish(*args, **kwargs)
