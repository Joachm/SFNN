from abc import ABC
import abc
from typing import Optional, Tuple
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx
from jaxtyping import Float, PyTree

TaskState = PyTree[...]
TaskParams = PyTree[...]
Params = PyTree[...]
Data = PyTree[...]

class BaseTask(ABC):
	
	"""
	"""
	#-------------------------------------------------------------------
	#-------------------------------------------------------------------

	@abc.abstractmethod
	def __call__(self, params: Params, key:jr.PRNGKeyArray, data: Optional[Data]=None)->Tuple[Float, Data]:
		
		...

	#-------------------------------------------------------------------


class QDTask(ABC):

	#-------------------------------------------------------------------
	
	@abc.abstractmethod
	def __call__(self, 
				 params: Params, 
				 key:jr.PRNGKeyArray, 
				 data: Optional[Data]=None) -> Tuple[Float, Float, Data]:
		
		...

	#-------------------------------------------------------------------

	@abc.abstractproperty
	def n_descriptors(self)->int:
		
		...



