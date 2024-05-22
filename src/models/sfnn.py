import sys
sys.path.insert(0, '../../')

from collections import namedtuple
from typing import Callable, NamedTuple, Optional, Tuple, Union, TypeAlias

from jaxtyping import Float, Int, Array
from scripts.ndp_mod.model import reservoir

from src.nn.layers import Neuron

from src.training.evolution import EvosaxTrainer
from src.training.logging import Logger

from src.tasks.rl import GymnaxTask

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx
import equinox.nn as nn
import evosax as ex

RecurrentNeuralNetwork: TypeAlias = Union[nn.GRUCell, Neuron]

class Network(NamedTuple):
	types: Int[Array, "N"]   # Node types
	e: Float[Array, "N N de"]# Edges hidden states
	A: Float[Array, "N N"]   # Adjacency matrix
	r: Float   				 # reward
	y_prev:Float[Array, 'N de']
	a_prev: Int

class SFNN(eqx.Module):
	"""
	Structurally Flexible Neural Network
	"""
	#-------------------------------------------------------------------
	# Parameters:
	node_cells: RecurrentNeuralNetwork
	edge_cells: RecurrentNeuralNetwork
	# Statics:
	de: int
	n_types: int
	n_nodes: int
	action_dims: int
	input_dims: int
	learning_rate: Array
	#-------------------------------------------------------------------

	def __init__(self, msg_dims: int, n_types: int, n_nodes: int, 
				input_dims:int, action_dims:int, cell_type: str="gru", activation: Callable=jnn.tanh, *, key: jax.Array):


		self.de = msg_dims
		self.n_types = n_types
		self.n_nodes = n_nodes
		self.action_dims = action_dims
		self.input_dims = input_dims

		key_ncells, key_nout, key_ecells = jr.split(key, 3)
		
		def init_node_cell(key):
			if cell_type == "gru":
				#return nn.GRUCell(msg_dims, hidden_dims, key=key)
				return Neuron(msg_dims, msg_dims, key=key)
			elif cell_type == "mgu":
				return MGU(msg_dims, hidden_dims, key=key)
			elif cell_type == "rnn":
				return RNN(hidden_dims, msg_dims, key=key)
			elif cell_type == "seq":
				return nn.Sequential([nn.Linear(msg_dims,hidden_dims, key=key), nn.Lambda(activation)])

			else : 
				raise ValueError(f"{cell_type} is not a known or managed cell type")

		self.node_cells = jax.vmap(init_node_cell)(jr.split(key_ncells, n_types))
		self.learning_rate = jnp.array([1.])
		#self.probabilities = jnp.array([1.,1.,1.,1.,1.])

		def init_edge_cell(key):
			in_dims = 2*msg_dims+1
			out_dims=msg_dims
			if cell_type == "gru":
				return nn.GRUCell(in_dims, out_dims, key=key)
			elif cell_type == "mgu":
				return MGU(in_dims, out_dims, key=key)
			elif cell_type == "rnn":
				return RNN(in_dims, out_dims, key=key)
			else : 
				raise ValueError(f"{cell_type} is not a known or managed cell type")

		self.edge_cells = jax.vmap(init_edge_cell)(jr.split(key_ecells, n_types))

	#-------------------------------------------------------------------

	def __call__(self, obs: jax.Array, net: Network, key: Optional[jax.Array]=None) -> Tuple[Int, Network]:
		"""
		TODO: 
			add obs as input to the network
		"""

		A = net.A	  # N x N
		e = net.e 	  # N x N x M
		N = e.shape[0]
		node_types = net.types # N: int
		edge_types = jnp.repeat(node_types[:,None], N, axis=1) # N x N: int

		m = net.y_prev[:,None,:] * (e * A[...,None])  	  # N x N x M
		x = m.sum(0)
		x= x.at[:self.input_dims, :].set(obs[:,None])
		y_pre = jax.vmap(self._apply_node_cell)(x, node_types) # N x H
		a_prev = jnp.zeros((self.action_dims,)).at[net.a_prev].set(1)
		x= x.at[-self.action_dims:, 0].set(a_prev)
		for _ in [0,1]:

			# 1. compute and aggregate signals
			m = y_pre[:, None,:] * (e * A[...,None])  	  # N x N x M
			x = m.sum(0) 	
			x = x.at[:self.input_dims, :].set(obs[:,None])
			x = x.at[-self.action_dims:, 0].set(a_prev)

			y_post = jax.vmap(self._apply_node_cell)(x, node_types) # N x H
			# 3. Update edge states
			y_pre = y_post
		yiyjr = jnp.concatenate(	# N x N x 2M+1
				[
					jnp.repeat(net.y_prev[:,None], N, axis=1),
			 		jnp.repeat(y_post[None,:], N, axis=0),
			 		jnp.ones((N,N,1)) *net.r
		 		],
			 	 axis=-1
			)
		e = e + 0.01*self.learning_rate*jax.vmap(jax.vmap(self._apply_edge_cell))(yiyjr, e, edge_types) # N x N x M

		# Get action
		a = y_pre[-self.action_dims:, 0]
		a = jnp.argmax(a)
		return a, net._replace(e=e, y_prev=y_pre, a_prev=a)

	#-------------------------------------------------------------------

	def _apply_node_cell(self, x: jax.Array, typ: Int):
		cell = jax.tree_map(lambda x: x[typ] if eqx.is_array(x) else x, self.node_cells)
		y = cell(x)
		return y

	def _apply_edge_cell(self, x: jax.Array, e: jax.Array, typ: Int):
		cell = jax.tree_map(lambda x: x[typ] if eqx.is_array(x) else x, self.edge_cells)
		return cell(x, e)

	#-------------------------------------------------------------------

	def initialize(self, key: jax.Array)->Network:
		key_types, key_A, key_h, key_e = jr.split(key,4)
		e = jr.uniform(key_e,(self.n_nodes, self.n_nodes, self.de), minval=-.1, maxval=.1)
		types = jr.randint(key_types, (self.n_nodes,), 1, self.n_types-1).at[-self.action_dims:].set(self.n_types-1)
		types = types.at[:self.input_dims].set(0)  #set input nodes to distinct node type
		A = reservoir(key_A, self.n_nodes, self.input_dims, self.action_dims, .5, .5, .5, .5, .0)
		y_prev = jnp.zeros((self.n_nodes, self.de))
		return Network(e=e, types=types, A=A, r=0., y_prev=y_prev, a_prev=0)

#=======================================================================

class Config(NamedTuple):
	seed: int=1
	# --- Model ---
	N: int = 32 # number of nodes
	de: int = 2 # edge state features
	n_types: int = 2 # Number of distinct node types
	# --- Env ---
	env_name: str = "CartPole-v1"
	env_lib: str = "gymnax"
	# ---Optimizer ---
	strategy: str = "CMA_ES"
	popsize: int = 64
	generations: int = 2048
	wandb_log: bool=True
	eval_reps: int=3 #Number of evaluations ftiness is averaged over (monte carlo samplings)

default_config = Config()

#=======================================================================













