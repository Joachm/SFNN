import sys
sys.path.insert(0, '../../')

from typing import NamedTuple
from scripts.exputils import load_config
import pprint

import jax.random as jr

from scripts.sfnn.model import make as model_factory
from scripts.sfnn.trainer import make as trainer_factory
from scripts.sfnn.task import make as task_factory
from scripts.sfnn.task import EnvManager

import equinox as eqx

N_EPISODES = 8
MSG_DIMS= 4
N_NODES = 32
N_TYPES = 3
SEED = 100

class Config(NamedTuple):
	seed: int=SEED
	project: str="sfnn_multi"
	baseline: int=0
	# --- trainer ---
	generations: int=10000
	popsize: int=128
	strategy: str="CMA_ES"
	ckpt_file: str=""
	eval_reps: int=1
	log: int=1
	env_name: str='multi_env'

	# --- task ---
	n_episodes: int= N_EPISODES
	# --- model ---
	msg_dims: int=MSG_DIMS
	n_types: int=N_TYPES
	n_nodes: int=N_NODES




class ConfigCart(NamedTuple):
	seed: int=SEED
	#project: str="sfnn_cart2"

	# --- task ---
	n_episodes: int=N_EPISODES
	env_name: str="CartPole-v1"
	max_score = 500
	min_score = 8
	# --- model ---
	msg_dims: int=MSG_DIMS
	n_types: int=N_TYPES
	n_nodes: int=N_NODES
	obs_dims: int=4
	action_dims: int=2



class ConfigAcro(NamedTuple):
	seed: int=SEED
	#project: str="sfnn_cart2"

	# --- task ---
	n_episodes: int=N_EPISODES
	env_name: str="Acrobot-v1"
	max_score = -80
	min_score = -500 
	# --- model ---
	msg_dims: int=MSG_DIMS
	n_types: int=N_TYPES
	n_nodes: int=N_NODES
	obs_dims: int=6
	action_dims: int=3



class ConfigMount(NamedTuple):
	seed: int=SEED
	#project: str="sfnn_cart2"

	# --- task ---
	n_episodes: int=N_EPISODES
	env_name: str="MountainCar-v0"
	max_score = -110
	min_score = -200
	# --- model ---
	msg_dims: int=MSG_DIMS
	n_types: int=N_TYPES
	n_nodes: int=N_NODES
	obs_dims: int=2
	action_dims: int=3



if __name__ == '__main__':

	config = load_config(Config)
	pprint.pprint(config._asdict())
	key = jr.key(config.seed)
	model_key, train_key = jr.split(key)

################ init Cart model and task #############################

	config_cart = load_config(ConfigCart)
	#pprint.pprint(config_cart._asdict())


	model_cart = model_factory(config_cart, key)
	params, statics_cart = eqx.partition(model_cart, eqx.is_array)

	task_cart = task_factory(config_cart, statics_cart)


################ init Acro model and task #############################

	config_acro = load_config(ConfigAcro)
	#pprint.pprint(config_acro._asdict())


	model_acro = model_factory(config_acro, key)
	params, statics_acro = eqx.partition(model_acro, eqx.is_array)

	task_acro = task_factory(config_acro, statics_acro)


################ init Mount model and task #############################

	config_mount = load_config(ConfigMount)
	#pprint.pprint(config_mount._asdict())


	model_mount = model_factory(config_mount, key)
	params, statics_mount = eqx.partition(model_mount, eqx.is_array)

	task_mount = task_factory(config_mount, statics_mount)



################# gather environements ########################3
	
	task = EnvManager([task_cart, task_acro, task_mount])


	trainer = trainer_factory(config, task, params)
	if config.log:
		trainer.logger.wandb_init(config.project, config._asdict()) #type:ignore
	trainer.init_and_train_(train_key)
	if config.log:
		trainer.logger.wandb_finish() #type:ignore




