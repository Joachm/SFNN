from src.models.sfnn import SFNN
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn

def make(config, key):
	return SFNN(
				config.msg_dims,
				config.n_types,
				config.n_nodes,
				config.obs_dims,
				config.action_dims,
				key=key)

if __name__ == '__main__':
	
	sfnn = SFNN(4, 1, 3, 8, 4, 2, key=jr.key(1))
	state = sfnn.initialize(jr.key(2))
	key = jr.key(2)
	for i in range(10):
		key, key_p, key_o = jr.split(key, 3)
		obs = jr.uniform(key_o, (4,), minval=-1, maxval=1)
		action, state = sfnn(obs, state, key_p)
