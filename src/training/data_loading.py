from torch.utils import data
import jax
import numpy as np


def numpy_collate(batch):
	return jax.tree_map(np.asarray, data.default_collate(batch))
	
class NumpyLoader(data.DataLoader):
	#-------------------------------------------------------------------
	def __init__(self, 	
				 dataset, 
				 batch_size, 
				 shuffle = None, 
				 sampler = None, 
				 batch_sampler = None, 
				 num_workers = 0, 
				 collate_fn = None, 
				 pin_memory = False, 
				 drop_last = False, 
				 timeout = 0,
				 worker_init_fn = None):
		super().__init__(dataset, batch_size, shuffle, sampler,
			batch_sampler, num_workers, collate_fn, pin_memory, 
			drop_last, timeout, worker_init_fn)
	#-------------------------------------------------------------------