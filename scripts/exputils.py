import argparse

def load_config(factory):
	default_config = factory()
	parser = argparse.ArgumentParser()
	bools = []
	for k, v in default_config._asdict().items():
		dtype = int if isinstance(v, bool) else type(v)
		dv = int(v) if isinstance(v, bool) else v
		if isinstance(v, bool): bools.append(k)
		parser.add_argument(f"--{k}", type=dtype, default=dv)
	config = vars(parser.parse_args())
	for k in bools:
		config[k] = bool(config[k])
	config = factory(**config)
	return config