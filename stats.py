import numpy as np
from scipy.stats import beta

def wilson_interval(array: np.ndarray, z: float) -> tuple: 
	ns = array.sum()
	nf = (1-array).sum()
	n = nf + ns
	raise NotImplementedError

def bayes_interval(n_successes, n_failures, z: float = .95) -> tuple:
	prior_alpha = .5
	prior_beta = .5

	rv = beta.interval(z, prior_alpha + n_successes, prior_beta + n_failures)
	# import pdb
	# pdb.set_trace()
	# return (0,0)
	return rv#(round(rv[0], 4), round(rv[1], 4))
