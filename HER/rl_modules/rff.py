#Use random fourier features for this
import numpy as np
import torch
import math

def create_rff_map(d, r, bandwidth = 10):
	#d: dimension of input vector
	#r: number of features 
	omega = np.random.normal(size=(d, r))
	b = np.random.rand(r)*2*math.pi
	np_rrf_map = lambda x:  np.cos(bandwidth*x@omega + b)/r**.5

	torch_omega = torch.tensor(omega)
	torch_b = torch.tensor(b)
	torch_rrf_map = lambda x:  torch.cos(bandwidth*x@torch_omega + torch_b)/r**.5
	return np_rrf_map, torch_rrf_map
	

# def create_rff_map(d, r, bandwidth = 10):
# 	#d: dimension of input vector
# 	#r: number of features 
# 	omega = np.random.normal(size=(r, d))
# 	# b = np.random.rand(r)*2*math.pi
# 	np_rrf_map = lambda x:  2.71**(-((x-omega)**2/r**.5).sum(axis=-1))#/r**.5

# 	torch_omega = torch.tensor(omega)
# 	# torch_rrf_map = lambda x:  torch.cos(bandwidth*x@torch_omega + torch_b)/r**.5
# 	torch_rrf_map = lambda x:  2.71**(-((x-omega)**2/r**.5).sum(dim=-1))#/r**.5
# 	return np_rrf_map, torch_rrf_map
	
