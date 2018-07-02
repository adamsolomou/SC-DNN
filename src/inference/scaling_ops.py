import math 
import numpy as np 

def scalar_next_int_power_of_2(x):
	"""
	Return the next integer power of 2
	"""
	return 1<<(math.ceil(abs(x))-1).bit_length()

def next_int_power_of_2(x):
	"""
	Vectorized form of scalar_next_int_power_of_2
	""" 
	return np.vectorize(scalar_next_int_power_of_2)(x)

def sc_matmul_scaling(S_in,W):
	"""
	The function computes the output scalings of a stochastic computing 
	based matrix multiplication of the form X*W, where X is a 3D numpy 
	array of stochastic bit-streams and W is a 2D numpy array of weights. 

	Parameters 
	----------
	S_in: 2D numpy array with the same dimensions as X. Each entry holds the scaling 
	of the corresponding bit-stream in X. 
	W: 2D numpy array with floating point weights 

	Returns
	------- 
	A 2D numpy array with the scalings of the output bit-streams

	Raises
	------
	A ValueError if there is a dimension mismatch 
	"""
	(w_rows,w_cols) = W.shape
	(s_rows,s_cols) = S_in.shape

	# Check that dimensions match 
	if(s_cols != w_rows):
		raise ValueError('Dimension mismatch in sc_matmul_scaling')	

	# Find the max scaling from each row and replicate the resulting array
	S_max = np.amax(S_in,axis=1)
	S_max = np.repeat(S_max,w_cols,axis=0).reshape(s_rows,w_cols)

	W_sum = np.sum(abs(W),axis=0)
	W_sum = np.tile(W_sum,(s_rows,1))

	S_out = (S_max*W_sum)

	# Scale up each number to the next integer power of 2
	S_out = next_int_power_of_2(S_out)

	return S_out.astype(int) 

def add_scale(s1,s2):
	"""
	Parameters
	----------
	s1,s2: Input scaling parameters

	Returns
	-------
	The output scaling of a two input scaled added
	"""
	return max(s1,s2)*2

def v_add_scale(S1,S2):
	"""
	Vectorized form of add_scale() function
	"""
	return np.vectorize(add_scale)(S1,S2)

def sc_matvec_add_scaling(S_in,b):
	"""
	The function computes the output scalings of a stochastic computing 
	based matrix addition of the form X + b, where X is a 3D numpy array
	of stochastic bit-streams and b is a 1D numpy array with biases.


	Parameters 
	----------
	S_in: 2D numpy array with the same dimensions as X. Each entry holds 
	the scaling of the corresponding bit-stream in X. 
	W: 1D numpy array with floating point bias terms

	Returns
	------- 
	A 2D numpy array with the scalings of the output bit-streams

	Raises
	------
	A ValueError if there is a dimension mismatch 
	"""
	n_biases = b.size
	(s_rows,s_cols) = S_in.shape

	# Check that dimensions match 
	if(s_cols != n_biases):
		raise ValueError('Dimension mismatch in sc_matadd_scaling')

	# Compute the scaling of each bias coefficient
	S_b = next_int_power_of_2(b)

	# Replicate the vector of biases to a 2D array with dimensions (s_rows,s_cols) 
	S_b = np.tile(S_b,(s_rows,1))

	# Calculate the 2D array with output scalings 
	S_out = v_add_scale(S_in,S_b)

	return S_out.astype(int)

def relu(S):
	"""
	Returns the output scaling of a ReLU 
	"""
	return S.astype(int)