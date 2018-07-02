import math
import numpy as np 
import multiprocessing

"""
Stochastic bit-streams are represented as NumPy arrays. Multi-dimensional arrays 
are used to encode numerical vectors and matrices. 
"""

def scalar_next_int_power_of_2(x):
    """
    Return the next integer power of 2 of x
    """
	# return 1<<(int(math.ceil(abs(x))-1)).bit_length()
	return 2**(math.ceil(math.log(abs(x), 2)))

def next_int_power_of_2(x):
    """
    Vectorized form of scalar_next_int_power_of_2
    """
	return np.vectorize(pyfunc=scalar_next_int_power_of_2,
						otypes=[np.float32])(x)

def sng(x,no_stoch_samples):
	"""
    Bipolar stochastic number generator 

	Parameters
	----------
	x: Floating-point input value 
	no_stoch_samples: Bit-stream length (int)

	Returns 
	-------
	1D numpy array holding the bit-stream representing the input x
	"""

	# Draw samples from a uniform distribution
	r = np.random.uniform(-1.0,1.0,size=no_stoch_samples)

	# Bit-wise comparison
	y = x > r

	return y.astype(np.int8)

def vec_sng(x,no_stoch_samples):
	"""
	Vectorised form of sng

	Parameters
	----------
	x: 1D numpy array of floating-point input values
    no_stoch_samples: Bit-stream length

	Returns 
	-------
    2D numpy array holding the bit-streams representing the values in x
	"""

	# Initialise the output 
	y = np.empty((x.size,no_stoch_samples),dtype=np.int8)

	for idx in range(x.size):
		y[idx,:] = sng(x[idx],no_stoch_samples)

	return y.astype(np.int8)

def mat_sng(x,no_stoch_samples):
	"""
    Vectorised form of sng

    Parameters
    ----------
    x: 2D numpy array of floating-point input values
    no_stoch_samples: Bit-stream length

    Returns 
    -------
    2D numpy array holding the bit-streams representing the values in x
    """
	rows = x.shape[0]
	cols = x.shape[1]

	y = np.empty((rows,cols,no_stoch_samples),dtype=np.int8)

	for row_idx in range(rows):
		for col_idx in range(cols):
			y[row_idx,col_idx,:] = sng(x[row_idx,col_idx],no_stoch_samples)

	return y.astype(np.int8)

def get_value(x):
	"""
    Estimate the value of a stochastic bit-stream 

    Parameters 
    ----------
	x: 1D numpy array holding the input bit-stream

    Returns 
    -------
    The value encoded by the bit-stream 
	"""
	y = 2*np.mean(x) - 1

	return np.array(y.astype(np.float32))

def vec_sc_value(x): 
	"""
	Vectorised version of the get_value function 

    Parameters 
    ----------
	x: 2D numpy array with stochastic bit-streams
	"""
	no_inputs = x.shape[0]

	# Initialise the output variable
	y = np.empty(no_inputs,dtype=np.float32)

	for idx in range(no_inputs):
		y[idx] = get_value(x[idx,:])

	return y

def mat_sc_value(x):
	"""
    Vectorised version of the get_value function 

    Parameters
    ----------
	x: 3D numpy array with stochastic bit-streams
	"""
	rows = x.shape[0]
	cols = x.shape[1]

	# Initialise the output variable 
	y = np.empty((rows,cols),dtype=np.float32)

	for row_idx in range(rows):
		for col_idx in range(cols):
			y[row_idx,col_idx] = get_value(x[row_idx,col_idx,:])

	return y

def multiply(x,w):
	"""
    Stochastic XNOR multiplier 

    Parameters
    ----------
	x: Stochastic bit-stream
	w: Floating point weight 

    Returns 
    -------
    A bit-stream representing the product of x and w
	"""
	w_seq = sng(w,x.size)

	product = np.logical_not(np.logical_xor(x,w_seq))

	return np.array(product.astype(np.int8))

def square(x): 
	# Stochastic squarring function 
	x_shifted = np.roll(x,shift=1)

	sq = np.logical_not(np.logical_xor(x,x_shifted))

	return np.array(sq.astype(np.int8))

def invert(x): 
	# Change of sign 

	return np.logical_not(x).astype(np.int8)

def add(x,y,s_in,upscale=False,s_out=None):
	"""
    Scaled addition of two stochastic bit-streams 

    Parameters
    ----------
	x,y: 1D numpy arrays holding bit-streams to be added
	s_in: 1D numpy array holding the scalings of x and y respectively
	upsale: Boolean indicating whether saturation arithmetic should be applied

    Returns 
    -------
    A 1D numpy array holding the output bit-stream

	"""
	x_rescaled = np.empty(x.size, dtype=np.int8)
	y_rescaled = np.empty(y.size, dtype=np.int8)

	# Rescaling of the input bit-streams
	if (s_in[0] > s_in[1]):
		""" Rescale y """
		max_scaling = s_in[0]
		s_ratio = s_in[1]/s_in[0]

		x_rescaled = x
		y_rescaled = multiply(y,s_ratio)

	elif (s_in[0] < s_in[1]):
		""" Rescale x """
		max_scaling = s_in[1]
		s_ratio = s_in[0]/s_in[1]

		x_rescaled = multiply(x,s_ratio)
		y_rescaled = y

	else:
		""" Inputs have the same scaling """
		max_scaling = s_in[0]

		x_rescaled = x
		y_rescaled = y


	# Perform addition with the re-scaled bit-streams
	select_prob = 0
	select_line = sng(select_prob,x.size)

	x_plus_y = np.empty(x.size, dtype=np.int8)
	x_plus_y[select_line==0] = y_rescaled[select_line==0]
	x_plus_y[select_line==1] = x_rescaled[select_line==1]
	x_plus_y = x_plus_y.astype(np.int8)
	
    # Saturation arithmetic
	if(upscale):
        gain = 2 
		states = 32
		
		x_plus_y_upscaled = lin_gain(x_plus_y,N=states,gain=gain)

	else:
		x_plus_y_upscaled = x_plus_y

	return x_plus_y_upscaled

def sc_matvec_add(X,b,s_in,upscale=False):
	"""
    Matrix to vector addition of the form X + b. The vector b is broaccast along the rows of X.
    The implementation assumes that the inputs have a common scaling parameter.

    Parameters 
    ----------
    X: 3D numpy array holding the input bit-streams
    b: 1D numpy array with floating-point bias terms
    s_in: Integer input scaling of the bit-streams in X 

	Returns 
    -------
    3D numpy array holding the output stochastic bit-streams 

    Raises
    ------
    A ValueError if there is a dimension mismatch 
	"""
	(x_rows,x_cols,no_samples) = X.shape
	n_biases = b.size

	# Check that dimensions match 
	if(x_cols != n_biases):
		raise ValueError('Dimension mismatch in sc_matvec_add')

	# Scale the floating point biases 
	b_scalings = next_int_power_of_2(b)
	scaled_b = b/b_scalings 

	# Broadcast the 1D bias array along the 1st dimension of X
	scaled_B = np.tile(scaled_b,(x_rows,1))

	# Convert scaled floating point biases to stochastic bit-streams
	B_sc = mat_sng(scaled_B,no_samples)

	# Initialise the output array
	Y = np.empty((x_rows,x_cols,no_samples),dtype=np.int8)

	# Use the two input scaled adder iteratively
	for row_idx in range(x_rows):
		for col_idx in range(x_cols):
			Y[row_idx,col_idx,:] = add(X[row_idx,col_idx,:],
											  B_sc[row_idx,col_idx,:],
											  np.array([s_in,b_scalings[col_idx]]),
											  upscale)

	return Y

def dot(x,w,s_in,s_out,upscale=False,gain=None):
	"""
	Scaled inner product in stochastic computing. The implementation assumes that 
    the input bit-streams have a common scaling parameter.

    Parameters 
    ----------
    x: 2D numpy array holding the input bit-streams
    w: 1D numpy array holding the floating point weights 
    s_in: Scalar specifying the input scaling parameter
    s_out: Scalar specifying the output scaling parameter
    upscale: Boolean indicating whether saturation arithmetic should be applied
    gain: Scalar specifying the linear gain to be applied if upscale is True

    Returns 
    -------
    1D NumPy array holding the output bit-stream
	"""
    (no_inputs, no_samples) = x.shape

	# Initialize a temporal array z
	z = np.empty((no_inputs,no_samples),dtype=np.int8)

    # Change the sign of inputs associated to negative weights
	for idx in range(no_inputs):
		if(w[idx] < 0):
			z[idx,:] = invert(x[idx,:])
		else:
			z[idx,:] = x[idx,:]

    # Construct a probability distribution over the weights 
	w_sum = sum(abs(w))
	p = abs(w)/w_sum

	rand_input = np.random.choice(no_inputs,size=no_samples,replace=True,p=p)

	# Initialise the output array
	y = np.empty(no_samples,dtype=np.int8)

    # Compute the output bit-stream 
	for sample in range(no_samples):
		y[sample] = z[rand_input[sample],sample]

	# Rescale the output bit-stream to s_out 
	curr_scaling = s_in*w_sum
	rescale_ratio = curr_scaling/s_out
	y_downscaled = multiply(y,rescale_ratio)

    # Apply saturation arithmetic 
	if(upscale):
		states = 32
		if(gain==32):
			gain_1 = 4
			gain_2 = 8
			y_upscaled_ = lin_gain(y_downscaled,N=states,gain=gain_1)
			y_upscaled = lin_gain(y_upscaled_,N=states,gain=gain_2)
		elif(gain==16):
			gain_1 = 4
			gain_2 = 4
			y_upscaled_ = lin_gain(y_downscaled,N=states,gain=gain_1)
			y_upscaled = lin_gain(y_upscaled_,N=states,gain=gain_2)
		else:
			y_upscaled = lin_gain(y_downscaled,N=states,gain=gain)

	else:
		y_upscaled = y_downscaled

	y_value = get_value(y_upscaled) 

	return y_value

def dec_dot(x,w,s_in,no_nodes,s_int,upscale=False,g_int=None,g_out=None):
	"""
    Stochastic inner product with decomposition. The implementation assumes that 
    the input bit-streams have a common scaling parameter.

    Parameters 
    ----------
    x: 2D numpy array holding the input bit-streams
    w: 1D numpy array holding the floating point weights 
    s_in: Scalar specifying the input scaling parameter
    no_nodes: Integer specifying the number of nodes in the dot-product decomposition
    s_int: Scalar specifying the intermediate scaling parameter
    upscale: Boolean indicating whether saturation arithmetic should be applied
    g_int: Scalar specifying the linear gain to be applied to the intermediate signals
    g_out: Scalar specifying the linear gain to be applied to the output signal

    Returns
    -------
    1D NumPy array holding the output bit-stream
	"""
    (no_inputs, no_samples) = x.shape

	# Initialize a temporal array z
    z = np.empty((no_inputs,no_samples),dtype=np.int8)

	# Change the sign of inputs associated to negative weights
	for idx in range(no_inputs):
		if(w[idx] < 0):
			z[idx,:] = invert(x[idx,:])
		else:
			z[idx,:] = x[idx,:]

	# Number of inputs in each sub-product 
	node_input_size = int(no_inputs/no_nodes)

	# Initialise an array to hold the intermediate results 
    y_int = np.empty((no_nodes,no_samples),dtype=np.int8)

	# Calculate and re-scale each intermediate result
	for node_idx in range(no_nodes):
		start_idx = node_idx*node_input_size
		end_idx = (node_idx+1)*node_input_size

		z_int = z[start_idx:end_idx,:] 
		w_int = w[start_idx:end_idx]

		# Construct a probability distribution over the weights
		w_sum = sum(abs(w_int)) 
		p = abs(w_int)/w_sum 

		rand_input = np.random.choice(node_input_size,size=no_samples,replace=True,p=p)

		# Compute the intermediate bit-streams
		for sample in range(no_samples):
			y_int[node_idx,sample] = z_int[rand_input[sample],sample]

        # Rescale the intermediate signals to s_int
		curr_scaling = s_in*w_sum
		rescale_ratio = curr_scaling/s_int
        y_downscaled = multiply(y_int[node_idx,:],rescale_ratio)

        # Saturation arithmetic 
		if(upscale):
			states = 32
			y_int[node_idx,:] = lin_gain(y_downscaled,N=states,gain=g_int)
		else:
			# Do not modify the output
			y_int[node_idx,:] = y_downscaled

	# Accumulate the intermediate signals to a single bit-stream
	y = np.empty(no_samples,dtype=np.int8) 

	# Generate random indeces with equal probabilities
	rand_input = np.random.choice(no_nodes,
								  size=no_samples,
								  replace=True)

    # Compute the output bit-stream
	for sample in range(no_samples):
		y[sample] = y_int[rand_input[sample],sample]
	
    # Saturation arithmetic 
	if(upscale): 
		states = 32
		y_upscaled = lin_gain(y,N=states,gain=g_out)
	else:
		y_upscaled = y

	y_value = get_value(y_upscaled) 

	return y_value

def sc_matmul(X,W,Sin,Sout,no_nodes=None,Sint=None,upscale=False,g_int=None,g_out=None):
	"""
    Stochastic matrix multiplication of the form X*W. The implementation assumes that 
    the input bit-streams have a common scaling parameter. 

    Note: Requires dot product implementations to return floating point results. Queue() 
    stops working if bit-streams are returned by dot product. 

    Parameters 
    ----------
    X: 3D numpy array holding the input bit-streams
    W: 2D numpy array holding the floating point weights 
    s_in: Scalar specifying the input scaling parameter
    s_out: Scalar specifying output scaling. Used only in the single MUX dot product
    no_nodes: Integer specifying the number of nodes in the dot-product decomposition
    s_int: Scalar specifying the intermediate scaling parameter
    upscale: Boolean indicating whether saturation arithmetic should be applied
    g_int: Scalar specifying the linear gain to be applied to the intermediate signals
    g_out: Scalar specifying the linear gain to be applied to the output signal

    Returns
    -------
    3D numpy array holding the output bit-streams

    Raises
    ------
    A ValueError if there is a dimension mismatch 
    """

	# Check that dimensions match 
	if(X.shape[1] != W.shape[0]):
		raise ValueError('Dimension mismatch in sc_matmul')

    (w_rows,w_cols) = W.shape
    (x_rows,x_cols,no_samples) = X.shape
    
	# Initialise the output array
	matmul_res = np.empty((x_rows,w_cols,no_samples),dtype=np.int8)

	# Define a helper function 
	def helper(row_idx,queue):
		loc_res = np.empty(w_cols)
		if no_nodes is not None and Sint is not None: 
			# Use the dot product with decomposition 
			for col_idx in range(w_cols):
				# Customised dot product returns the result in a float
				loc_res[col_idx] = dec_dot(X[row_idx,:,:], 
												W[:,col_idx], 
												Sin,
												no_nodes,
												Sint,
												upscale,
												g_int,
												g_out)

			tpl = (row_idx,loc_res)
			queue.put(tpl)
		else:
			# Use the single MUX dot product 
			for col_idx in range(w_cols):   
				# Customised dot product returns the result in a float
				loc_res[col_idx] = dot(X[row_idx,:,:],
											W[:,col_idx],
											Sin,
											Sout,
											upscale,
											g_int)

			tpl = (row_idx,loc_res)
			queue.put(tpl)

	def do_it(): 
		procs = []
		queue = multiprocessing.Queue()

		for row_idx in range(x_rows):
			proc = multiprocessing.Process(target=helper,args=(row_idx,queue))
			procs.append(proc)
			proc.start()

		for proc in procs:
			proc.join()

		for row_idx in range(x_rows):
			idx, res = queue.get()
			matmul_res[idx,:,:] = vec_sng(res,no_samples)

	do_it()

	return matmul_res

def popcount(x): 
	return sum(x)

def non_lin_sat(x,d):
    """
    Approximates a non-linear block 
    """ 
	no_samples = x.size
	z = np.empty(no_samples,dtype=np.int8)

	for sample in range(no_samples):
		start_idx = sample
		end_idx = (sample+d-1)%no_samples

		if(popcount(x[start_idx:end_idx]) >= d/2):
			z[sample] = 1
		else:
			z[sample] = 0

	return z

def Stanh(x,N): 
    """
    Stochastic hyperbolic tangent

    Parameters
    ----------
    x: Input stochastic bit-stream 
    N: Integer specifying the number of states

    Returns
    -------
    A bit-stream approximating tanh(xN/2
    """
	no_samples = x.size
	z = np.empty(no_samples,dtype=np.int8)

	# Initialise FSM parameters 
	S_min = 0
	S_max = N-1 
	S_bound = N/2

	# Initialise the state S
	S = S_bound

	# Run the FSM 
	for sample in range(no_samples): 
		# State transition
		if(x[sample] == 0): 
			S = S-1
		else:
			S = S+1
		
		# Saturate the counter 
		if(S < S_min): 
			S = S_min
		elif(S > S_max):
			S = S_max

		# Output logic 
		if(S >= S_bound):
			z[sample] = 1
		else:
			z[sample] = 0

	return z

def lin_gain(x,N,gain):
    """
    Linear gain with saturation. The implementation currently supports only a 
    linear gain of 2, 4 and 8. 

    Parameters
    ----------
    x: Input stochastic bit-stream 
    N: Integer specifying the number of states
    gain: Integer specifying the gain to be applied

    Returns
    -------
    Output bit-stream 
    
    Raises
    ------
    A ValueError if a gain different than 2,4 and 8 is specified
    """
	# Initialise the output 
	no_samples = x.size
	z = np.empty(no_samples,dtype=np.int8) 

	# Generate a bit-stream representing K
	if(gain == 2):  
		K = -0.35
	elif(gain == 4): 
		K = 0.25
	elif(gain == 8): 
		K = 0.78
	else: 
		raise ValueError('Provided gain does not exist in the look up table')

	k = sng(K,no_samples)

	# Initialise FSM parameters 
	S_min = 0
	S_max = N-1 
	S_bound = N/2

	# Initialise the state S
	S = S_bound

	# Run the FSM 
	for sample in range(no_samples): 
		# State transitions 
		if(S >= S_bound): 
			# Current state is on the RHS 
			if(S == S_max): 
				# Handle the state transitions of Smax
				if(x[sample] == 1): 
					S = S_max
				else:
					S = S-1
			else: 
				# Handle the rest of RHS states
				if(x[sample] == 1 and k[sample] == 0): 
					# Self transition
					S = S
				elif(x[sample] == 1 and k[sample] == 1): 
					# Right transition
					S = S+1
				elif(x[sample] == 0): 
					# Left transition 
					S = S-1
		else: 
			# Current state is on the LHS 
			if(S == S_min): 
				# Handle the state transitions of Smin 
				if(x[sample] == 0): 
					S = S_min 
				else: 
					S = S+1
			else: 
				# Handle the rest of the LHS states 
				if(x[sample] == 0 and k[sample] == 0): 
					# Self transition 
					S = S
				elif(x[sample] == 0 and k[sample] == 1): 
					# Left transition 
					S = S-1
				elif(x[sample] == 1): 
					S = S+1

		# Output logic 
		if(S >= S_bound):
			z[sample] = 1
		else:
			z[sample] = 0

	return z

def Sexp(x,G,N): 
	"""
	Stochastic exponentiation function

    Note: G should be such that G<<N

	Parameters
	----------
	x: Stochastic bit-stream 
	G: Positive integer 
	N: Integer specifying the number of states 

	Returns
	-------
	A stochastic bit-stream approximating exp(-2Gx)
	"""
	no_samples = x.size
	z = np.empty(no_samples,dtype=np.int8)

	# Initialise FSM Parameters
	S_min = 0
	S_max = N-1
	S_bound = N-G

	# Initialise the state S
	S = N/2

	# Run the FSM 
	for sample in range(no_samples):
		# State transition
		if(x[sample] == 0):
			S = S-1
		else:
			S = S+1

		# Saturate the counter 
		if(S < S_min): 
			S = S_min
		elif(S > S_max):
			S = S_max

		# Output logic 
		if(S >= S_bound):
			z[sample] = 0
		else:
			z[sample] = 1

	return z

def Sabs(x,N):
	"""
	Stochastic approximation of the absolute value function 

	Parameters
	----------
	x: Stochastic bit-stream
	N: Integer specifying the number of states 

	Returns
    -------
	A stochastic bit-stream approximating abs(x)
	"""
	no_samples = x.size
	z = np.empty(no_samples,dtype=np.int8)

	# Initialise FSM Parameters 
	S_min = 0
	S_max = N-1
	S_bound = N/2

	# Initialise the state S
	S = S_bound

	# Run the FSM 
	for sample in range(no_samples):
		# State transition
		if(x[sample] == 0): 
			S = S-1
		else:
			S = S+1
		
		# Saturate the counter 
		if(S < S_min): 
			S = S_min
		elif(S > S_max):
			S = S_max

		# Output logic
		if(S >= S_bound):
			if(S % 2 ): 
				z[sample] = 1 # sample is odd
			else:
				z[sample] = 0 # sample is even
		else:
			if(S % 2): 
				z[sample] = 0 # sample is odd
			else:
				z[sample] = 1 # sample is even

	return z

def Smax(a,b,N=32):
	"""
	Stochastic max unit

    Parameters
    ----------
    a,b: Input bit-streams 
    N: Integer specifying the number of states

    Returns 
    -------
    A bit-stream approximating max(a,b)
	"""
	# Scaled subtraction
	x = add(a,invert(b),np.array([1,1]))

    # Generate select line y
	y = Stanh(x,N=N)

	# MUX with y as the select line 
	max_a_b = np.empty(x.size, dtype=np.int8)
	max_a_b[y==0] = b[y==0]
	max_a_b[y==1] = a[y==1]
	max_a_b = max_a_b.astype(np.int8)

	return max_a_b

def vSmax(lst):
	"""
	Compute the max value from the list
	"""
	if(len(lst) == 1): 
		return lst[0]
	else:
		return Smax(lst[0],vSmax(lst[1:]))

def Smin(a,b,N=32):
	"""
    Stochastic min unit

    Parameters
    ----------
    a,b: Input bit-streams 
    N: Integer specifying the number of states

    Returns 
    -------
    A bit-stream approximating min(a,b)
    """
	# Scaled subtraction
	x = add(a,invert(b),np.array([1,1]))

    # Generate select line y
	y = Stanh(x,N=N)

	# MUX with y as the select line 
	min_a_b = np.empty(x.size, dtype=np.int8)
	min_a_b[y==0] = a[y==0]
	min_a_b[y==1] = b[y==1]
	min_a_b = min_a_b.astype(np.int8)

	return min_a_b

def vSmin(lst):
	"""
	Compute the min value from the list
	"""
	if(len(lst) == 1): 
		return lst[0]
	else:
		return Smin(lst[0],vSmin(lst[1:]))

def Srelu(x,N=32):
	"""
	Stochastic approximation of the ReLU

    Parameters
    ----------
    x: Input stochastic bit-stream 
    N: Integer specifying the number of states 

    Returns 
    -------
    A bit-stream approximating relu(x) 
	"""
	z = sng(0,x.size)
	return Smax(z,x,N=N)

def vec_Srelu(x):
	"""
	Vectorised form of Srelu 
	"""
    (x_rows, x_cols, no_samples) = x.shape

	# Initialise the output 3D-array 
	z = np.empty((x_rows,x_cols,no_samples),dtype=np.int8)

	for row_idx in range(x_rows): 
		for col_idx in range(x_cols): 
			z[row_idx,col_idx,:] = Srelu(x[row_idx,col_idx,:])

	return z
