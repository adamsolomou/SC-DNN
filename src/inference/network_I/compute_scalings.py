from __future__ import print_function

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import numpy as np
import scaling_ops 

""" Network Parameters """
n_classes  = 10 	# MNIST total classes (0-9 digits)
n_features = 784 	# MNIST data input (img shape: 28*28)
n_hidden_1 = 128 	# 1st layer number of neurons
in_scaling = 1	 	# Normalised MNIST data lie in the range [0,1]

""" Load trained coefficients """
weights = {
    'h1':   np.genfromtxt('coeffs/W1.csv', delimiter=','),
    'out':  np.genfromtxt('coeffs/Wout.csv', delimiter=','),
}

biases = {
    'b1':   np.genfromtxt('coeffs/b1.csv', delimiter=','),
    'out':  np.genfromtxt('coeffs/bout.csv', delimiter=','),
}

""" Declare scaling ndarrays """
n_points = 10000 # Number of test data points 
S_x = np.ones([n_points,n_features])*in_scaling # Input scaling 2D array

""" Create a SC model description """
# Hidden layer 
S1_matmul = scaling_ops.sc_matmul_scaling(S_x,weights['h1'])
S1_add = scaling_ops.sc_matvec_add_scaling(S1_matmul,biases['b1'])

# Output layer 
Sout_matmul = scaling_ops.sc_matmul_scaling(S1_add,weights['out'])
Sout_add = scaling_ops.sc_matvec_add_scaling(Sout_matmul,biases['out'])

print("Input: ",     np.unique(S_x))
print("L1 matmul: ", np.unique(S1_matmul))
print("L1 matadd: ", np.unique(S1_add))
print("Lo matmul: ", np.unique(Sout_matmul))
print("Lo matadd: ", np.unique(Sout_add))