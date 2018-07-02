from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import numpy as np
import tensorflow as tf

"""
Network Parameters
"""
n_hidden_1 = 128 # 1st layer number of neurons
n_features = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 	 # MNIST total classes (0-9 digits)

"""
tf Graph Input 
"""
X = tf.placeholder(tf.float32, [None, n_features])
Y = tf.placeholder(tf.float32, [None, n_classes])

"""
Declare network variables
"""
weights = {
	'h1': tf.Variable(tf.zeros([n_features, n_hidden_1]), name="w1"),
	'out': tf.Variable(tf.zeros([n_hidden_1, n_classes]), name="wout")
}

biases = {
	'b1': tf.Variable(tf.zeros([n_hidden_1]), name="b1"),
	'out': tf.Variable(tf.zeros([n_classes]), name="bout")
}

"""
Create the model graph 
"""

# Hidden fully connected layer with 256 neurons
l1_matmul = tf.matmul(X, weights['h1'])
l1_matadd = tf.add(l1_matmul, biases['b1'])

# Output fully connected layer with a neuron for each class
lout_matmul = tf.matmul(l1_matadd, weights['out'])
out_layer = tf.add(lout_matmul, biases['out'])

# Construct model
logits = out_layer
prediction = tf.nn.softmax(logits)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Add ops to save and restore all the variables
saver = tf.train.Saver()

with tf.Session() as sess:

	# Run the initialisation 
	sess.run(init)

	# Restore trained weights and biases 
	saver.restore(sess, "/tmp/learned_sc_model.ckpt")
 	print("Model restored.")

	H1_m, H1_a, Hout_m, Hout_a = sess.run([l1_matmul, 
	                              		   l1_matadd, 
                                  		   lout_matmul,
                                  		   out_layer], 
	                              		   feed_dict={X: mnist.test.images,
                                           			  Y: mnist.test.labels})

np.savetxt("values/learned/H1_m.csv", 	    H1_m, 	 	delimiter=",")
np.savetxt("values/learned/H1_a.csv", 	    H1_a, 	 	delimiter=",")
np.savetxt("values/learned/Hout_m.csv",    	Hout_m,  	delimiter=",")
np.savetxt("values/learned/Hout_a.csv",    	Hout_a,  	delimiter=",")
