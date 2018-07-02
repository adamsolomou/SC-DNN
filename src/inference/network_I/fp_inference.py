from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import numpy as np
import tensorflow as tf

# Network parameters 
n_hidden_1 = 128 # 1st layer number of neurons
n_features = 784 # MNIST data input (img shape: 28*28)
n_classes = 10   # MNIST total classes (0-9 digits)

# Load trained coefficients 
w_dic = {
    'h1':   np.genfromtxt('coeffs/W1.csv', delimiter=','),
    'out':  np.genfromtxt('coeffs/Wout.csv', delimiter=','),
}

b_dic = {
    'b1':   np.genfromtxt('coeffs/b1.csv', delimiter=','),
    'out':  np.genfromtxt('coeffs/bout.csv', delimiter=','),
}

# Graph input 
X = tf.placeholder(tf.float32, [None, n_features])
Y = tf.placeholder(tf.float32, [None, n_classes])

# Network coefficients 
weights = {
    'h1': tf.placeholder(tf.float32, [n_features, n_hidden_1], name="w1"),
    'out': tf.placeholder(tf.float32, [n_hidden_1, n_classes], name="wout")
}

biases = {
    'b1': tf.placeholder(tf.float32, [n_hidden_1], name="b1"),
    'out': tf.placeholder(tf.float32, [n_classes], name="bout")
}

# Network graph 
def neural_net(x):
    # Hidden layer 
    layer_1 = tf.add( tf.matmul(x, weights['h1']), biases['b1'] ) 
    # Output layer 
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']

    return out_layer

logits = neural_net(X)
prediction = tf.nn.softmax(logits)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:

    # Run floating point network inference
    print("Testing accuracy: ", 
        sess.run(accuracy, feed_dict={X: mnist.train.images,
                                      Y: mnist.train.labels,
                                      weights['h1']:  w_dic['h1'],
                                      weights['out']: w_dic['out'],
                                      biases['b1']:   b_dic['b1'],
                                      biases['out']:  b_dic['out']}))
