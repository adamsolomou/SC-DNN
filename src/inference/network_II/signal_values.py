from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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

# Graph inputs
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

""" Network Graph """
# Hidden layer
l1_matmul = tf.matmul(X, weights['h1'])
l1_matadd = tf.add(l1_matmul, biases['b1'])

# Output layer 
lout_matmul = tf.matmul(l1_matadd, weights['out'])
out_layer = tf.add(lout_matmul, biases['out'])

logits = out_layer
prediction = tf.nn.softmax(logits)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:

    H1_m, H1_a, Hout_m, Hout_a = sess.run([l1_matmul, 
                                           l1_matadd, 
                                           lout_matmul,
                                           out_layer], 
                                           feed_dict={X: mnist.train.images,
                                                      Y: mnist.train.labels,
                                                      weights['h1']:  w_dic['h1'],
                                                      weights['out']: w_dic['out'],
                                                      biases['b1']:   b_dic['b1'],
                                                      biases['out']:  b_dic['out']})


    # Histograms for signal values 
    plt.figure()
    plt.hist(H1_m.flatten(), bins='auto', facecolor='blue', alpha=0.75)
    plt.title('Weight product - Hidden layer')
    plt.ylabel('Frequency')
    plt.xlabel('Signal Values')
    plt.savefig('h1_matmul.eps', format='eps', dpi=1000)

    plt.figure()
    plt.hist(H1_a.flatten(), bins='auto', facecolor='blue', alpha=0.75)
    plt.title('Bias addition - Hidden layer')
    plt.ylabel('Frequency')
    plt.xlabel('Signal Values')
    plt.savefig('h1_matddd.eps', format='eps', dpi=1000)

    plt.figure()
    plt.hist(Hout_m.flatten(), bins='auto', facecolor='blue', alpha=0.75)
    plt.title('Weight product - Output layer')
    plt.ylabel('Frequency')
    plt.xlabel('Signal Values')
    plt.savefig('hout_matmul.eps', format='eps', dpi=1000)

    plt.figure()
    plt.hist(Hout_a.flatten(), bins='auto', facecolor='blue', alpha=0.75)
    plt.title('Bias addition - Output layer')
    plt.ylabel('Frequency')
    plt.xlabel('Signal Values')
    plt.savefig('hout_matadd.eps', format='eps', dpi=1000)
    plt.show()