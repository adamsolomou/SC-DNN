from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import numpy as np
import tensorflow as tf

n_features = 784 # MNIST data input (img shape: 28*28)
n_classes  = 10   # MNIST total classes (0-9 digits)

X = tf.placeholder(tf.float32, [None, n_features])
Y = tf.placeholder(tf.float32, [None, n_classes])

with tf.Session() as sess:

    features, labels = sess.run([X,Y,], 
                                feed_dict={X: mnist.test.images,
                                           Y: mnist.test.labels})

np.savetxt("data/features.csv", features, delimiter=",")
np.savetxt("data/labels.csv",   labels,   delimiter=",")

