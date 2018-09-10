import numpy as np
import tensorflow as tf

from tensorflow import py_func
from Model.helpingFunctions_v2 import generatingWeightMatrix_py
def _fft(x):
    r = []
    for i in range(128):
        r.append(np.abs(np.fft.fftshift(np.fft.fft2(x[i,:].reshape([28,28])))).astype(np.float32).reshape(28*28))
    return np.array(r)
    # return np.abs(np.fft.fft2(x)).astype(np.float32) # this seems to be an interesting approach

def fftImage(x):
    r = py_func(_fft, [x], [tf.float32])[0]
    return r

def lamda_variable(shape):
    initializer = tf.truncated_normal_initializer(dtype=tf.float32, stddev=1e-1)
    return tf.get_variable("lamda", shape,initializer=initializer, dtype=tf.float32)
def theta_variable(shape):
    initializer = tf.truncated_normal_initializer(dtype=tf.float32, stddev=1e-1)
    return tf.get_variable("theta", shape,initializer=initializer, dtype=tf.float32)
def generatingWeightMatrix(images, labels, epoch, division, batch):
    W = py_func(generatingWeightMatrix_py, [images, labels, epoch, division, batch], [tf.float32])[0]
    return W
def weight_variable(shape):
    initializer = tf.truncated_normal_initializer(dtype=tf.float32, stddev=1e-1)
    return tf.get_variable("weights", shape,initializer=initializer, dtype=tf.float32)

def bias_variable(shape):
    initializer = tf.constant_initializer(0.0)
    return tf.get_variable("biases", shape, initializer=initializer, dtype=tf.float32)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

class MNISTcnn(object):
    def __init__(self, x, y, x_re, x_d, conf, Hex_flag=False):
        self.x = tf.reshape(x, shape=[-1, 28, 28, 1])
        self.x_re=tf.reshape(x_re,shape=[-1, 1, 784])
        self.x_d=x_d
        self.y = y
        self.keep_prob = tf.placeholder(tf.float32)
        self.e=tf.placeholder(tf.float32)
        self.batch=tf.placeholder(tf.float32)
        # conv1
        with tf.variable_scope('hex'):
            with tf.variable_scope('conv1'):
                W_conv1 = weight_variable([5, 5, 1, 32])
                if conf.re==1:
                    tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(0.001)(W_conv1))
                b_conv1 = bias_variable([32])
                h_conv1 = tf.nn.relu(conv2d(self.x, W_conv1) + b_conv1)
                h_pool1 = max_pool_2x2(h_conv1)

            # conv2
            with tf.variable_scope('conv2'):
                W_conv2 = weight_variable([5, 5, 32, 64])
                b_conv2 = bias_variable([64])
                h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
                h_pool2 = max_pool_2x2(h_conv2)

            # fc1
            with tf.variable_scope("fc1"):
                shape = int(np.prod(h_pool2.get_shape()[1:]))
                W_fc1 = weight_variable([shape, 1024])
                b_fc1 = bias_variable([1024])
                h_pool2_flat = tf.reshape(h_pool2, [-1, shape])
                h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

            # dropout
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

            # fc2
            with tf.variable_scope("fc2"):
                W_fc2 = weight_variable([1024, 10])
                b_fc2 = bias_variable([10])
                y_conv_loss = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
            ######################################Sentiment######################


        sess_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=y_conv_loss))
        if Hex_flag==False:
            if conf.re==1:
                tf.add_to_collection("losses",sess_loss)
                self.loss = tf.add_n(tf.get_collection("losses"))
            else:
                 self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=y_conv_loss))
        self.pred = tf.argmax(y_conv_loss, 1)

        self.correct_prediction = tf.equal(tf.argmax(y_conv_loss,1), tf.argmax(self.y,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))