__author__ = 'Haohan Wang'

import numpy as np
import tensorflow as tf

from tensorflow import py_func
from Model.helpingFunctions_v2 import generatingWeightMatrix_py, checkInformation_py
def lamda_variable(shape):
    initializer = tf.random_uniform_initializer(dtype=tf.float32, minval=0, maxval=shape[0])
    return tf.get_variable("lamda", shape,initializer=initializer, dtype=tf.float32)

def theta_variable(shape):
    initializer = tf.random_uniform_initializer(dtype=tf.float32, minval=0, maxval=shape[0])
    return tf.get_variable("theta", shape,initializer=initializer, dtype=tf.float32)

def generatingWeightMatrix(images, labels, epoch, division, batch):
    W = py_func(generatingWeightMatrix_py, [images, labels, epoch, division, batch], [tf.float32])
    return W

def checkInformation(rep, epoch, s, y):
    X = py_func(checkInformation_py, [rep, epoch, s, y], [tf.float32])[0]
    return X

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
def fc(x, num_in, num_out, name, relu=True):
    """Create a fully connected layer."""
    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights', shape=[num_in, num_out],
                                  trainable=True)
        biases = tf.get_variable('biases', [num_out], trainable=True)

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

    if relu:
        # Apply ReLu non linearity
        relu = tf.nn.relu(act)
        return relu
    else:
        return act


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name,
             padding='SAME'):
    """Create a max pooling layer."""
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    """Create a local response normalization layer."""
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)


def dropout(x, keep_prob):
    """Create a dropout layer."""
    return tf.nn.dropout(x, keep_prob)

def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1):
    """Create a convolution layer.
    Adapted from: https://github.com/ethereon/caffe-tensorflow
    """
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights', shape=[filter_height,
                                                    filter_width,
                                                    input_channels/groups,
                                                    num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])

    if groups == 1:
        conv = convolve(x, weights)

    # In the cases of multiple groups, split inputs & weights and
    else:
        # Split input and weights and convolve them separately
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weight_groups = tf.split(axis=3, num_or_size_splits=groups,
                                 value=weights)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

        # Concat the convolved output together again
        conv = tf.concat(axis=3, values=output_groups)

    # Add biases
    bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

    # Apply relu function
    relu = tf.nn.relu(bias, name=scope.name)

    return relu


class MNISTcnn(object):
    def __init__(self,x, y, x_re, x_d, conf, Hex_flag=False):
        self.x = tf.reshape(x, shape=[-1, 4096])
        self.x_re=tf.reshape(x_re,shape=[-1,1,784])
        self.x_d=tf.reshape(x_re,shape=[-1,1,784])
        self.y = y
        self.keep_prob = tf.placeholder(tf.float32)
        self.e=tf.placeholder(tf.float32)
        self.batch=tf.placeholder(tf.float32)
        #self.WEIGHTS_PATH='/Users/hzxue/Desktop/CMU/project/artificial-pattern/src/HEX_719/PACS/PACS/bvlc_alexnet.npy'

        #####################glgcm#########################

        with tf.variable_scope('glgcm'):
            lamda = lamda_variable([conf.ngray,1])
            theta= theta_variable([conf.ngray,1])
            g=tf.matmul(tf.minimum(tf.maximum(tf.subtract(self.x_d,lamda),1e-5),1),tf.minimum(tf.maximum(tf.subtract(self.x_re,theta),1e-5),1), transpose_b=True)
            #g=tf.reduce_sum(index,reduction_indices=2)
            #print(g.get_shape())


        with tf.variable_scope("glgcm_fc1"):
            g_flat = tf.reshape(g, [-1, conf.ngray*conf.ngray])
            glgcm_W_fc1 = weight_variable([conf.ngray*conf.ngray, 32])
            glgcm_b_fc1 = bias_variable([32])
            glgcm_h_fc1 = tf.nn.relu(tf.matmul(g_flat, glgcm_W_fc1) + glgcm_b_fc1)
        # glgcm_h_fc1_drop = tf.nn.dropout(glgcm_h_fc1, self.keep_prob)

        glgcm_h_fc1 = tf.nn.l2_normalize(glgcm_h_fc1, 0)

        #####################################glgcm############################
        ######################################hex#############################
        # H = glgcm_h_fc1
        ######################################hex############################

        ######################################Sentiment######################

        rep = tf.nn.l2_normalize(self.x, 0)

        dropout7 = dropout(rep, self.keep_prob)

        # 8th Layer: FC and return unscaled activations
        #self.fc8 = fc(dropout7, 4096, self.NUM_CLASSES, relu=False, name='fc8')
            # conv2
            # dropout
        #h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
        h_fc1_drop = dropout7


        yconv_contact_loss=tf.concat([h_fc1_drop, glgcm_h_fc1],1)
        pad=tf.zeros_like(glgcm_h_fc1, tf.float32)
        yconv_contact_pred=tf.concat([h_fc1_drop, pad],1)
        pad2 = tf.zeros_like(rep, tf.float32)
        yconv_contact_H = tf.concat([pad2, glgcm_h_fc1],1)
        # fc2
        with tf.variable_scope("fc2"):
            W_fc2 = weight_variable([4128, 7])
            b_fc2 = bias_variable([7])
            y_conv_loss = tf.matmul(yconv_contact_loss, W_fc2) + b_fc2
            y_conv_pred = tf.matmul(yconv_contact_pred, W_fc2) + b_fc2
            y_conv_H = tf.matmul(yconv_contact_H, W_fc2) + b_fc2
        ######################################Sentiment######################
        # H = y_conv_H
        # y_conv_loss = tf.nn.l2_normalize(y_conv_loss, 0)
        # y_conv_H = tf.nn.l2_normalize(y_conv_H, 0)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=y_conv_pred))
        self.pred = tf.argmax(y_conv_pred, 1)

        self.correct_prediction = tf.equal(tf.argmax(y_conv_pred,1), tf.argmax(self.y,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.loss -= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_conv_H, logits=y_conv_pred))