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
    initializer = tf.random_uniform_initializer(dtype=tf.float32, minval=0, maxval=shape[0])
    return tf.get_variable("lamda", shape,initializer=initializer, dtype=tf.float32)
def theta_variable(shape):
    initializer = tf.random_uniform_initializer(dtype=tf.float32, minval=0, maxval=shape[0])
    return tf.get_variable("theta", shape,initializer=initializer, dtype=tf.float32)
def generatingWeightMatrix(images, labels, epoch, division, batch, g):
    W = py_func(generatingWeightMatrix_py, [images, labels, epoch, division, batch, g], [tf.float32])
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
        self.x = tf.reshape(x, shape=[-1, 28, 28, 3])
        self.x_re=tf.reshape(x_re,shape=[-1,1,784])
        self.x_d=tf.reshape(x_re,shape=[-1,1,784])
        self.y = y
        self.keep_prob = tf.placeholder(tf.float32)
        self.e=tf.placeholder(tf.float32)
        self.batch=tf.placeholder(tf.float32)

        # with tf.variable_scope("fc0"):
        #     W_fc2 = weight_variable([28*28*3, 32])
        #     b_fc2 = bias_variable([32])
        #     x_flat = tf.reshape(x, [-1, 28*28*3])
        #     glgcm_h_fc1 = tf.matmul(x_flat, W_fc2) + b_fc2

        #####################glgcm#########################
        with tf.variable_scope('glgcm'):
            lamda = lamda_variable([conf.ngray,1])
            theta= theta_variable([conf.ngray,1])
            g=tf.matmul(tf.minimum(tf.maximum(tf.subtract(self.x_d,lamda),1e-5),1),tf.minimum(tf.maximum(tf.subtract(self.x_re,theta),1e-5),1), transpose_b=True)
            #print(g.get_shape())
        with tf.variable_scope("glgcm_fc1"):
            g_flat = tf.reshape(g, [-1, conf.ngray*conf.ngray])
            glgcm_W_fc1 = weight_variable([conf.ngray*conf.ngray, 32])
            glgcm_b_fc1 = bias_variable([32])
            glgcm_h_fc1 = tf.nn.relu(tf.matmul(g_flat, glgcm_W_fc1) + glgcm_b_fc1)

        glgcm_h_fc1 = tf.nn.l2_normalize(glgcm_h_fc1, 0)

        self.H = glgcm_h_fc1

        #####################################glgcm############################
        ######################################hex#############################
        #H = glgcm_h_fc1
        ######################################hex############################

        ######################################Sentiment######################
        # conv1
        with tf.variable_scope("fc2"):
            W_fc2 = weight_variable([32, 10])
            b_fc2 = bias_variable([10])
            y_conv_loss = tf.matmul(glgcm_h_fc1, W_fc2) + b_fc2
            y_conv_pred = tf.matmul(glgcm_h_fc1, W_fc2) + b_fc2
          ######################################Sentiment######################


        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=y_conv_loss))
        self.pred = tf.argmax(y_conv_pred, 1)


        self.correct_prediction = tf.equal(tf.argmax(y_conv_pred,1), tf.argmax(self.y,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))