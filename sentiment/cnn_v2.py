import numpy as np
import tensorflow as tf

from tensorflow import py_func
from Model.helpingFunctions_v2 import generatingWeightMatrix_py, checkInformation_py
def lamda_variable(shape):
    initializer = tf.random_uniform_initializer(dtype=tf.float32, minval=0, maxval=16)
    return tf.get_variable("lamda", shape,initializer=initializer, dtype=tf.float32)

def theta_variable(shape):
    initializer = tf.random_uniform_initializer(dtype=tf.float32, minval=0, maxval=16)
    return tf.get_variable("theta", shape,initializer=initializer, dtype=tf.float32)

def generatingWeightMatrix(images, labels, epoch, division, batch):
    W = py_func(generatingWeightMatrix_py, [images, labels, epoch, division, batch], [tf.float32])[0]
    return W

def checkInformation(rep, epoch, s):
    X = py_func(checkInformation_py, [rep, epoch, s], [tf.float32])[0]
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

class MNISTcnn(object):
    def __init__(self, x, y, x_re, x_d, conf, Hex_flag=False):
        self.x = tf.reshape(x, shape=[-1, 28, 28, 1])
        self.x_re=tf.reshape(x_re,shape=[-1,conf.ngray,784])
        self.x_d=x_d
        self.y = y
        self.keep_prob = tf.placeholder(tf.float32)
        self.e=tf.placeholder(tf.float32)
        self.batch=tf.placeholder(tf.float32)
        #####################glgcm#########################

        with tf.variable_scope('glgcm'):
            lamda = lamda_variable([conf.ngray,1])
            theta= theta_variable([conf.ngray,1])
            # index=tf.multiply(tf.minimum(tf.maximum(tf.subtract(self.x_d,lamda),1e-5),1),tf.minimum(tf.maximum(tf.subtract(self.x_re,theta),1e-5),1))
            # g=tf.reduce_sum(index,reduction_indices=2)
            g=tf.matmul(tf.minimum(tf.maximum(tf.subtract(self.x_d,lamda),0),1),tf.minimum(tf.maximum(tf.subtract(self.x_re,theta),0),1), transpose_b=True)
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

            # h_fc1 = checkInformation(h_fc1, self.e)

            h_fc1 = tf.nn.l2_normalize(h_fc1, 0)
            # dropout
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)


            yconv_contact_loss=tf.concat([h_fc1_drop, glgcm_h_fc1],1)
            #yconv_contact_loss=tf.concat([tf.zeros_like(h_fc1_drop, tf.float32),tf.zeros_like(glgcm_h_fc1_drop, tf.float32)],1)

            pad=tf.zeros_like(glgcm_h_fc1, tf.float32)
            yconv_contact_pred=tf.concat([h_fc1_drop, pad],1)

            pad2 = tf.zeros_like(h_fc1, tf.float32)
            yconv_contact_H = tf.concat([pad2, glgcm_h_fc1],1)

            # fc2
            with tf.variable_scope("fc2"):
                W_fc2 = weight_variable([1056, 7])
                b_fc2 = bias_variable([7])
                y_conv_loss = tf.matmul(yconv_contact_loss, W_fc2) + b_fc2
                y_conv_pred = tf.matmul(yconv_contact_pred, W_fc2) + b_fc2
                y_conv_H = tf.matmul(yconv_contact_H, W_fc2) + b_fc2
            ######################################Sentiment######################

            """
            t_histo_rows = [
            tf.histogram_fixed_width(
                tf.gather(x, [row]),
                [0.0, 256.0], 100)
            for row in range(128)]

            H = tf.stack(t_histo_rows, axis=0)
            """
        # H = y_conv_H

        sess_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=y_conv_loss))
        if Hex_flag==False:
            if conf.re==1:
                tf.add_to_collection("losses",sess_loss)
                self.loss = tf.add_n(tf.get_collection("losses"))
            else:
                 self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=y_conv_loss))
        self.pred = tf.argmax(y_conv_pred, 1)

        # H = y_conv_H
        # H = tf.argmax(y_conv_H, 1)
        # y_H = tf.one_hot(H, depth=7)

        # y_conv_pred = checkInformation(y_conv_pred, self.e, 'hey')
        # H = checkInformation(H, self.e, 'ha')

        self.correct_prediction = tf.equal(tf.argmax(y_conv_pred,1), tf.argmax(self.y,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        if Hex_flag:
            # loss = tf.sqrt(tf.reshape(tf.cast(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=y_conv_loss), tf.float32), [-1, 1]) + 1e-10)

            # y_conv_loss = generatingWeightMatrix(y_conv_H, y_conv_loss, self.e, conf.div, self.batch)

            # y_conv_loss = tf.nn.l2_normalize(y_conv_loss, 0)
            # y_conv_H = tf.nn.l2_normalize(y_conv_H, 0)

            y_conv_loss = y_conv_loss - tf.matmul(tf.matmul(tf.matmul(y_conv_H, tf.matrix_inverse(tf.matmul(y_conv_H, y_conv_H, transpose_a=True))), y_conv_H, transpose_b=True), y_conv_loss)

            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=y_conv_loss))

            # self.loss = tf.reduce_mean(tf.multiply(W, tf.cast(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=y_conv_loss), tf.float32)))

            # tf.stop_gradient(W)
            # if conf.re==1:
            #     sess_loss = tf.matmul(tf.matmul(loss, W, transpose_a=True), loss)
            #
            #     tf.add_to_collection("losses",tf.reshape(sess_loss,[]))
            #     self.loss = tf.add_n(tf.get_collection("losses"))
            # else:
            #     self.loss=tf.matmul(tf.matmul(loss, W, transpose_a=True), loss)