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
    W = py_func(generatingWeightMatrix_py, [images, labels, epoch, division, batch], [tf.float32])
    return W

def checkInformation(rep, epoch, s):
    X = py_func(checkInformation_py, [rep, epoch, s], [tf.float32])
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
        self.x = tf.reshape(x, shape=[-1, 227, 227, 3])
        self.x_re=tf.reshape(x_re,shape=[-1,1,784])
        self.x_d=tf.reshape(x_re,shape=[-1,1,784])
        self.y = y
        self.keep_prob = tf.placeholder(tf.float32)
        self.e=tf.placeholder(tf.float32)
        self.batch=tf.placeholder(tf.float32)
        self.WEIGHTS_PATH='weights/bvlc_alexnet.npy'
        #self.WEIGHTS_PATH='/Users/hzxue/Desktop/CMU/project/artificial-pattern/src/HEX_719/PACS/PACS/bvlc_alexnet.npy'

        #####################glgcm#########################

        conv1 = conv(self.x, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        norm1 = lrn(conv1, 2, 1e-05, 0.75, name='norm1')
        pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')
        
        # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
        conv2 = conv(pool1, 5, 5, 256, 1, 1, groups=2, name='conv2')
        norm2 = lrn(conv2, 2, 1e-05, 0.75, name='norm2')
        pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')
        
        # 3rd Layer: Conv (w ReLu)
        conv3 = conv(pool2, 3, 3, 384, 1, 1, name='conv3')

        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.reshape(pool5, [-1, 6*6*256])
        fc6 = fc(flattened, 6*6*256, 4096, name='fc6')
        dropout6 = dropout(fc6, self.keep_prob)

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = fc(dropout6, 4096, 4096, name='fc7')

        fc7 = tf.nn.l2_normalize(fc7, 0)

        dropout7 = dropout(fc7, self.keep_prob)

        # 8th Layer: FC and return unscaled activations
        #self.fc8 = fc(dropout7, 4096, self.NUM_CLASSES, relu=False, name='fc8')
            # conv2
            # dropout
        #h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
        h_fc1_drop = dropout7

        self.rep = fc7

        # fc2
        with tf.variable_scope("fc2"):
            W_fc2 = weight_variable([4096, 7])
            b_fc2 = bias_variable([7])
            y_conv_loss = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        ######################################Sentiment######################
        # H = y_conv_H
        # y_conv_loss = tf.nn.l2_normalize(y_conv_loss, 0)
        # y_conv_H = tf.nn.l2_normalize(y_conv_H, 0)
        
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

    def load_initial_weights(self, session):
        """Load weights from file into network.
        As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
        come as a dict of lists (e.g. weights['conv1'] is a list) and not as
        dict of dicts (e.g. weights['conv1'] is a dict with keys 'weights' &
        'biases') we need a special load function
        """
        # Load the weights into memory
        weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()

        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:

            # Check if layer should be trained from scratch
            if op_name != 'fc8':

                with tf.variable_scope(op_name, reuse=True):

                    # Assign weights/biases to their corresponding tf variable
                    for data in weights_dict[op_name]:

                        # Biases
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable=True)
                            session.run(var.assign(data))

                        # Weights
                        else:
                            var = tf.get_variable('weights', trainable=True)
                            session.run(var.assign(data))

