""" Multilayer Perceptron.

A Multilayer Perceptron (Neural Network) implementation example using
TensorFlow library. This example is using the MNIST database of handwritten
digits (http://yann.lecun.com/exdb/synthetic/).

Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/synthetic/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

# ------------------------------------------------------------------
#
# THIS EXAMPLE HAS BEEN RENAMED 'neural_network.py', FOR SIMPLICITY.
#
# ------------------------------------------------------------------


from __future__ import print_function

import sys
sys.path.append('../')

import numpy as np
import tensorflow as tf
from tensorflow import py_func

from dataGeneration.dataLoader import loadData
from helpingFunctions import generatingWeightMatrix_py

def generatingWeightMatrix(images, labels):
    W = py_func(generatingWeightMatrix_py, [images, labels], [tf.float32])[0]
    return W

def loss(logits, labels, h, HEX=True):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    if not HEX:
        return tf.reduce_mean(cross_entropy)
    cross_entropy = tf.sqrt(tf.reshape(cross_entropy, [-1, 1]) + 1e-10)
    W = generatingWeightMatrix(h, labels)
    tf.stop_gradient(W)
    cross_entropy = tf.matmul(tf.matmul(cross_entropy, W, transpose_a=True), cross_entropy)
    # cross_entropy = tf.matmul(cross_entropy, cross_entropy, transpose_a=True)
    return tf.reduce_mean(cross_entropy)

# import numpy as np
# np.random.seed(1)
tf.set_random_seed(1)

# Experiment Setting

def experiment(seed, HEX_flag):

    n = 500
    p = 1000
    group = 100

    # Parameters
    learning_rate = 1e-3
    training_epochs = 100
    batch_size = 100
    display_step = 1

    # Network Parameters
    n_hidden_1 = 50 # 1st layer number of neurons
    n_hidden_2 = 10 # 2nd layer number of neurons
    n_input = p # MNIST data input (img shape: 28*28)
    n_classes = 2 # synthetic 2 classes

    Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = loadData(seed, n, p, group)

    # tf Graph input
    X = tf.placeholder("float", [None, n_input])
    Y = tf.placeholder("float", [None, n_classes])

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1],seed=0)),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2],seed=0)),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes],seed=0))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1],seed=0)),
        'b2': tf.Variable(tf.random_normal([n_hidden_2],seed=0)),
        'out': tf.Variable(tf.random_normal([n_classes],seed=0))
    }


    # Create model
    def multilayer_perceptron(x):
        # Hidden fully connected layer with 256 neurons
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        # Hidden fully connected layer with 256 neurons
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        # Output fully connected layer with a neuron for each class
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        return layer_1, layer_2, out_layer

    # Construct model
    layer1, layer2, logits = multilayer_perceptron(X)

    # Define loss and optimizer
    loss_op = loss(logits, Y, h=layer1, HEX=HEX_flag)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)
    # Initializing the variables
    init = tf.global_variables_initializer()

    # maxiVal = 0
    # saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        pred = tf.nn.softmax(logits)  # Apply softmax to logits
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(n/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                # batch_x, batch_y = synthetic.train.next_batch(batch_size)
                batch_x = Xtrain[i*batch_size:(i+1)*batch_size,:]
                batch_y = Ytrain[i*batch_size:(i+1)*batch_size,:]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))

                val = accuracy.eval({X: Xval, Y: Yval})
                print("\tValidation Accuracy: ={:.9f}".format(val))

                # if val > maxiVal:
                #     maxiVal = val
                #     saver.save(sess, 'current_best')

        print("Optimization Finished!")

        # Test model

        score = accuracy.eval({X: Xtest, Y: Ytest})
        print("Accuracy:", score)
        return score

if __name__ == '__main__':
    # results = []
    # for corr in [0.8]:
    #     for HEX_flag in [False, True]:
    #         result = []
    #         for seed in range(10):
    #             a = experiment(seed=seed, HEX_flag=HEX_flag)
    #             result.append(a)
    #         results.append(result)
    # results = np.array(results)
    # np.save('results_useful', results)
    experiment(seed=3, HEX_flag=True)