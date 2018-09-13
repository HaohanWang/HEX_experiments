# -*- encoding=utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import csv
import time
import math
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
from datagenerator import ImageDataGenerator
from tensorflow.contrib.data import Iterator

sys.path.append('../')

#from tensorflow import py_func


import tensorflow as tf

from alex_cnn_top import MNISTcnn
def preparion(img,args):
    row=args.row
    column=args.col
    x=np.copy(img)
    x_d=np.copy(img)
    x_re=np.copy(img)

    x=x.reshape(x.shape[0],784)
    x_re=x_re.reshape(x_re.shape[0],784)
    x_d=x_d.reshape(x_d.shape[0],784)

    direction=np.diag((-1)*np.ones(784))
    for i in range(784):
        x=int(math.floor(i/28))
        y=int(i%28)
        if x+row<28 and y+column<28:
            direction[i][i+row*28+column]=1

    for i in range(x_re.shape[0]):
        x_re[i] = np.asarray(1.0 * x_re[i] * (args.ngray-1) / x_re[i].max(), dtype=np.int16) # 0-255变换为0-15
        x_d[i]=np.dot(x_re[i],direction)
    return x_d, x_re

def oneHotRepresentation(y):
    n = y.shape[0]
    r = np.zeros([n, 7])
    for i in range(r.shape[0]):
        r[int(y[i])] = 1
    return r

def set_path(choice):

    if choice=='sketch':
        s_tr = 'sourceonly/sketch/train.txt'
        s_val = 'sourceonly/sketch/val.txt'
        s_te='sourceonly/sketch/test.txt'
        return s_tr, s_val, s_te
    if choice=='cartoon':
        c_tr = 'sourceonly/cartoon/train.txt'
        c_val = 'sourceonly/cartoon/val.txt'
        c_te='sourceonly/cartoon/test.txt'
        return c_tr, c_val, c_te
    if choice=='photo':
        p_tr = 'sourceonly/photo/train.txt'
        p_val = 'sourceonly/photo/val.txt'
        p_te='sourceonly/photo/test.txt'
        return p_tr, p_val, p_te
    if choice=='art':
        a_tr = 'sourceonly/art_painting/train.txt'
        a_val = 'sourceonly/art_painting/val.txt'
        a_te='sourceonly/art_painting/test.txt'
        return a_tr, a_val, a_te

def loadData(cat='sketch'):
    path = 'representations/'+cat+'_'
    train_rep = np.load(path+'train_rep.npy')
    train_re = np.load(path+'train_re.npy')
    train_d = np.load(path+'train_d.npy')
    train_y = np.load(path+'train_y.npy')

    val_rep = np.load(path+'val_rep.npy')
    val_re = np.load(path+'val_re.npy')
    val_d = np.load(path+'val_d.npy')
    val_y = np.load(path+'val_y.npy')

    test_rep = np.load(path+'test_rep.npy')
    test_re = np.load(path+'test_re.npy')
    test_d = np.load(path+'test_d.npy')
    test_y = np.load(path+'test_y.npy')

    return train_rep, train_re, train_d, train_y, val_rep, val_re, val_d, val_y, test_rep, test_re, test_d, test_y


def train(args, use_hex=True):
        num_classes = 7
        dataroot = '../data/PACS/'
        batch_size=args.batch_size

        cat = 'photo'

        train_rep, train_re, train_d, train_y, val_rep, val_re, val_d, val_y, test_rep, test_re, test_d, test_y = loadData(cat)

        # iterator = Iterator.from_structure(tr_data.data.output_types,
        #                                tr_data.data.output_shapes)
        # next_batch = iterator.get_next()
        #
        # training_init_op = iterator.make_initializer(tr_data.data)
        # validation_init_op = iterator.make_initializer(val_data.data)
        # test_init_op = iterator.make_initializer(test_data.data)
        #
        # train_batches_per_epoch = int(np.floor(tr_data.data_size/args.batch_size))
        # val_batches_per_epoch = int(np.floor(val_data.data_size / args.batch_size))
        # test_batches_per_epoch = int(np.floor(test_data.data_size / args.batch_size))

        num_batches = train_rep.shape[0] // args.batch_size

        validation = True
        val_num_batches = val_rep.shape[0] // args.batch_size

        test_num_batches = test_rep.shape[0] // args.batch_size

        x_re = tf.placeholder(tf.float32, (None,28*28))
        x_d = tf.placeholder(tf.float32, (None, 28*28))
        x = tf.placeholder(tf.float32,(None,4096))
        y = tf.placeholder(tf.float32, (None, num_classes))
        model = MNISTcnn(x, y, x_re, x_d, args, Hex_flag=use_hex)

        # optimizer = tf.train.AdamOptimizer(1e-4).minimize(model.loss)
        optimizer = tf.train.AdamOptimizer(1e-4) # default was 0.0005
        first_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"glgcm") + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"glgcm_fc1") \
                           + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"fc2")
        first_train_op = optimizer.minimize(model.loss, var_list=first_train_vars)
        second_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"fc2")\
                            # + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"conv1") + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"conv2") \
                            # + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"conv3") + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"conv4") \
                            # + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"conv5") + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"fc6") \
                            # + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"fc7")
        second_train_op = optimizer.minimize(model.loss, var_list=second_train_vars)
        # second_train_op = optimizer.minimize(model.loss)

        # train_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"glgcm") + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"glgcm_fc1") \
        #            + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"fc2") + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"fc7") \
        #            + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"fc6")
        # first_train_op = optimizer.minimize(model.loss, var_list=train_vars)

        saver = tf.train.Saver(tf.trainable_variables())

        with tf.Session() as sess:

            # print('Starting training')
            # print('load Alex net weights')
            # model.load_initial_weights(sess)

            sess.run(tf.initialize_all_variables())
            if args.load_params:
                ckpt_file = os.path.join(args.ckpt_dir, 'mnist_model.ckpt')
                print('Restoring parameters from', ckpt_file)
                saver.restore(sess, ckpt_file)


            validation = True


            best_validate_accuracy = 0
            score = 0
            train_acc=[]
            test_acc=[]
            val_acc=[]
            for epoch in range(args.epochs):

                begin = time.time()
                # sess.run(training_init_op)
                #sess.run(validation_init_op)
                #sess.run(test_init_op)
                # train
                ######

                train_accuracies = []
                train_losses = []
                for i in range(num_batches):
                    batch_x = train_rep[i*args.batch_size:(i+1)*args.batch_size,:]
                    batch_xd = train_d[i*args.batch_size:(i+1)*args.batch_size,:]
                    batch_re = train_re[i*args.batch_size:(i+1)*args.batch_size,:]
                    batch_y = train_y[i*args.batch_size:(i+1)*args.batch_size,:]

                    if epoch < args.div:
                        _, acc, loss = sess.run([first_train_op, model.accuracy, model.loss], feed_dict={x: batch_x,
                                                        x_re: batch_re,
                                                        x_d: batch_xd,
                                                        y: batch_y,
                                                        model.keep_prob: 0.5,
                                                        model.e: epoch,
                                                        model.batch: i})
                    else:
                        _, acc, loss = sess.run([second_train_op, model.accuracy, model.loss], feed_dict={x: batch_x,
                                                        x_re: batch_re,
                                                        x_d: batch_xd,
                                                        y: batch_y,
                                                        model.keep_prob: 0.5,
                                                        model.e: epoch,
                                                        model.batch: i})

                    train_accuracies.append(acc)
                    train_losses.append(loss)
                train_acc_mean = np.mean(train_accuracies)
                train_acc.append(train_acc_mean)
                train_loss_mean = np.mean(train_losses)

                # print ()

                # compute loss over validation data
                if validation:
                    # sess.run(validation_init_op)
                    val_accuracies = []
                    for i in range(val_num_batches):
                        batch_x = val_rep[i*args.batch_size:(i+1)*args.batch_size,:]
                        batch_xd = val_d[i*args.batch_size:(i+1)*args.batch_size,:]
                        batch_re = val_re[i*args.batch_size:(i+1)*args.batch_size,:]
                        batch_y = val_y[i*args.batch_size:(i+1)*args.batch_size,:]

                        acc = sess.run(model.accuracy, feed_dict={x: batch_x, x_re:batch_re,
                                                        x_d: batch_xd, y: batch_y,
                                                        model.keep_prob: 1.0,
                                                        model.e: epoch,
                                                        model.batch: i})
                        val_accuracies.append(acc)
                    val_acc_mean = np.mean(val_accuracies)
                    val_acc.append(val_acc_mean)
                    # log progress to console
                    print("\nEpoch %d, time = %ds, train accuracy = %.4f, loss = %.4f,  validation accuracy = %.4f" % (epoch, time.time()-begin, train_acc_mean,  train_loss_mean, val_acc_mean))
                else:
                    print("\nEpoch %d, time = %ds, train accuracy = %.4f" % (epoch, time.time()-begin, train_acc_mean))
                sys.stdout.flush()

                #test

                if val_acc_mean > best_validate_accuracy:
                    best_validate_accuracy = val_acc_mean

                    test_accuracies = []
                    # sess.run(test_init_op)
                    for i in range(test_num_batches):

                        batch_x = test_rep[i*args.batch_size:(i+1)*args.batch_size,:]
                        batch_xd = test_d[i*args.batch_size:(i+1)*args.batch_size,:]
                        batch_re = test_re[i*args.batch_size:(i+1)*args.batch_size,:]
                        batch_y = test_y[i*args.batch_size:(i+1)*args.batch_size,:]

                        acc = sess.run(model.accuracy, feed_dict={x: batch_x,
                                                        x_re: batch_re, x_d: batch_xd, y: batch_y,
                                                        model.keep_prob: 1.0,
                                                        model.e: epoch,
                                                        model.batch: i})
                        test_accuracies.append(acc)
                    score = np.mean(test_accuracies)

                    print("Best Validated Model Prediction Accuracy = %.4f " % (score))
                test_acc.append(score)

                if (epoch + 1) % 10 == 0:
                    ckpt_file = os.path.join(args.ckpt_dir, 'mnist_model.ckpt')

            ckpt_file = os.path.join(args.ckpt_dir, 'mnist_model.ckpt')
            saver.save(sess, ckpt_file)
            """ reuse """
            print("Best Validated Model Prediction Accuracy = %.4f " % (score))
            return (train_acc,val_acc,test_acc)



def main(args):
    print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':')))

    if args.hex==1:
        (h_train_acc,h_val_acc,h_test_acc)=train(args, True)
        hex_acc=np.array((h_train_acc,h_val_acc,h_test_acc))
        np.save(args.save+'hex_acc_'+str(args.corr)+'_'+str(args.row)+'_'+str(args.col)+'_'+str(args.div)+'.npy',hex_acc)
    else:
        (n_train_acc,n_val_acc,n_test_acc)=train(args, False)
        acc=np.array((n_train_acc,n_val_acc,n_test_acc))
        np.save(args.save+'acc_'+str(args.corr)+'_'+str(args.row)+'_'+str(args.col)++'_'+str(args.div)+'.npy',acc)
    #draw_all(h_train_acc,h_val_acc,h_test_acc,n_train_acc,n_val_acc,n_test_acc,corr)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt_dir', type=str, default='ckpts/', help='Directory for parameter checkpoints')
    parser.add_argument('-l', '--load_params', dest='load_params', action='store_true', help='Restore training from previous model checkpoint?')
    parser.add_argument("-o", "--output",  type=str, default='prediction.csv', help='Prediction filepath')
    parser.add_argument('-e', '--epochs', type=int, default=25000, help='How many epochs to run in total?')
    parser.add_argument('-b', '--batch_size', type=int, default=128, help='Batch size during training per GPU') # todo: default was 128
    parser.add_argument('-re', '--re', type=int, default=0, help='regularization?')
    parser.add_argument('-corr', '--corr', type=int, default=8, help='correlation')
    parser.add_argument('-hex','--hex',type=int, default=1, help='use hex?')
    parser.add_argument('-save','--save',type=str, default='hex2/', help='save acc npy path?')
    parser.add_argument('-row', '--row', type=int, default=0, help='direction delta in row')
    parser.add_argument('-col', '--col', type=int, default=0, help='direction delta in column')
    parser.add_argument('-ng', '--ngray', type=int, default=16, help='regularization gray level')
    parser.add_argument('-div', '--div', type=int, default=200, help='how many epochs before HEX start')
    #print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':')))

    args = parser.parse_args()

    tf.set_random_seed(100)
    np.random.seed()

    if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    # pretty print args
    print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':')))
    main(args)


