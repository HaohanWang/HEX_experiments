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
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../')

from tensorflow import py_func

from dataLoader import loadDigitClassification
import tensorflow as tf

from cnn_glcm import MNISTcnn

def oneHotRepresentation(y):
    n = y.shape[0]
    r = np.zeros([n, 10])
    for i in range(r.shape[0]):
        r[i,int(y[i])] = 1
    return r
def check(img):
    print("start check img")
    cv2.imshow('1',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def toGray(img):
    gray=np.zeros([img.shape[0],28,28])
    for i in range(img.shape[0]):
        #check(img[i])
        im=np.zeros([28,28])

        if img[i][0,0,0]==0.0:

            im1=np.array(img[i]*255, dtype=np.uint8)
            im=im1[:,:,0]
            #check(im)
            #print(im)
        else:
            #check(img[i])
            #print(img[i])
            im=np.array(img[i]*255, dtype=np.uint8)
            im=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)

            #check(im)
            #print(im)
        #print(im.shape)
        gray[i]=im
    return gray

def preparion(img,args):
    gray_img=toGray(np.copy(img))
    row=args.row
    column=args.col
    x=np.copy(gray_img)
    x_d=np.copy(gray_img)
    x_re=np.copy(gray_img)

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
    #print(x_d.shape,x_re.shape)
    return x_d,x_re
#from tensorflow.examples.tutorials.mnist import input_data


def train(args, Xtrain,Ytrain, Xval, Yval, Xtest, Ytest, ztrain,zval,ztest,corr,use_hex=True):
    #""" reuse """
    #with tf.variable_scope('model',reuse=tf.AUTO_REUSE ) as scope:

    train_xd, train_re = preparion(Xtrain, args)
    val_xd, val_re = preparion(Xval, args)
    test_xd, test_re = preparion(Xtest, args)

    print("prepare for training with corr=%f, hex=%d" % (corr,use_hex))
    num_class = 10
    print (Xtrain.shape, Xval.shape, Xtest.shape)

    x_re = tf.placeholder(tf.float32, (None, 28*28))
    x_d = tf.placeholder(tf.float32, (None,28*28))
    x = tf.placeholder(tf.float32,(None,28,28,3))
    y = tf.placeholder(tf.float32, (None, num_class))
    model = MNISTcnn(x, y, x_re, x_d, args, Hex_flag=use_hex)

    #optimizer = tf.train.AdamOptimizer(1e-4).minimize(model.loss)
    optimizer = tf.train.AdamOptimizer(1e-2)
    first_train_op = optimizer.minimize(model.loss)
    second_train_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"hex")
    second_train_op = optimizer.minimize(model.loss, var_list=second_train_vars)

    """ reuse """
    #tf.get_variable_scope().reuse_variables()

    saver = tf.train.Saver(tf.trainable_variables())

    with tf.Session() as sess:
        print('Starting training')
        #sess.run(tf.global_variables_initializer())
        sess.run(tf.initialize_all_variables())

        if args.load_params:
            ckpt_file = os.path.join(args.ckpt_dir, 'mnist_model.ckpt')
            print('Restoring parameters from', ckpt_file)
            saver.restore(sess, ckpt_file)

        num_batches = Xtrain.shape[0] // args.batch_size

        validation = True
        val_num_batches = Xval.shape[0] // args.batch_size

        test_num_batches = Xtest.shape[0] // args.batch_size

        best_validate_accuracy = 0
        score = 0
        train_acc=[]
        test_acc=[]
        val_acc=[]

        Reps = None
        Label = None
        Pattern = None

        for epoch in range(args.epochs):

            begin = time.time()

            Reps = None
            Label = None
            Pattern = None
            # train
            ######

            train_accuracies = []
            for i in range(num_batches):

                batch_x = Xtrain[i*args.batch_size:(i+1)*args.batch_size,:,:,:]
                #batch_xd = Xtrain_d[i*args.batch_size:(i+1)*args.batch_size,:]
                #batch_re = Xtrain_re[i*args.batch_size:(i+1)*args.batch_size,:]
                batch_y = Ytrain[i*args.batch_size:(i+1)*args.batch_size,:]
                batch_xd = train_xd[i*args.batch_size:(i+1)*args.batch_size,:]
                batch_re = train_re[i*args.batch_size:(i+1)*args.batch_size,:]
                batch_z = ztrain[i*args.batch_size:(i+1)*args.batch_size,:]

                if epoch<args.div:
                    _, acc, rep = sess.run([first_train_op, model.accuracy, model.H], feed_dict={x: batch_x, x_re: batch_re,
                                                                            x_d: batch_xd, y: batch_y,
                                                                            model.keep_prob: 0.5,
                                                                            model.e: epoch,
                                                                            model.batch: i})
                else:
                    _, acc, rep = sess.run([second_train_op, model.accuracy,model.H], feed_dict={x: batch_x, x_re: batch_re,
                                                                            x_d: batch_xd, y: batch_y,
                                                                            model.keep_prob: 0.5,
                                                                            model.e: epoch,
                                                                            model.batch: i})


                if Reps is None:
                    Reps = rep
                    Label = batch_y
                    Pattern = batch_z
                else:
                    Reps = np.append(Reps, rep, 0)
                    Label = np.append(Label, batch_y, 0)
                    Pattern = np.append(Pattern, batch_z, 0)


                train_accuracies.append(acc)

            Label = np.argmax(Label, 1)
            Pattern = np.argmax(Pattern, 1)

            np.save('results/labels_hex_2_'+str(epoch), Label)
            np.save('results/patterns_hex_2_'+str(epoch), Pattern)
            np.save('results/representations_hex_2_'+str(epoch), Reps)

            train_acc_mean = np.mean(train_accuracies)
            train_acc.append(train_acc_mean)
            # print ()

            # compute loss over validation data

            if (epoch + 1) % 10 == 0:
                ckpt_file = os.path.join(args.ckpt_dir, 'mnist_model.ckpt')
                saver.save(sess, ckpt_file)

        ckpt_file = os.path.join(args.ckpt_dir, 'mnist_model.ckpt')
        saver.save(sess, ckpt_file)
        """ reuse """
        # scope.reuse_variables()
        #draw(train_acc,val_acc,test_acc,corr,args.epochs)

        print("Best Validated Model Prediction Accuracy = %.4f " % (score))
        return (train_acc,val_acc,test_acc)
        # return score

        # predict test data
        # predict(sess, x, model.keep_prob, model.pred, Xtest, Ytest, args.output)


        # origiinal test data from 'http://yann.lecun.com/exdb/mnist/'
        # """
        # acc = sess.run(model.accuracy, feed_dict={x: data.test.images, y: data.test.labels, model.keep_prob: 1.0})
        # print("test accuracy %g"%acc)
        # """

def main(args):
    Xtrain, Ytrain, Xval, Yval, Xtest, Ytest, ztrain, zval, ztest = loadDigitClassification()
    #toGray(Xtrain)
    #return
    print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':')))


    if args.hex==1:
        (h_train_acc,h_val_acc,h_test_acc)=train(args,
                                                Xtrain, Ytrain,
                                                Xval, Yval,
                                                Xtest,Ytest,
                                                ztrain,zval,ztest,
                                                corr, True)
        hex_acc=np.array((h_train_acc,h_val_acc,h_test_acc))
        np.save(args.save+'hex_acc_'+str(args.corr)+'_'+str(args.row)+'_'+str(args.col)+'_'+str(args.div)+'.npy',hex_acc)
    else:
        (n_train_acc,n_val_acc,n_test_acc)=train(args,
                                                Xtrain, Ytrain,
                                                Xval, Yval,
                                                Xtest, Ytest,
                                                ztrain,zval,ztest,
                                                corr, False)
        acc=np.array((n_train_acc,n_val_acc,n_test_acc))
        np.save(args.save+'acc_'+str(args.corr)+'_'+str(args.row)+'_'+str(args.col)+'_'+str(args.div)+'.npy',acc)
    #draw_all(h_train_acc,h_val_acc,h_test_acc,n_train_acc,n_val_acc,n_test_acc,corr)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt_dir', type=str, default='hex2/v2/ckpts/', help='Directory for parameter checkpoints')
    parser.add_argument('-l', '--load_params', dest='load_params', action='store_true', help='Restore training from previous model checkpoint?')
    parser.add_argument("-o", "--output",  type=str, default='prediction.csv', help='Prediction filepath')
    parser.add_argument('-e', '--epochs', type=int, default=1000, help='How many epochs to run in total?')
    parser.add_argument('-b', '--batch_size', type=int, default=128, help='Batch size during training per GPU')
    parser.add_argument('-re', '--re', type=int, default=0, help='regularization?')
    parser.add_argument('-corr', '--corr', type=int, default=8, help='correlation')
    parser.add_argument('-hex','--hex',type=int, default=1, help='use hex?')
    parser.add_argument('-save','--save',type=str, default='hex2/v2/', help='save acc npy path?')
    parser.add_argument('-row', '--row', type=int, default=0, help='direction delta in row')
    parser.add_argument('-col', '--col', type=int, default=1, help='direction delta in column')
    parser.add_argument('-ng', '--ngray', type=int, default=16, help='regularization gray level')
    parser.add_argument('-div', '--div', type=int, default=2, help='how many epochs before HEX start')
    parser.add_argument('-test', '--test', type=int, default=0, help='which data set to test?')
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
    corr=float(args.corr/10.0)
    main(args)

   
