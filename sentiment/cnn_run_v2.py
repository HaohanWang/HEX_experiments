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

sys.path.append('../')

from tensorflow import py_func

from dataGeneration.dataLoader import loadDataSentiment

import tensorflow as tf

from cnn_v2 import MNISTcnn

def oneHotRepresentation(y):
    n = y.shape[0]
    r = np.zeros([n, 10])
    for i in range(r.shape[0]):
        r[i,int(y[i])] = 1
    return r

#from tensorflow.examples.tutorials.mnist import input_data
def predict(sess, x, keep_prob, pred, Xtest, Ytest, output_file):
    feed_dict = {x:Xtest, keep_prob: 1.0}
    prediction = sess.run(pred, feed_dict=feed_dict)

    with open(output_file, "w") as file:
        writer = csv.writer(file, delimiter = ",")
        writer.writerow(["id","label"])
        for i in range(len(prediction)):
            writer.writerow([str(i), str(prediction[i])])

    print("Output prediction: {0}". format(output_file))

def draw(train_acc,val_acc,test_acc,corr,x):
    print (train_acc,val_acc,test_acc,corr,x)
    
    plt.plot(train_acc,color='r')
    plt.plot(val_acc,color='g')
    plt.plot(test_acc,color='b')
    plt.xticks(np.arange(1, x+1, 1.0))
    #plt.rcParams['figure.figsize'] = (8, 6)
    plt.xlabel("Num of Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy vs Validation Accuracy vs Test Accuracy")
    plt.legend(['train','validation','test'])
    plt.savefig(str(int(corr*10))+'.png')
    #plt.show()

def draw_all(h_train_acc,h_val_acc,h_test_acc,n_train_acc,n_val_acc,n_test_acc,corr,):
    
    plt.plot(h_train_acc,linestyle='-',color='r')
    plt.plot(h_val_acc,linestyle='-',color='g')
    plt.plot(h_test_acc,linestyle='-',color='b')
    plt.plot(n_train_acc,linestyle=':',color='r')
    plt.plot(n_val_acc,linestyle=':',color='g')
    plt.plot(n_test_acc,linestyle=':',color='b')
    #plt.xticks(np.arange(1, x+1, 1.0))
    #plt.rcParams['figure.figsize'] = (8, 6)
    plt.xlabel("Num of Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy vs Validation Accuracy vs Test Accuracy")
    plt.legend(['hex_train','hex_validation','hex_test','train','validation','test'])
    plt.savefig(str(int(corr*10))+'_all.png')
    #plt.show()        

def train(args, Xtrain, Xtrain_re,Xtrain_d,Ytrain, Xval,Xval_re, Xval_d, Yval, Xtest, Xtest_re,Xtest_d,Ytest,corr,use_hex=True):
    #""" reuse """
    #with tf.variable_scope('model',reuse=tf.AUTO_REUSE ) as scope:
        print("prepare for training with corr=%f, hex=%d" % (corr,use_hex))
        num_class = 7
        print (Xtrain.shape, Xtrain_re.shape, Xtrain_d.shape, Xval.shape, Xval_re.shape, Xval_d.shape, Xtest.shape,Xtest_re.shape, Xtest_d.shape)

        
        x_re = tf.placeholder(tf.float32, (None, args.ngray,28*28))
        x_d = tf.placeholder(tf.float32, (None,args.ngray, 28*28)) 
        x = tf.placeholder(tf.float32,(None,28*28))
        y = tf.placeholder(tf.float32, (None, num_class))
        model = MNISTcnn(x, y, x_re, x_d, args, Hex_flag=use_hex)

        # optimizer = tf.train.AdamOptimizer(1e-4).minimize(model.loss)
        optimizer = tf.train.AdamOptimizer(5e-4)
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
            for epoch in range(args.epochs): 
                    
                begin = time.time()

                # train
                ######
            
                train_accuracies = []
                train_losses = []
                for i in range(num_batches):

                    batch_x = Xtrain[i*args.batch_size:(i+1)*args.batch_size,:]
                    batch_xd = Xtrain_d[i*args.batch_size:(i+1)*args.batch_size,:]
                    batch_re = Xtrain_re[i*args.batch_size:(i+1)*args.batch_size,:]
                    batch_y = Ytrain[i*args.batch_size:(i+1)*args.batch_size,:]

                    _, acc, loss = sess.run([first_train_op, model.accuracy, model.loss], feed_dict={x: batch_x, x_re: batch_re,x_d: batch_xd, y: batch_y, model.keep_prob: 0.5, model.e: epoch,model.batch: i})
                    # if epoch < args.div:
                    #     _, acc, loss = sess.run([first_train_op, model.accuracy, model.loss], feed_dict={x: batch_x, x_re: batch_re,x_d: batch_xd, y: batch_y, model.keep_prob: 0.5, model.e: epoch,model.batch: i})
                    # else:
                    #     _, acc, loss = sess.run([second_train_op, model.accuracy, model.loss], feed_dict={x: batch_x, x_re: batch_re,x_d: batch_xd, y: batch_y, model.keep_prob: 0.5, model.e: epoch,model.batch: i})

                    # if i%5!=4:
                    #     print (acc, end='\t')
                    # else:
                    #     print (acc)
                    train_accuracies.append(acc)
                    train_losses.append(loss)
                train_acc_mean = np.mean(train_accuracies)
                train_acc.append(train_acc_mean)

                train_loss_mean = np.mean(train_losses)

                # print ()

                # compute loss over validation data
                if validation:
                    val_accuracies = []
                    for i in range(val_num_batches):
                        batch_x = Xval[i*args.batch_size:(i+1)*args.batch_size,:]
                        batch_xd = Xval_d[i*args.batch_size:(i+1)*args.batch_size,:]
                        batch_re = Xval_re[i*args.batch_size:(i+1)*args.batch_size,:]
                        batch_y = Yval[i*args.batch_size:(i+1)*args.batch_size,:]
                        acc = sess.run(model.accuracy, feed_dict={x: batch_x, x_re:batch_re,x_d: batch_xd, y: batch_y, model.keep_prob: 1.0, model.e: epoch})
                        val_accuracies.append(acc)
                    val_acc_mean = np.mean(val_accuracies)
                    val_acc.append(val_acc_mean)
                    # log progress to console
                    print("\nEpoch %d, time = %ds, train accuracy = %.4f, loss = %.4f,  validation accuracy = %.4f" % (epoch, time.time()-begin, train_acc_mean,  train_loss_mean, val_acc_mean))
                else:
                    print("\nEpoch %d, time = %ds, train accuracy = %.4f" % (epoch, time.time()-begin, train_acc_mean))
                sys.stdout.flush()

                #test
                test_accuracies = []
                for i in range(test_num_batches):
                    batch_x = Xtest[i*args.batch_size:(i+1)*args.batch_size,:]
                    batch_xd = Xtest_d[i*args.batch_size:(i+1)*args.batch_size,:]
                    batch_re = Xtest_re[i*args.batch_size:(i+1)*args.batch_size,:]
                    batch_y = Ytest[i*args.batch_size:(i+1)*args.batch_size,:]
                    acc = sess.run(model.accuracy, feed_dict={x: batch_x, x_re: batch_re, x_d: batch_xd, y: batch_y, model.keep_prob: 1.0, model.e: epoch})
                    test_accuracies.append(acc)
                score = np.mean(test_accuracies)


                if val_acc_mean > best_validate_accuracy:
                    # best_validate_accuracy = val_acc_mean
                    # test_accuracies = []
                    # for i in range(test_num_batches):
                    #     batch_x = Xtest[i*args.batch_size:(i+1)*args.batch_size,:]
                    #     batch_xd = Xtest_d[i*args.batch_size:(i+1)*args.batch_size,:]
                    #     batch_re = Xtest_re[i*args.batch_size:(i+1)*args.batch_size,:]
                    #     batch_y = Ytest[i*args.batch_size:(i+1)*args.batch_size,:]
                    #     acc = sess.run(model.accuracy, feed_dict={x: batch_x, x_re: batch_re, x_d: batch_xd, y: batch_y, model.keep_prob: 1.0, model.e: epoch})
                    #     test_accuracies.append(acc)
                    # score = np.mean(test_accuracies)

                    print("Best Validated Model Prediction Accuracy = %.4f " % (score))
                test_acc.append(score)

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

def load_and_deal(corr,row,column,ngray=16):
    # load data
    print ("load date with corr=%f..." % (corr))
    xtrain, ytrain, xval, yval, xtest, ytest=loadDataSentiment(corr=corr)
    indices = np.random.permutation(xtrain.shape[0])
    xtrain=xtrain[indices,:]
    ytrain=ytrain[indices,:]

    indices = np.random.permutation(xval.shape[0])
    xval=xval[indices,:]
    yval=yval[indices,:]

    indices = np.random.permutation(xtest.shape[0])
    xtest=xtest[indices,:]
    ytest=ytest[indices,:]
    """
    xtrain=np.load(input+'xtrain.npy')
    ytrain=np.load(input+'ytrain.npy')
    xval=np.load(input+'xval.npy')
    yval=np.load(input+'yval.npy')
    xtest=np.load(input+'xtest.npy')
    ytest=np.load(input+'ytest.npy')
    ytrain=oneHotRepresentation(ytrain)
    yval=oneHotRepresentation(yval)
    ytest=oneHotRepresentation(ytest)
    """
    # deal data: get <start pixel> 
    print ("deal date with ngray=%d..." % (ngray))
    direction=np.diag((-1)*np.ones(784))
    for i in range(784):
        x=int(math.floor(i/28))
        y=int(i%28)
        if x+row<28 and y+column<28:   
            direction[i][i+row*28+column]=1

    xtrain_d=np.copy(xtrain)
    xtrain_re=np.copy(xtrain)

    xval_d=np.copy(xval)
    xval_re=np.copy(xval)

    xtest_d=np.copy(xtest)
    xtest_re=np.copy(xtest)

    #regularized train_re
    for i in range(xtrain_re.shape[0]):
        xtrain_re[i] = np.asarray(1.0 * xtrain_re[i] * (ngray-1) / xtrain_re[i].max(), dtype=np.int16) # 0-255变换为0-15
        xtrain_d[i]=np.dot(xtrain_re[i],direction)
    for i in range(xval_re.shape[0]):
        xval_re[i] = np.asarray(1.0 * xval_re[i] * (ngray-1) / xval_re[i].max(), dtype=np.int16) # 0-255变换为0-15
        xval_d[i]=np.dot(xval_re[i],direction)
    for i in range(xtest_re.shape[0]):
        xtest_re[i] = np.asarray(1.0 * xtest_re[i] * (ngray-1) / xtest_re[i].max(), dtype=np.int16) # 0-255变换为0-15
        xtest_d[i]=np.dot(xtest_re[i],direction)

    #xtr xva xte 中间变量
    xtr=np.repeat(xtrain_re,ngray,0)
    xtrain_re=xtr.reshape(xtrain_re.shape[0],ngray,784)
    #print (xtrain[0].shape)

    xva=np.repeat(xval_re,ngray,0)
    xval_re=xva.reshape(xval_re.shape[0],ngray,784)

    xte=np.repeat(xtest_re,ngray,0)
    xtest_re=xte.reshape(xtest_re.shape[0],ngray,784) 

    ################# delta ################
    xtr_d=np.repeat(xtrain_d,ngray,0) 
    xtrain_d=xtr_d.reshape(xtrain_d.shape[0],ngray,784)
    #print (xtr_d.shape)
    #print(xtrain_d.shape)
    
    xva_d=np.repeat(xval_d,ngray,0)
    xval_d=xva_d.reshape(xval_d.shape[0],ngray,784)
    #print (xva_d.shape)
    #print (xval_d.shape)
    
    xte_d=np.repeat(xtest_d,ngray,0)
    xtest_d=xte_d.reshape(xtest_d.shape[0],ngray,784)
    ################# delta ################

    return xtrain, xtrain_re, xtrain_d, ytrain, xval, xval_re, xval_d, yval, xtest, xtest_re, xtest_d, ytest
def main(args):
     #Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = loadDataSentiment(corr=corr)
    Xtrain, Xtrain_re,Xtrain_d, Ytrain, Xval, Xval_re, Xval_d, Yval, Xtest, Xtest_re, Xtest_d, Ytest = load_and_deal(corr,args.row,args.col,args.ngray)

    #data = input_data.read_data_sets(args.data_dir, one_hot=True, reshape=False, validation_size=args.val_size)
    
    print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':'))) 

    
    if args.hex==1:
        (h_train_acc,h_val_acc,h_test_acc)=train(args, 
                                                Xtrain,Xtrain_re,Xtrain_d, Ytrain, 
                                                Xval, Xval_re, Xval_d, Yval, 
                                                Xtest, Xtest_re, Xtest_d, Ytest,
                                                corr, True)
        hex_acc=np.array((h_train_acc,h_val_acc,h_test_acc))
        np.save(args.save+'hex_acc_'+str(args.corr)+'_'+str(args.row)+'_'+str(args.col)+'_'+str(args.div)+'.npy',hex_acc)
    else:
        (n_train_acc,n_val_acc,n_test_acc)=train(args, 
                                                Xtrain, Xtrain_re, Xtrain_d, Ytrain, 
                                                Xval, Xval_re, Xval_d, Yval, 
                                                Xtest, Xtest_re, Xtest_d, Ytest,
                                                corr, False)
        acc=np.array((n_train_acc,n_val_acc,n_test_acc))
        np.save(args.save+'acc_'+str(args.corr)+'_'+str(args.row)+'_'+str(args.col)+'_'+str(args.div)+'.npy',acc)
    #draw_all(h_train_acc,h_val_acc,h_test_acc,n_train_acc,n_val_acc,n_test_acc,corr)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt_dir', type=str, default='ckpts/', help='Directory for parameter checkpoints')
    parser.add_argument('-l', '--load_params', dest='load_params', action='store_true', help='Restore training from previous model checkpoint?')
    parser.add_argument("-o", "--output",  type=str, default='prediction.csv', help='Prediction filepath')
    parser.add_argument('-e', '--epochs', type=int, default=1000, help='How many epochs to run in total?')
    parser.add_argument('-b', '--batch_size', type=int, default=128, help='Batch size during training per GPU')
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
    corr=float(args.corr/10.0)
    main(args)

   
