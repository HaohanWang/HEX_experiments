__author__ = 'Haohan Wang'

# -*- encoding=utf-8 -*-
import cv2
# Standard library
import time
import gzip
import zipfile
import numpy as np
import os
import math
from PIL import Image
import matplotlib.pyplot as plt
import cPickle as pickle

def fft(img):
    return np.fft.fft2(img)
def fftshift(img):
    return np.fft.fftshift(fft(img))
def ifft(img):
    return np.fft.ifft2(img)
def ifftshift(img):
    return ifft(np.fft.ifftshift(img))

def distance(i,j,w,h,r):
    dis=np.sqrt((i-14)**2+(j-14)**2)
    if dis<r:
        return 0.0
    else:
        return 1.0

def addingPattern(r, mask):
    fftshift_img_r=fftshift(r)
    fftshift_result_r = fftshift_img_r * mask
    result_r = ifftshift(fftshift_result_r)
    mr=np.abs(result_r)
    return mr

def mask_radial_MM(isGray=True):
    mask = np.zeros((28,28))
    for i in range(28):
        for j in range(28):
            mask[i,j]=distance(i,j,28,28,r=3.5)
    return mask

def mask_random_MM(p = 0.5,isGray=True):
    mask=np.random.binomial(1,1-p,(28,28))
    return mask

def addMultiDomainPattern(r, l, randomMask=None, radioMask=None):
    if l == 0:
        return r
    elif l == 1:
        return addingPattern(r, radioMask)
    else:
        return addingPattern(r, randomMask)


def loadMultiDomainMNISTData():
    '''
    :param testCase:
            0 for original distribution as testing
            1 for random kernel as testing
            2 for radial kernel as testing
    :return:
    '''

    np.random.seed(1)

    f = gzip.open('../data/MNIST/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f)
    f.close()

    noMask = np.ones([28, 28])
    randomMask = mask_random_MM()
    radioMask = mask_radial_MM()

    _Xtrain=np.zeros((training_data[0].shape[0],28*28))

    Images = np.zeros([30*3, 30*11])

    Images[1:29, 1:29] = noMask
    Images[1+30:29+30, 1:29] = radioMask
    Images[1+60:29+60, 1:29] = randomMask

    for i in range(30):
        r = training_data[0][i]
        r=r.reshape(28,28)
        l = 0
        img = addMultiDomainPattern(r, l, randomMask=randomMask, radioMask=radioMask)
        j = i%10
        Images[1+l*30:1+l*30+28,1+(j+1)*30:1+(j+1)*30+28] = img
        l = 1
        img = addMultiDomainPattern(r, l, randomMask=randomMask, radioMask=radioMask)
        j = i%10
        Images[1+l*30:1+l*30+28,1+(j+1)*30:1+(j+1)*30+28] = img
        l = 2
        img = addMultiDomainPattern(r, l, randomMask=randomMask, radioMask=radioMask)
        j = i%10
        Images[1+l*30:1+l*30+28,1+(j+1)*30:1+(j+1)*30+28] = img


    cv2.imwrite('MNIST.jpg', Images*256)

if __name__ == '__main__':
    loadMultiDomainMNISTData()