# -*- encoding=utf-8 -*-
import os
import sys
import csv
import time
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
# from skimage.transform import rotate
# from skimage.feature import local_binary_pattern
# from skimage import data
# from skimage.feature import hog
# from skimage.io import imread
# from skimage.color import label2rgb
# import skimage
import scipy.io as scio

import cv2

sys.path.append('../')
np.set_printoptions(suppress=True)


def get_glgcm_features(mat):
    '''根据灰度梯度共生矩阵计算纹理特征量，包括小梯度优势，大梯度优势，灰度分布不均匀性，梯度分布不均匀性，能量，灰度平均，梯度平均，
    灰度方差，梯度方差，相关，灰度熵，梯度熵，混合熵，惯性，逆差矩'''
    sum_mat = mat.sum()
    small_grads_dominance = big_grads_dominance = gray_asymmetry = grads_asymmetry = energy = gray_mean = grads_mean = 0
    gray_variance = grads_variance = corelation = gray_entropy = grads_entropy = entropy = inertia = differ_moment = 0
    for i in range(mat.shape[0]):
        gray_variance_temp = 0
        for j in range(mat.shape[1]):
            small_grads_dominance += mat[i][j] / ((j + 1) ** 2)
            big_grads_dominance += mat[i][j] * j ** 2
            energy += mat[i][j] ** 2
            if mat[i].sum() != 0:
                gray_entropy -= mat[i][j] * np.log(mat[i].sum())
            if mat[:, j].sum() != 0:
                grads_entropy -= mat[i][j] * np.log(mat[:, j].sum())
            if mat[i][j] != 0:
                entropy -= mat[i][j] * np.log(mat[i][j])
                inertia += (i - j) ** 2 * np.log(mat[i][j])
            differ_moment += mat[i][j] / (1 + (i - j) ** 2)
            gray_variance_temp += mat[i][j] ** 0.5

        gray_asymmetry += mat[i].sum() ** 2
        gray_mean += i * mat[i].sum() ** 2
        gray_variance += (i - gray_mean) ** 2 * gray_variance_temp
    for j in range(mat.shape[1]):
        grads_variance_temp = 0
        for i in range(mat.shape[0]):
            grads_variance_temp += mat[i][j] ** 0.5
        grads_asymmetry += mat[:, j].sum() ** 2
        grads_mean += j * mat[:, j].sum() ** 2
        grads_variance += (j - grads_mean) ** 2 * grads_variance_temp
    small_grads_dominance /= sum_mat
    big_grads_dominance /= sum_mat
    gray_asymmetry /= sum_mat
    grads_asymmetry /= sum_mat
    gray_variance = gray_variance ** 0.5
    grads_variance = grads_variance ** 0.5
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            corelation += (i - gray_mean) * (j - grads_mean) * mat[i][j]
    glgcm_features = [small_grads_dominance, big_grads_dominance, gray_asymmetry, grads_asymmetry, energy, gray_mean,
                      grads_mean,
                      gray_variance, grads_variance, corelation, gray_entropy, grads_entropy, entropy, inertia,
                      differ_moment]
    return np.round(glgcm_features, 4)


def loadSurfFeatures(imageFileName, dataSetName, folderName):
    num = imageFileName.split('.')[0].split('_')[1]
    fileName = '../data/office/office31_surf/' + dataSetName + '/interest_points/' + folderName + '/histogram_' + num + '.SURF_SURF.amazon_800.SURF_SURF.mat'  # todo: why they are all in amazon
    # print fileName
    try:
        mat = scio.loadmat(fileName)
        # print mat.keys()
        A = mat['histogram']
        return A
    except:
        return None
    # print mat['descriptor_vq'].shape
    # print mat['y'].shape
    # print mat['x'].shape


def glgcm(img_gray, ngrad=16, ngray=16):
    gsx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    gsy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    height, width = img_gray.shape
    grad = (gsx ** 2 + gsy ** 2) ** 0.5  # 计算梯度值
    grad = np.asarray(1.0 * grad * (ngrad - 1) / grad.max(), dtype=np.int16)
    gray = np.asarray(1.0 * img_gray * (ngray - 1) / img_gray.max(), dtype=np.int16)  # 0-255变换为0-15
    gray_grad = np.zeros([ngray + 1, ngrad + 1])  # 灰度梯度共生矩阵
    for i in range(height):
        for j in range(width):
            gray_value = gray[i][j]
            grad_value = grad[i][j]
            gray_grad[gray_value][grad_value] += 1
    gray_grad = gray_grad.reshape(1, 16 * 16)

    # if you just need glgcm matrix:
    return gray_grad
    # if you need the statistcs  from glgcm matrix:
    gray_grad = 1.0 * gray_grad / (height * width)  # 归一化灰度梯度矩阵，减少计算量
    glgcm_features = get_glgcm_features(gray_grad)
    return glgcm_features


def get_OFFICE_31_features():
    dic = {'calculator': 5, 'ring_binder': 27, 'printer': 12,
           'keyboard': 30, 'scissors': 26, 'laptop_computer': 7,
           'mouse': 18, 'monitor': 3, 'mug': 24,
           'tape_dispenser': 17, 'pen': 19, 'bike': 10,
           'speaker': 8, 'back_pack': 2, 'desktop_computer': 22,
           'punchers': 15, 'mobile_phone': 0, 'paper_notebook': 1,
           'ruler': 23, 'letter_tray': 9, 'file_cabinet': 16,
           'phone': 25, 'bookcase': 20, 'projector': 4,
           'stapler': 13, 'trash_can': 11, 'bike_helmet': 28,
           'headphones': 14, 'desk_lamp': 6, 'desk_chair': 21,
           'bottle': 29}

    input_path = '../data/office/Original_images/webcam/images/'
    webcam_feature = []
    webcam = []
    webcam_surf = []
    webcam_label = []
    for filename in os.listdir(input_path):
        fi = input_path + filename
        if fi.find('.DS') == -1:
            print fi
            for img_path in os.listdir(fi):
                surfFeatures = loadSurfFeatures(img_path, 'webcam', filename)
                if surfFeatures is not None:
                    img = cv2.imread(fi + '/' + img_path, 0)
                    feature = glgcm(img, 15, 15)
                    webcam_feature.append(feature)
                    webcam_surf.append(surfFeatures.reshape(surfFeatures.shape[1]))
                    webcam.append(img)
                    webcam_label.append(dic[filename])
    webcam_feature = np.array(webcam_feature)
    webcam = np.array(webcam)
    webcam_label = np.array(webcam_label)
    webcam_surf = np.array(webcam_surf)

    input_path = '../data/office/Original_images/amazon/images/'
    amazon_feature = []
    amazon = []
    amazon_surf = []
    amazon_label = []

    for filename in os.listdir(input_path):
        fi = input_path + filename
        if fi.find('.DS') == -1:
            print fi
            for img_path in os.listdir(fi):
                surfFeatures = loadSurfFeatures(img_path, 'amazon', filename)
                if surfFeatures is not None:
                    img = cv2.imread(fi + '/' + img_path, 0)
                    feature = glgcm(img, 15, 15)
                    amazon_feature.append(feature)
                    amazon.append(img)
                    amazon_surf.append(surfFeatures.reshape(surfFeatures.shape[1]))
                    amazon_label.append(dic[filename])
    amazon_feature = np.array(amazon_feature)
    amazon = np.array(amazon)
    amazon_label = np.array(amazon_label)
    amazon_surf = np.array(amazon_surf)

    input_path = '../data/office/Original_images/dslr/images/'
    dslr_feature = []
    dslr = []
    dslr_label = []
    dslr_surf = []
    for filename in os.listdir(input_path):
        fi = input_path + filename
        if fi.find('.DS') == -1:
            print fi
            for img_path in os.listdir(fi):
                surfFeatures = loadSurfFeatures(img_path, 'dslr', filename)
                if surfFeatures is not None:
                    img = cv2.imread(fi + '/' + img_path, 0)
                    feature = glgcm(img, 15, 15)
                    dslr_feature.append(feature)
                    dslr.append(img)
                    dslr_surf.append(surfFeatures.reshape(surfFeatures.shape[1]))
                    dslr_label.append(dic[filename])

    dslr_feature = np.array(dslr_feature)
    dslr = np.array(dslr)
    dslr_label = np.array(dslr_label)
    dslr_surf = np.array(dslr_surf)

    return webcam_surf, webcam_feature, webcam_label, amazon_surf, amazon_feature, amazon_label, dslr_surf, dslr_feature, dslr_label

def organizeFeatures(dataSet):
    imgs = np.load('../data/office/numpyData/'+dataSet+'.npy')
    # resnetFeature = np.load('../data/office/numpyData/'+dataSet+'_feature.npy')
    # labels = np.load('../data/office/numpyData/'+dataSet+'_label.npy')

    glgcmResults = []
    for i in range(imgs.shape[0]):
        # print imgs[i][0][0]
        gray_image = cv2.cvtColor(imgs[i].astype(np.uint8), cv2.COLOR_BGR2GRAY)
        g = glgcm(gray_image, 15, 15)
        glgcmResults.append(g.reshape([256]))
    glgcmResults = np.array(glgcmResults)
    np.save('../data/office/numpyData/' + dataSet + '_glgcm.npy', glgcmResults)

def checkDimension(dataSet):
    imgs = np.load('../data/office/numpyData/'+dataSet+'.npy')
    resnetFeature = np.load('../data/office/numpyData/'+dataSet+'_feature.npy')
    labels = np.load('../data/office/numpyData/'+dataSet+'_label.npy')

    print imgs.shape
    print resnetFeature.shape
    print labels.shape



if __name__ == '__main__':
    # webcam_surf, webcam_feature, webcam_label, amazon_surf, amazon_feature, amazon_label, dslr_surf, dslr_feature, dslr_label = get_OFFICE_31_features()
    # print webcam_surf.shape
    # print amazon_surf.shape
    # print dslr_surf.shape
    #
    # print webcam_feature.shape
    # print amazon_feature.shape
    # print dslr_feature.shape
    #
    # print webcam_label.shape
    # print amazon_label.shape
    # print dslr_label.shape
    #
    # np.save('../data/office/webcam_surf', webcam_surf)
    # np.save('../data/office/webcam_glgcm', webcam_feature)
    # np.save('../data/office/webcam_label', webcam_label)
    # np.save('../data/office/amazon_surf', amazon_surf)
    # np.save('../data/office/amazon_glgcm', amazon_feature)
    # np.save('../data/office/amazon_label', amazon_label)
    # np.save('../data/office/dslr_surf', dslr_surf)
    # np.save('../data/office/dslr_glgcm', dslr_feature)
    # np.save('../data/office/dslr_label', dslr_label)

    organizeFeatures('amazon')
    organizeFeatures('dslr')
    organizeFeatures('webcam')

    # checkDimension('amazon')
    # checkDimension('dslr')
    # checkDimension('webcam')
