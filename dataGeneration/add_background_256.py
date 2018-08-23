# -*- encoding=utf-8 -*-
import cv2
# Standard library
import cPickle
import time
import gzip
import zipfile
import os
from PIL import Image

# Third-party libraries
import numpy as np

total = 0
background_path = '/Users/hzxue/Desktop/CMU/project/artificial-pattern/data/new/'
face_path = '/media/haohanwang/Info/SentimentImages/original/'
save_path = '/Users/hzxue/Desktop/CMU/project/artificial-pattern/data/FERG_DB_256_save/'
dic_background = {'anger': 'drawn.jpg', 'disgust': 'festival.jpg', 'fear': 'firework.jpg', 'joy': 'forest.jpg',
                  'neutral': 'scare.jpg', 'sadness': 'train.jpg', 'surprise': 'winter.jpg'}


def add_image(facepath, backgroundpath):
    files = facepath.split('/')
    savepath = os.path.join(save_path, files[-3])
    if os.path.exists(savepath) is not True:
        os.makedirs(savepath)
    savepath = os.path.join(savepath, files[-2])
    if os.path.exists(savepath) is not True:
        os.makedirs(savepath)
    savepath = os.path.join(savepath, files[-1])

    global total
    total += 1

    if os.path.exists(savepath) is True:
        return
        # print backgroundpath,savepath,facepath
    # global total
    img1 = cv2.imread(facepath, cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(backgroundpath, cv2.IMREAD_UNCHANGED)
    h, w, _ = img1.shape
    img_mix = np.zeros((256, 256, 4), np.uint8)
    img1 = cv2.resize(img1, (256, 256), interpolation=cv2.INTER_CUBIC)
    img2 = cv2.resize(img2, (256, 256), interpolation=cv2.INTER_CUBIC)
    for i in range(h):
        for j in range(w):
            (r1, g1, b1, a1) = img1[i, j]
            if a1 == 0:
                (r2, g2, b2) = img2[i, j]
                img_mix[i, j] = (r2, g2, b2, 255)
            else:
                img_mix[i, j] = (r1, g1, b1, 255)

    cv2.imwrite(savepath, img_mix)
    return
    # print savepath


def solve(facepath):
    for i in dic_background:
        if facepath.find(i) != -1:
            file = os.path.join(background_path, dic_background[i])
            add_image(facepath, file)
            return


def gci(filepath):
    files = os.listdir(filepath)
    for fi in files:
        fi_d = os.path.join(filepath, fi)
        if os.path.isdir(fi_d):
            gci(fi_d)
        else:
            if fi_d.find('txt') == -1 and fi_d.find('.DS_Store') == -1:
                solve(fi_d)


gci(face_path)
print total
