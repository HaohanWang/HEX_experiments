__author__ = 'Haohan Wang'

import cv2

import os

import numpy as np

background_path = '../images/background/'
face_path = '/media/haohanwang/Info/SentimentImages/original/'

sentiment_dic = {'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3, 'neutral': 4, 'sadness': 5, 'surprise': 6}
background_dic = {0: 'drawn.jpg', 1: 'festival.jpg', 2: 'firework.jpg', 3: 'forest.jpg', 4: 'scare.jpg', 5: 'train.jpg', 6: 'winter.jpg'}

names = {'aia', 'bonnie', 'jules', 'malcolm', 'mery', 'ray'}

def add_image(facepath, bgps, sent, corr, save_path):
    files = facepath.split('/')
    savepath = os.path.join(save_path, files[-3])
    if os.path.exists(savepath) is not True:
        os.makedirs(savepath)
    savepath = os.path.join(savepath, files[-2])
    if os.path.exists(savepath) is not True:
        os.makedirs(savepath)
    savepath = os.path.join(savepath, files[-1])

    if os.path.exists(savepath) is True:
        return
        # print backgroundpath,savepath,facepath
    # global total

    ind = int(facepath.split('_')[-1].split('.')[0])
    if ind % 10 < 8:
        if np.random.random() < corr:
            img2 = bgps[sentiment_dic[sent]]
        else:
            i = np.random.randint(7)
            img2 = bgps[i]
    else:
        i = np.random.randint(7)
        img2 = bgps[i]

    img1 = cv2.imread(facepath, cv2.IMREAD_UNCHANGED)
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
    print savepath
    return

def loadBackground():
    bgps = {}
    for k in background_dic:
        bgp = background_path + background_dic[k]
        img2 = cv2.imread(bgp, cv2.IMREAD_UNCHANGED)
        bgps[k] = img2
    return bgps

def run(corr=0.8):
    count = 0
    bgps = loadBackground()
    if corr == 0:
        c = 0
    elif corr == 0.4:
        c = 4
    else:
        c= 8
    save_path = '/media/haohanwang/Info/SentimentImages/background_'+str(c) + '/'
    for n in names:
        for k in sentiment_dic:
            inputPath = face_path+n+'/'+n+'_'+k+'/'
            for r, d, f in os.walk(inputPath):
                for fn in f:
                    count += 1
                    print count, '\t',
                    add_image(inputPath+fn, bgps, k, corr, save_path)
                    if count%1000 == 0:
                        print '============================='
                        print 'We have worked on ', count, 'images'
                        print '============================='


if __name__ == '__main__':
    corr = 0.8
    run(corr=corr)