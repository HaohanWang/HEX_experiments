# addBackground_28.py
import cv2

import os

import numpy as np

# background_path = '/Users/hzxue/Desktop/CMU/project/artificial-pattern/data/background/'
# face_path = '/Users/hzxue/Desktop/CMU/project/artificial-pattern/data/original/'
background_path = '../images/background/'
face_path = '/media/haohanwang/Data/SentimentImages/original/'

sentiment_dic = {'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3, 'neutral': 4, 'sadness': 5, 'surprise': 6}
background_dic = {0: 'drawn.jpg', 1: 'festival.jpg', 2: 'firework.jpg', 3: 'forest.jpg', 4: 'scare.jpg', 5: 'train.jpg', 6: 'winter.jpg'}

names = {'aia', 'bonnie', 'jules', 'malcolm', 'mery', 'ray'}
"""
left right and middle no need to change
"""
def middle(img1,img2):
    img_mix = np.zeros((512,512, 3), np.uint8)
    h,w,_=img2.shape
    for i in range(h):
        for j in range(w):
            if i>=0 and i<256 and j>=0 and j<512:
                img_mix[i,j]=img2[i,j]
            else:
                if i>=256 and i<512 and j >= 0 and j<128:
                    img_mix[i,j]=img2[i,j]
                else:
                    if i>=256 and i<512 and j>=383 and j<512:
                        img_mix[i,j]=img2[i,j]
                    else:
                        #print i,j
                        (r1,g1,b1,a1)=img1[i-256,j-128]
                        #print img1[i,j]
                        if  a1==0:
                            img_mix[i,j]=img2[i,j]
                        else:
                            img_mix[i,j]=(r1,g1,b1)        
    return img_mix

def left(img1,img2):
    img_mix = np.zeros((512,512, 3), np.uint8)

    h,w,_=img2.shape
    for i in range(h):
        for j in range(w):
            if i>=0 and i<256 and j>=0 and j<512:
                # upper
                img_mix[i,j]=img2[i,j]
            else:
                if i>=256 and i<512 and j >= 256 and j<512:
                    img_mix[i,j]=img2[i,j]
                    #left down
                else:
                        #middle
                        (r1,g1,b1,a1)=img1[i-256,j]
                        #print img1[i,j]
                        if a1==0:
                            img_mix[i,j]=img2[i,j]
                        else:
                            img_mix[i,j]=(r1,g1,b1)
    return img_mix 

def right(img1,img2): 
    img_mix = np.zeros((512,512, 3), np.uint8)
    h,w,_=img2.shape
    for i in range(h):
        for j in range(w):
            if i>=0 and i<256 and j>=0 and j<512:
                # upper
                img_mix[i,j]=img2[i,j]
            else:
                if i>=256 and i<512 and j<256:
                    img_mix[i,j]=img2[i,j]
                    #left down
                else:
                        #middle
                        (r1,g1,b1,a1)=img1[i-256,j-256]
                        #print img1[i,j]
                        if a1==0:
                            img_mix[i,j]=img2[i,j]
                        else:
                            img_mix[i,j]=(r1,g1,b1)
    return img_mix 

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
    #print backgroundpath,savepath,facepath
    # global total

    ind = int(facepath.split('_')[-1].split('.')[0])
    # 80% data with probability of 0.8 are associate with label[0,1,2,3,4,5,6,7]
    """  """
    if ind % 10 < 8:
        if np.random.random() < corr:
            img2 = bgps[sentiment_dic[sent]]
        else:
            i = np.random.randint(7)
            img2 = bgps[i]
    else:
        # [8,9] random
        i = np.random.randint(7)
        img2 = bgps[i] 
    img1 = cv2.imread(facepath, cv2.IMREAD_UNCHANGED)
    h, w, _ = img1.shape
    img_mix = np.zeros((512, 512, 3), np.uint8)
    img1 = cv2.resize(img1, (256, 256), interpolation=cv2.INTER_CUBIC)
    img2 = cv2.resize(img2, (512, 512), interpolation=cv2.INTER_CUBIC)
    x = np.random.randint(3)
    if x == 0:
        img_mix = middle(img1,img2)
    if x == 1:
        img_mix = left(img1,img2)
    if x == 2:
        img_mix = right(img1,img2)
    # img_mix=cv2.cvtColor(img_mix,cv2.COLOR_RGB2GRAY)
    # img_mix=cv2.resize(img_mix,(28,28),interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(savepath, img_mix)
    #print savepath
    return

def loadBackground():
    bgps = {}
    for k in background_dic:
        bgp = background_path + background_dic[k]
        img2 = cv2.imread(bgp, cv2.IMREAD_UNCHANGED)
        #print bgp
        bgps[k] = img2
    return bgps

def run(corr=0.8):
    count = 0
    bgps = loadBackground()
    """
    if corr == 0:
        c = 0
    elif corr == 0.4:
        c = 4
    else:
        c= 8
    """
    c=int(corr*10)
    # save_path = '/Users/hzxue/Desktop/CMU/project/artificial-pattern/data/background_'+str(c) + '/'
    save_path = '/media/haohanwang/Data/SentimentImages/background_'+str(c) + '/'
    print save_path
    for n in names:
        for k in sentiment_dic:
            inputPath = face_path+n+'/'+n+'_'+k+'/'
            for r, d, f in os.walk(inputPath):
                for fn in f:
                    if fn.find('Store')==-1:
                        count += 1
                        # print count, '\t',
                        add_image(inputPath+fn, bgps, k, corr, save_path)
                        if count%1000 == 0:
                            print '============================='
                            print 'We have worked on ', count, 'images'
                            print '============================='


if __name__ == '__main__':
    # np.random.seed(0)
    # corr=0.0
    # while corr<=1.0:
    #     print 'WE ARE WORKING ON', corr
    #     if corr!=0.8:
    #         run(corr=corr)
    #     corr+=0.1
    # run(corr=0.8)
    np.random.seed(0)
    print 'WE ARE WORKING ON', 0.1
    run(corr=0.1)
    np.random.seed(0)
    print 'WE ARE WORKING ON', 0.2
    run(corr=0.2)
    np.random.seed(0)
    print 'WE ARE WORKING ON', 0.3
    run(corr=0.3)
    np.random.seed(0)
    print 'WE ARE WORKING ON', 0.4
    run(corr=0.4)
    np.random.seed(0)
    print 'WE ARE WORKING ON', 0.5
    run(corr=0.5)
    np.random.seed(0)
    print 'WE ARE WORKING ON', 0.6
    run(corr=0.6)
    np.random.seed(0)
    print 'WE ARE WORKING ON', 0.7
    run(corr=0.7)
    np.random.seed(0)
    print 'WE ARE WORKING ON', 0.9
    run(corr=0.9)
