__author__ = 'Haohan Wang'

import cv2
import numpy as np

def loadImages(cn):
    results = []
    for dn in ['amazon', 'dslr', 'webcam']:
        r = []
        for i in range(1, 10):
            filename = '../data/office/Original_images/'+dn+'/images/'+cn+'/frame_000'+str(i) + '.jpg'
            print filename
            img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
            r.append(img)
        results.append(r)
    return results

def writeMergeImages(cn):
    Images = np.zeros([260*3, 260*9, 3])
    results = loadImages(cn)
    for i in range(len(results)):
        for j in range(len(results[i])):
            Images[1+i*260:1+i*260+256, 1+j*260:1+j*260+256,:] = results[i][j]

    cv2.imwrite('resultImages/'+cn+'.jpg', Images)



if __name__ == '__main__':
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
    for cn in dic:
        try:
            writeMergeImages(cn)
        except:
            pass