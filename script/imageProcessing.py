# -*- encoding=utf-8 -*-
import cv2
import numpy as np
import math

def run(ngray=16):
    img = cv2.imread('dog.jpg', cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)

    print ("deal date with ngray=%d..." % (ngray))
    row = 0
    column = 1
    direction=np.diag((-1)*np.ones(256*256))
    for i in range(256*256):
        x=int(math.floor(i/256))
        y=int(i%256)
        if x+row<256 and y+column<256:
            direction[i][i+row*256+column]=1

    imgray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    re = np.copy(imgray).reshape(256*256)
    d = np.copy(imgray).reshape(256*256)

    re = np.asarray(1.0 * re * (ngray-1) / re.max(), dtype=np.int16)
    d =np.dot(re,direction)

    re = re.reshape([256, 256])
    d = d.reshape([256, 256])

    re = 256*re/float(re.max())
    d = 256*d/float(d.max())


    cv2.imwrite('x.jpg', img)
    cv2.imwrite('re.jpg', re)
    cv2.imwrite('d.jpg', d)

if __name__ == '__main__':
    run()
