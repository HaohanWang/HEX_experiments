__author__ = 'Haohan Wang'

import numpy as np

def checkScore(cat):
    scores = []
    for i in range(10):
        text = [line.strip() for line in open('../results/MNIST_R/hex_'+str(cat)+'_'+str(i)+'.txt')][-1]
        scores.append(float(text.split()[-1]))
    return np.mean(scores)

if __name__ == '__main__':
    for i in range(6):
        print checkScore(i)