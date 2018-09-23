__author__ = 'Haohan Wang'

import numpy as np

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

from dataLoader import loadDigitClassification

from matplotlib import pyplot as plt

plt.style.use('bmh')

def testScores(cat):
    pl = []
    ll = []
    for i in range(100):
        X = np.load('results/representations_'+cat+'_'+str(i)+'.npy')
        p = np.load('results/patterns_'+cat+'_'+str(i)+'.npy')
        l = np.load('results/labels_'+cat+'_'+str(i)+'.npy')

        print i

        nb = GaussianNB()
        ps = cross_val_score(nb, X, p, cv=3)
        ls = cross_val_score(nb, X, l, cv=3)

        pl.append(np.mean(ps))
        ll.append(np.mean(ls))

    return np.array(pl), np.array(ll)

# def plot():
#     pl, ll = testScores()
#
#     # np.save('pattern_nglcm', pl)
#     # np.save('label_nglcm', ll)
#
#     plt.plot(pl, color='r')
#     plt.plot(ll, color='b')
#     plt.show()

def calculateScores():
    methods = ['mlp', 'nglcm', 'mlp_2', 'nglcm_2']
    colors = ['b', 'r', 'c', 'm']

    for i in range(len(methods)):
        print i
        pl, ll = testScores(methods[i])

        # print m, np.mean(pl), np.std(pl), np.mean(ll), np.std(ll)
        plt.plot(pl, ls='-', color=colors[i])
        plt.plot(ll, ls='-.', color=colors[i])

    plt.show()


if __name__ == '__main__':
    # plot()
    calculateScores()