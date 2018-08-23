__author__ = 'Haohan Wang'

from dataLoader import *

seed = 0
corr = 0.8

# n = 500
# p = 1000
# group = 2
#
# Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = loadData(seed, n, p, corr, group)
#
# from matplotlib import pyplot as plt
#
# plt.scatter(xrange(n), Ytrain[:,0])
# # plt.scatter(xrange(n), Ytest[:,0]+0.1)
# plt.ylim(-0.1, 1.2)
# plt.show()
#
# X = np.append(Xtrain, Xval, 0)
# X = np.append(X, Xtest, 0)
# plt.imshow(X)
# plt.show()
#
# d = np.dot(Xtrain, Xtrain.T)
# np.matrix.sort(d, 0)
# np.matrix.sort(d, 1)
#
# plt.imshow(d)
# plt.show()

r = np.load('../Model/results_useful.npy')
print r