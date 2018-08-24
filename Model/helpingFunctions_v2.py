__author__ = 'Haohan Wang'

import scipy.linalg as linalg
import scipy
import numpy as np
from scipy import stats
import scipy.optimize as opt

Wlist = []

def checkInformation_py(X, epoch, s):
    if epoch > 0:
        # print X.shape
        print X[0,:], s
    return np.float32(X)

def generatingWeightMatrix_py(X, y, epoch, division, batch):

    # print np.linalg.matrix_rank(X), '\t', np.linalg.matrix_rank(y),
    # for i in range(10):
    #     print np.mean(X[i,:]), np.where(y[i]==1)[0]

    # X = X.reshape([128, 28*28]) #todo: note to change these back
    # X = X[:, :5]

    batch = int(batch)

    if epoch < division:
        #print epoch
        return np.float32(np.eye(X.shape[0]))
    else:
        # print np.linalg.matrix_rank(X), '\t', np.linalg.matrix_rank(np.dot(X.T, X))
        #
        # for i in range(10):
        #     print np.mean(X[i,:]), np.where(y[i]==1)[0]
        # if len(Wlist) == batch: #todo: let's try to change modelling power
        #
        #     factor, S, U = fitting_null_py(X, y)
        #     W = np.linalg.inv(np.dot(np.dot(U, np.diag(S)), U.T)*factor+np.eye(X.shape[0]))
        #
        #     # W = np.eye(X.shape[0])
        #     # W = W/np.mean(W)  # this line was not there in the sentiment experiment
        #
        #     # W = columnWiseNormalize(W)
        #     # W = columnWiseNormalize(W.T).T
        #
        #     # W = np.diag(np.diag(W))
        #
        #     # W = np.eye(X.shape[0]) - np.dot(X, np.dot(np.linalg.inv(np.dot(X.T, X)), X.T))
        #     # W = columnWiseNormalize(W)
        #     # W = columnWiseNormalize(W.T).T
        #     #
        #
        #     Wlist.append(W)
        #
        # return np.float32(Wlist[batch])


        factor, S, U = fitting_null_py(X, y)
        W = np.linalg.inv(np.dot(np.dot(U, np.diag(S)), U.T)*factor+np.eye(X.shape[0]))
        # #
        # # # W = np.eye(X.shape[0])
        # # # W = W/np.mean(W)
        # #
        # # W = columnWiseNormalize(W)
        # # W = columnWiseNormalize(W.T).T
        # #
        # X = X.reshape([X.shape[0], 1])
        # W = np.eye(X.shape[0]) - np.dot(X, np.dot(np.linalg.inv(np.dot(X.T, X)), X.T))
        # # # W = columnWiseNormalize(W)
        # # # W = columnWiseNormalize(W.T).T
        #
        return np.float32(W)

def rescale(a):
    return a / np.max(np.abs(a))

def selectValues(Kva):
    r = np.zeros_like(Kva)
    n = r.shape[0]
    tmp = rescale(Kva)
    ind = 0
    for i in range(n/2, n-2):
        if tmp[i + 1] - tmp[i] > 1.0 / n:
            ind = i + 1
            break
    r[ind:] = Kva[ind:]
    r[n - 1] = Kva[n - 1]
    return r

def columnWiseNormalize(X):
    # col_norm = 1.0/np.sqrt((1.0/X.shape[0])*np.diag(np.dot(X.T, X)))
    # return np.dot(X, np.diag(col_norm))
    [n, p] = X.shape
    col_norm = np.ones(X.shape[1])
    for i in range(p):
        s = (1.0/n)*np.dot(X[:, i].T, X[:,i])
        if s != 0:
            col_norm[i] = 1.0/np.sqrt(s)
            X[:, i] = X[:,i]*col_norm[i]
    return X

def fitting_null_py(X, y):
    ldeltamin = -5
    ldeltamax = 5
    numintervals=500

    # X = X.reshape([X.shape[0], 1])
    X = columnWiseNormalize(X)
    xmean = np.mean(X, 0)
    X = X - xmean
    y = np.argmax(y, axis=1)
    y = y - np.mean(y)
    y = y.reshape([y.shape[0], 1])
    y = columnWiseNormalize(y)

    # print y.T

    # ynorm = np.linalg.norm(y, ord=2, axis=0)
    # y = y / ynorm

    K = np.dot(X, X.T)
    S, U = linalg.eigh(K)

    # S = selectValues(S)
    # print S
    #
    # print np.linalg.matrix_rank(K)
    #
    # print len(np.where(S!=0)[0])

    # from matplotlib import pyplot as plt
    # plt.imshow(K)
    # plt.savefig('tmp.png')
    # plt.clf()

    Uy = scipy.dot(U.T, y)

    # grid search
    nllgrid = scipy.ones(numintervals + 1) * scipy.inf
    ldeltagrid = scipy.arange(numintervals + 1) / (numintervals * 1.0) * (ldeltamax - ldeltamin) + ldeltamin
    for i in scipy.arange(numintervals + 1):
        nllgrid[i] = nLLeval(ldeltagrid[i], Uy, S) # the method is in helpingMethods

    # nllmin = nllgrid.min()
    ldeltaopt_glob = ldeltagrid[nllgrid.argmin()]

    print ldeltaopt_glob,
    return np.float32(1.0/np.exp(ldeltaopt_glob)), S, U

def nLLeval(ldelta, Uy, S, REML=False):
    """
    evaluate the negative log likelihood of a random effects model:
    nLL = 1/2(n_s*log(2pi) + logdet(K) + 1/ss * y^T(K + deltaI)^{-1}y,
    where K = USU^T.
    Uy: transformed outcome: n_s x 1
    S:  eigenvectors of K: n_s
    ldelta: log-transformed ratio sigma_gg/sigma_ee
    """
    n_s = Uy.shape[0]
    delta = scipy.exp(ldelta)

    # evaluate log determinant
    Sd = S + delta
    ldet = scipy.sum(scipy.log(Sd))

    # evaluate the variance
    Sdi = 1.0 / Sd
    # Uy = Uy.flatten()
    # ss = 1. / n_s * (Uy.dot(Uy.T).dot(np.diag(Sdi))).sum()
    ss = 1. / n_s * (Uy*Uy*(Sdi.reshape(-1, 1))).sum()
    ss = ss / Uy.shape[1] + 1e-5

    # evalue the negative log likelihood
    nLL = 0.5 * (n_s * np.log(2.0 * scipy.pi) + ldet + n_s + n_s * np.log(ss))

    if REML:
        pass

    return nLL

if __name__ == '__main__':
    from dataGeneration.dataGeneration import *
    X = dataGeneration_SNP(n=500, p=1000)
    X = discreteMapping(X)
    from matplotlib import pyplot as plt
    # print np.corrcoef(X.T)
    plt.imshow(X)
    plt.show()
    # X = np.random.random([500, 1000])
    b = np.zeros([1000, 1])
    for i in range(100):
        b[i] = np.random.random()
    y = np.dot(X, b)
    y = binary_transform(y)
    y = y.reshape([y.shape[0], 1])
    fitting_null_py(X, y)
