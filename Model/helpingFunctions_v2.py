__author__ = 'Haohan Wang'

import scipy.linalg as linalg
import scipy
import numpy as np
from scipy import stats
import scipy.optimize as opt

Wlist = []

def checkInformation_py(X, epoch, s, y):
    if epoch > 0:
        # print X.shape
        print X[0,:], np.argmax(y[0,:]), s
    return np.float32(np.diag(np.ones(64)))

def generatingWeightMatrix_py(Xp, Xc, epoch, division, batch):

    if epoch < division:
        return np.float32(Xc)
    else:
        W = np.eye(Xc.shape[0]) - np.dot(Xp, np.dot(np.linalg.inv(np.dot(Xp.T, Xp)), Xp.T))
        Xc = np.dot(Xc, W)
        return np.float32(Xc)

    # lam = 1e-2
    #
    # if epoch < division:
    #     #print epoch
    #     return np.float32(0)
    #
    # p_pred = np.argmax(X, 1)
    # p_prob = np.max(X, 1)
    #
    # c_pred = np.argmax(y, 1)
    # c_prob = np.max(y, 1)
    #
    # # corr = np.dot(p_pred-np.mean(p_pred), c_pred-np.mean(c_pred))/(np.std(p_pred)*np.std(c_pred)*(X.shape[0]))
    #
    # a = np.zeros(X.shape[0])
    #
    # a[p_pred==c_pred] = 1
    # # print np.mean(a),
    # accu = np.mean(a)
    # print accu,
    # a[p_pred==c_pred] = c_prob[p_pred==c_pred]**2
    # # if batch == 0:
    # #     print a
    # if accu <=1.0/7 or np.isnan(accu):
    #     return np.float32(np.ones(X.shape[0]))
    # a += 1/accu
    # a = 1.0/a
    #
    # a = (a/np.sum(a))*X.shape[0]
    # return np.float32(a)

def generatingWeightMatrix_py2(X, y, epoch, division, batch):

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

    X = X.reshape([X.shape[0], 1])
    # X = columnWiseNormalize(X)
    # xmean = np.mean(X, 0)
    # X = X - xmean
    y = np.argmax(y, axis=1)
    # y = y - np.mean(y)
    y = y.reshape([y.shape[0], 1])
    # y = columnWiseNormalize(y)

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
