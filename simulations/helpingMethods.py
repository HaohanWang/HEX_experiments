__author__ = 'Haohan Wang'

import scipy.linalg as linalg
import scipy
import numpy as np
from scipy import stats
import scipy.optimize as opt

def generatingWeightMatrix_py(X, y):
    factor, S, U = fitting_null_py(X, y)
    # print factor
    W = np.linalg.pinv(-np.dot(np.dot(U, np.diag(S)), U.T)*factor+np.eye(X.shape[0])) # todo: pay attention to the extra minus sign here

    # W = np.eye(X.shape[0])
    # W = columnWiseNormalize(W)
    # W = columnWiseNormalize(W.T).T

    return np.float32(W)

def rescale(a):
    return a / np.max(np.abs(a))

def selectValues(Kva):
    r = np.zeros_like(Kva)
    n = r.shape[0]
    tmp = rescale(Kva)
    ind = 0
    for i in range(n-2, n/2, -1):
        if tmp[i + 1] - tmp[i] > 1.0 / n:
            ind = i + 1
            break
    r[ind:] = Kva[ind:]
    r[n - 1] = Kva[n - 1]
    return r

def columnWiseNormalize(X):
    col_norm = 1.0/np.sqrt((1.0/X.shape[0])*np.diag(np.dot(X.T, X)))
    return np.dot(X, np.diag(col_norm))

def fitting_null_py(X, y):
    ldeltamin = -5
    ldeltamax = 5
    numintervals=500

    X = columnWiseNormalize(X)
    xmean = np.mean(X, 0)
    X = X - xmean
    y = columnWiseNormalize(y)
    ymean = np.mean(y, 0)
    y = y - ymean
    # ynorm = np.linalg.norm(y, ord=2, axis=0)
    # y = y / ynorm

    K = np.dot(X, X.T)

    S, U = linalg.eigh(K)

    # S = selectValues(S)

    Uy = scipy.dot(U.T, y)

    # grid search
    nllgrid = scipy.ones(numintervals + 1) * scipy.inf
    ldeltagrid = scipy.arange(numintervals + 1) / (numintervals * 1.0) * (ldeltamax - ldeltamin) + ldeltamin
    for i in scipy.arange(numintervals + 1):
        nllgrid[i] = nLLeval(ldeltagrid[i], Uy, S) # the method is in helpingMethods

    # nllmin = nllgrid.min()
    ldeltaopt_glob = ldeltagrid[nllgrid.argmin()]

    # print ldeltaopt_glob
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