__author__ = 'Haohan Wang'

import numpy as np
from HEX_linear import Hex_linear
from Lasso import Lasso

from sklearn.metrics import mean_squared_error as mse

from matplotlib import pyplot as plt

n = 200
p = 10
k = 10
e = 1
corr = 0.8

def generateData():
    X1 = np.random.normal(size=[2*n, p])
    X2 = np.random.normal(size=[2*n, p])

    X2[:int(n*corr), :] = X1[:int(n*corr), :]

    X = np.append(X1, X2, axis=1)

    # plt.imshow(X)
    # plt.show()

    Xtr = X[:n, :]
    Xte = X[n:, :]

    beta1 = np.random.random(k) + 1
    beta2 = np.random.random(k) + 1

    ytr = (np.dot(Xtr[:,:k], beta1) + np.dot(Xtr[:, p:p+k], beta2))/2 #+ np.random.normal(size=[n])
    yte = np.dot(Xte[:,:k], beta1) #+ np.random.normal(size=[n])

    Z = np.random.normal(size=[n, 2*p])
    Zte = np.dot(Z[:, p:p+k], beta2) + np.random.normal(size=[n])

    return Xtr, Xte, ytr, yte, Z, Zte

def predict(X, model):
    beta = model.getBeta()[:p]
    return np.dot(X[:,:p], beta)

def run():
    Xtr, Xte, ytr, yte, Z, Zte = generateData()

    # print '-----------------'

    m0 = Lasso(lam=0, lr=1)
    m0.fit(Xtr[:, :p], ytr)
    yte0 = m0.predict(Xte[:,:p])
    zte0 = m0.predict(Z[:,:p])

    # print '-----------------'
    m1 = Lasso(lam=0, lr=1)
    m1.fit(Xtr, ytr)
    yte1 = predict(Xte, m1)
    zte1 = predict(Z, m1)
    #
    # print '-----------------'

    m2 = Hex_linear(hex_start=np.inf, ignoringIndex=p, lam=0, lr=1, project=True)
    m2.fit(Xtr, ytr)
    yte2 = predict(Xte, m2)
    zte2 = predict(Z, m2)

    # print '-----------------'

    m3 = Hex_linear(hex_start = 1e3, ignoringIndex=p, lam=0, lr=1) #1e2 works OK
    m3.fit(Xtr, ytr)
    yte3 = predict(Xte, m3)
    zte3 = predict(Z, m3)

    # print '================'
    # print '--------------'
    # print mse(yte, yte0)
    # print mse(yte, yte1)
    # # print mse(yte, yte2)
    # print mse(yte, yte3)
    # print '--------------'
    # print '================'
    # print '--------------'
    # print mse(Zte, zte0)
    # print mse(Zte, zte1)
    # # print mse(yte, yte2)
    # print mse(Zte, zte3)
    # print '--------------'
    # print '================'
    # print '--------------'
    # print m0.getBeta()
    # print m1.getBeta()
    # print m3.getBeta()
    # print '--------------'

    a0 = mse(yte, yte0)
    a1 = mse(yte, yte1)
    a2 = mse(yte, yte2)
    a3 = mse(yte, yte3)

    b0 = mse(Zte, zte0)
    b1 = mse(Zte, zte1)
    b2 = mse(Zte, zte2)
    b3 = mse(Zte, zte3)

    return a0, a1, a2, a3, b0, b1, b2, b3

if __name__ == '__main__':
    for seed in range(10):
        np.random.seed(seed)
        print 'Seed', seed, '\t',
        m = run()
        for a in m:
            print a, '\t',
        print