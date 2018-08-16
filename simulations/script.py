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

    print '-----------------'

    m0 = Lasso(lam=0, lr=1)
    m0.fit(Xtr[:, :p], ytr)
    yte0 = m0.predict(Xte[:,:p])
    zte0 = m0.predict(Z[:,:p])

    print '-----------------'
    m1 = Lasso(lam=0, lr=1)
    m1.fit(Xtr, ytr)
    yte1 = predict(Xte, m1)
    zte1 = predict(Z, m1)
    #
    # print '-----------------'
    #
    # m = Hex_linear(hex_start=0, ignoringIndex=p, lam=0, lr=1e0)
    # m.fit(Xtr, ytr)
    # yte2 = m.predict(Xte)

    print '-----------------'

    m3 = Hex_linear(hex_start = 300, ignoringIndex=p, lam=0, lr=1)
    m3.fit(Xtr, ytr)
    yte3 = predict(Xte, m3)
    zte3 = predict(Z, m3)

    print '================'
    print '--------------'
    print mse(yte, yte0)
    print mse(yte, yte1)
    # print mse(yte, yte2)
    print mse(yte, yte3)
    print '--------------'
    print '================'
    print '--------------'
    print mse(Zte, zte0)
    print mse(Zte, zte1)
    # print mse(yte, yte2)
    print mse(Zte, zte3)
    print '--------------'
    print '================'
    print '--------------'
    print m0.getBeta()
    print m1.getBeta()
    print m3.getBeta()
    print '--------------'

if __name__ == '__main__':
    np.random.seed(2)

    run()
