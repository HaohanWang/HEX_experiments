__author__ = 'Haohan Wang'

import numpy as np
from HEX_linear import Hex_linear
from Lasso import Lasso

from sklearn.metrics import mean_squared_error as mse

n = 500
p = 100
k = 10
e = 1

def generateData():
    X1 = np.random.normal(size=[n, p])
    X2 = np.zeros(shape=[n, p])

    X2[:n/2, :] = X1[:n/2, :] + np.random.normal(size=[n/2, p])*e

    X = np.append(X1, X2, axis=1)

    Xtr = X[:n/2, :]
    Xte = X[n/2:, :]

    beta = np.random.random(k)

    ytr = np.dot(Xtr[:, p:p+k], beta) + np.random.normal(size=[n/2])
    yte = np.dot(Xte[:,:k], beta) + np.random.normal(size=[n/2])

    Z = np.random.normal(size=[n, 2*p])
    Zte = np.dot(Z[:, p:p+k], beta) + np.random.normal(size=[n])

    return Xtr, Xte, ytr, yte, Z, Zte


def run():
    Xtr, Xte, ytr, yte, Z, Zte = generateData()

    print '-----------------'
    m = Lasso(lam=0, lr=0.5)
    m.fit(Xtr, ytr)
    yte1 = m.predict(Xte)
    zte1 = m.predict(Z)
    #
    # print '-----------------'
    #
    # m = Hex_linear(hex_start=0, ignoringIndex=p, lam=0, lr=1e0)
    # m.fit(Xtr, ytr)
    # yte2 = m.predict(Xte)

    print '-----------------'

    m = Hex_linear(hex_start = 100, ignoringIndex=p, lam=0, lr=0.5)
    m.fit(Xtr, ytr)
    yte3 = m.predict(Xte)
    zte3 = m.predict(Z)

    print '================'
    print '--------------'
    print mse(yte, yte1)
    # print mse(yte, yte2)
    print mse(yte, yte3)
    print '--------------'
    print '================'
    print '--------------'
    print mse(Zte, zte1)
    # print mse(yte, yte2)
    print mse(Zte, zte3)
    print '--------------'

if __name__ == '__main__':
    np.random.seed(1)

    run()
