__author__ = 'Haohan Wang'

import numpy as np
from HEX_linear import Hex_linear
from Lasso import Lasso

from sklearn.metrics import mean_squared_error as mse, accuracy_score

from matplotlib import pyplot as plt

n = 200
p = 10
k = 10
e = 1
corr = 0.8

logisticFlag = True

def evaluate(y1, y2, logistic=False):
    if not logistic:
        return mse(y1, y1)
    else:
        return accuracy_score(y1.astype(int), y2.astype(int))

def binary_transform(y):
    m = np.exp(y)
    m = m/(1+m)
    k = []
    for i in range(m.shape[0]):
        k.append(np.random.binomial(1, m[i], 1)[0])
    return np.array(k)

def predict(X, beta, logistic=False):
    if not logistic:
        return np.dot(X, beta)
    else:
        t = 1. / (1 + np.exp(-np.dot(X, beta)))
        y = np.zeros_like(t)
        y[t>0.5] = 1
        return y

def generateData():
    X1 = np.random.normal(size=[2*n, p])
    X2 = np.random.normal(size=[2*n, p])

    # X2[:int(n*corr), :] = X1[:int(n*corr), :]

    X = np.append(X1, X2, axis=1)

    # plt.imshow(X)
    # plt.show()

    Xtr = X[:n, :]
    Xte = X[n:, :]

    beta1 = np.random.random(k) + 1
    beta2 = np.random.random(k) + 1

    ytr = (np.dot(Xtr[:,:k], beta1) + np.dot(Xtr[:, p:p+k], beta2))/2 #+ np.random.normal(size=[n])
    yte = np.dot(Xte[:,:k], beta1) #+ np.random.normal(size=[n])

    ytr = binary_transform(ytr)
    yte = binary_transform(yte)

    Z = np.random.normal(size=[n, 2*p])
    Zte = np.dot(Z[:, p:p+k], beta2) #+ np.random.normal(size=[n])

    Zte = binary_transform(Zte)

    return Xtr, Xte, ytr, yte, Z, Zte, beta1

def run():
    Xtr, Xte, ytr, yte, Z, Zte, beta1 = generateData()

    print '-----------------'

    m0 = Lasso(lam=0, lr=1e0, logistic=False)
    m0.fit(Xtr[:, :p], ytr)
    yte0 = predict(Xte[:,:p], m0.getBeta(), logistic=logisticFlag)
    zte0 = predict(Z[:,:p], m0.getBeta(), logistic=logisticFlag)

    print '-----------------'
    m1 = Lasso(lam=0, lr=1e0, logistic=False)
    m1.fit(Xtr, ytr)
    yte1 = predict(Xte[:,:p], m1.getBeta()[:p], logistic=logisticFlag)
    zte1 = predict(Z[:,:p], m1.getBeta()[:p], logistic=logisticFlag)
    #
    print '-----------------'

    m2 = Hex_linear(hex_start=np.inf, ignoringIndex=p, lam=0, lr=1e0, project=True, logistic=False)
    m2.fit(Xtr, ytr)
    yte2 = predict(Xte[:,:p], m2.getBeta()[:p], logistic=logisticFlag)
    zte2 = predict(Z[:,:p], m2.getBeta()[:p], logistic=logisticFlag)

    print '-----------------'

    m3 = Hex_linear(hex_start = np.inf, ignoringIndex=p, lam=0, lr=1e0, logistic=False) #1e2 works OK
    m3.fit(Xtr, ytr)
    yte3 = predict(Xte[:,:p], m3.getBeta()[:p], logistic=logisticFlag)
    zte3 = predict(Z[:,:p], m3.getBeta()[:p], logistic=logisticFlag)

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
    # print beta1[:10]
    # print m0.getBeta()[:20]
    # print m1.getBeta()[:20]
    # print m2.getBeta()[:20]
    # print m3.getBeta()[:20]
    # print '--------------'

    a0 = evaluate(yte, yte0, logistic=logisticFlag)
    a1 = evaluate(yte, yte1, logistic=logisticFlag)
    a2 = evaluate(yte, yte2, logistic=logisticFlag)
    a3 = evaluate(yte, yte3, logistic=logisticFlag)

    b0 = evaluate(Zte, zte0, logistic=logisticFlag)
    b1 = evaluate(Zte, zte1, logistic=logisticFlag)
    b2 = evaluate(Zte, zte2, logistic=logisticFlag)
    b3 = evaluate(Zte, zte3, logistic=logisticFlag)

    return a0, a1, a2, a3, b0, b1, b2, b3

if __name__ == '__main__':
    for seed in range(10):
        np.random.seed(seed)
        print 'Seed', seed, '\t',
        m = run()
        for a in m:
            print a, '\t',
        print
    # np.random.seed(2)
    # m = run()
    # for a in m:
    #     print a, '\t',
