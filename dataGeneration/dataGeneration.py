__author__ = 'Haohan Wang'

import numpy as np

def binary_transform(y):
    m = np.exp(y)
    m = m/(1+m)
    k = []
    for i in range(m.shape[0]):
        k.append(np.random.binomial(1, m[i], 1)[0])
    return np.array(k)

def oneHotRepresentation(y):
    n = y.shape[0]
    r = np.zeros([n, 2])
    for i in range(r.shape[0]):
        r[i,y[i]] = 1
    return r

def sampleMultivariateGaussianDiagonalVariance(n, sigma_e, mu=0):
    return np.array([np.random.normal(mu, sigma_e) for i in range(n)])

def sampleData(n, p, corr, mu=0, sigma=1):
    # https://en.wikipedia.org/wiki/Autoregressive_model#Yule.E2.80.93Walker_equations
    c = np.random.normal(mu, sigma, size=n)
    sigma_e = np.sqrt((sigma ** 2) * (1 - corr ** 2))

    # Sample the auto-regressive process.
    signal = [c]
    for _ in range(1, p):
        s = corr * signal[-1] + np.random.normal(mu, sigma_e, size=n)
        signal.append(s)

    return np.array(signal).T

def dataGeneration(n, p):
    X = np.random.normal(size=[n, p])
    # b = np.zeros([p])
    # b[:10] = np.random.random(size=10) + 5
    # y = np.dot(X, b) + np.random.normal(size=[n])

    return X

def dataGeneration_Autoregressive(n, p, corr, group):
    p0 = p/group
    X = None
    for i in range(group):
        X0 = sampleData(n, p0, corr=corr)
        if i == 0:
            X = X0
        else:
            X = np.append(X, X0, 1)

    return X

def dataGeneration_SNP(n, p, popNum=5, groupNum=2,
                               totalGeneration=10,
                               splitGeneration=8, migrationRate=2,
                               MAF=0.1,
                               r1 = 5,
                               r2 = 0.5
                                 ):
    priorCount = np.zeros(shape=[p, 1])

    samples = np.random.multinomial(n - popNum * 2, [1.0 / popNum] * popNum, size=1)[0] + 2
    subSamples = [0]*popNum*groupNum
    for i in range(popNum):
        for j in range(groupNum-1):
            subSamples[i*groupNum+j] = samples[i]/groupNum
        subSamples[i*groupNum+groupNum-1] = samples[i] - sum(subSamples[i*groupNum:i*groupNum+groupNum-1])

    Z = np.zeros([n, len(subSamples)])
    s =-1
    for i in range(len(subSamples)):
        for j in range(subSamples[i]):
            s+=1
            Z[s,i] = 1

    X = None
    for i in range(Z.shape[1]):
        num = len(np.where(Z[:,i]==1)[0])
        xtmp = np.zeros(shape=[num, p])

        if i%5==0:
            for j in range(p):
                prob = np.random.random()/10 + 0.45
                priorCount[j] = np.random.binomial(2, prob)


        for j in range(p):
            prob = np.random.beta(1+r1*priorCount[j], 1+r1*(2-priorCount[j]))
            prior_c = np.random.binomial(2, prob)
            for k in range(num):
                prob = np.random.beta(1+r2*prior_c, 1+r2*(2-prior_c))
                xtmp[k,j] = np.random.binomial(2, prob)
        if i == 0:
            X = xtmp
        else:
            X = np.append(X, xtmp, 0)

    return X.astype(float)

def discreteMapping(X):
    X[X>0] = 2
    X[X<0] = 1
    return X

def generateData(seed, n, p, snp=True):
    np.random.seed(seed)
    Xtrain = dataGeneration_SNP(n=n, p=p)
    Xval = dataGeneration_SNP(n=n, p=p)
    Xtest = dataGeneration(n=n, p=p)

    # Xtrain = discreteMapping(Xtrain)
    # Xval = discreteMapping(Xval)
    # Xtest = discreteMapping(Xtest)

    b = np.zeros(p)
    for i in range(group):
        b[i*p/group] = np.random.random()
    Ytrain = oneHotRepresentation(binary_transform(np.dot(Xtrain, b)))
    Yval = oneHotRepresentation(binary_transform(np.dot(Xval, b)))
    Ytest = oneHotRepresentation(binary_transform(np.dot(Xtest, b)))

    dataPath = '../data/'+str(seed) + '_' + str(n) + '_' + str(p) + '_' + str(group) + '_'

    np.save(dataPath+'Xtrain', Xtrain)
    np.save(dataPath+'Xval', Xval)
    np.save(dataPath+'Xtest', Xtest)
    np.save(dataPath+'Ytrain', Ytrain)
    np.save(dataPath+'Yval', Yval)
    np.save(dataPath+'Ytest', Ytest)

if __name__ == '__main__':
    for seed in range(10):
        n = 5000
        p = 1000
        group = 100
        generateData(seed=seed, n=n, p=p, snp=True)