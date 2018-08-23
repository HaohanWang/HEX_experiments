# change dataloader to load npy with corr=0.0~1.0
import numpy as np
import cPickle
import cv2
import gzip

def oneHotRepresentation(y, num=10):
    r = []
    for i in range(y.shape[0]):
        l = np.zeros(num)
        l[y[i]] = 1
        r.append(l)
    return np.array(r)


def loadData(seed, n, p, group):
    dataPath = '../data/' + str(seed) + '_' + str(n) + '_' + str(p) + '_' + str(group) + '_'

    Xtrain = np.load(dataPath + 'Xtrain.npy').astype(np.float32)
    Xval = np.load(dataPath + 'Xval.npy').astype(np.float32)
    Xtest = np.load(dataPath + 'Xtest.npy').astype(np.float32)
    Ytrain = np.load(dataPath + 'Ytrain.npy').astype(np.float32)
    Yval = np.load(dataPath + 'Yval.npy').astype(np.float32)
    Ytest = np.load(dataPath + 'Ytest.npy').astype(np.float32)

    return Xtrain, Ytrain, Xval, Yval, Xtest, Ytest


def organizeSentimentData(filePath='/Users/hzxue/Desktop/CMU/project/artificial-pattern/data/background_8/'):
    small = True

    import os
    import cv2
    names = ['aia', 'bonnie', 'jules', 'malcolm', 'mery', 'ray']
    sentiment_dic = {'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3, 'neutral': 4, 'sadness': 5, 'surprise': 6}

    #filePath = '/media/haohanwang/Info/SentimentImages/background_8/'
    #filePath = '/Users/hzxue/Desktop/CMU/project/artificial-pattern/data/background_8/'

    train = []
    trL = []
    val = []
    valL = []
    test = []
    teL = []

    # for training set:
    count = 0
    for n in names:
        for k in sentiment_dic:
            filepath = filePath + n + '/' + n + '_' + k + '/'
            for r, d, f in os.walk(filepath):
                for fn in f:
                    try:
                        l = np.zeros(len(sentiment_dic))
                        count += 1
                        if count % 1000 == 0:
                            print 'finish', count, 'images'
                        im = cv2.imread(filepath + fn, 0)
                        ind = int(fn.split('_')[-1].split('.')[0])
                        if ind % 10 < 5:
                            train.append(im.reshape(28*28))
                            l[sentiment_dic[k]] = 1
                            trL.append(l)
                        elif ind % 10 < 8:
                            val.append(im.reshape(28*28))
                            l[sentiment_dic[k]] = 1
                            valL.append(l)
                        else:
                            test.append(im.reshape(28*28))
                            l[sentiment_dic[k]] = 1
                            teL.append(l)
                    except:
                        pass

    train = np.array(train)
    trL = np.array(trL)
    val = np.array(val)
    valL = np.array(valL)
    test = np.array(test)
    teL = np.array(teL)

    if small:
        np.save(filePath + 'trainData_small', train)
        np.save(filePath + 'trainLabel_small_onehot', trL)
        np.save(filePath + 'valData_small', val)
        np.save(filePath + 'valLabel_small_onehot', valL)
        np.save(filePath + 'testData_small', test)
        np.save(filePath + 'testLabel_small_onehot', teL)
    else:
        np.save(filePath + 'trainData', train)
        np.save(filePath + 'trainLabel', trL)
        np.save(filePath + 'valData', val)
        np.save(filePath + 'valLabel', valL)
        np.save(filePath + 'testData', test)
        np.save(filePath + 'testLabel', teL)

def loadDataSentimentDiffLabel(corr=0.8):
    #dataPath = '/media/haohanwang/Info/SentimentImages/background_8/'
    dataPath = '/media/student/Data/zexue/background_npy/npy_'+str(int(corr*10))+'/'

    #dataPath = '../../../data/background_npy/npy_'+str(int(corr*10))+'/'
    # np.random.seed(0)
    Xtrain = np.load(dataPath + 'trainData_small.npy').astype(np.float32)
    # np.random.shuffle(Xtrain)
    Xval = np.load(dataPath + 'valData_small.npy').astype(np.float32)
    Xtest = np.load(dataPath + 'testData_small.npy').astype(np.float32)

    Ytrain = np.load(dataPath + 'trainLabel_small_onehot.npy').astype(np.float32)
    Yval = np.load(dataPath + 'valLabel_small_onehot.npy').astype(np.float32)
    Ytest = np.load(dataPath + 'testLabel_small_onehot.npy').astype(np.float32)
    names = ['aia', 'bonnie', 'jules', 'malcolm', 'mery', 'ray']
    sentiment_dic = {'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3, 'neutral': 4, 'sadness': 5, 'surprise': 6}
    trL=[]
    valL=[]
    teL=[]
    for i in range(Xtrain.shape[0]):
        l = np.zeros(len(sentiment_dic))
        if np.random.random()>=(1-corr):
             trL.append(Ytrain[i])
        else:
             k=np.random.randint(7)
             l[k] = 1
             trL.append(l)
    for i in range(Xval.shape[0]):
        l = np.zeros(len(sentiment_dic))
        if np.random.random()>=(1-corr):
              valL.append(Yval[i])
        else:
              k=np.random.randint(7)
              l[k] = 1
              valL.append(l)
    
    for i in range(Xtest.shape[0]):
        l = np.zeros(len(sentiment_dic))
        if np.random.random()>=(1-corr):
              teL.append(Ytest[i])
        else:
              k=np.random.randint(7)
              l[k] = 1
              teL.append(l)

    trL=np.array(trL)
    valL=np.array(valL)
    teL=np.array(teL)

    indices = np.random.permutation(Xtrain.shape[0])
    Xtrain=Xtrain[indices,:]
    trL=trL[indices,:]


    indices=np.random.permutation(Xval.shape[0])
    Xval=Xval[indices,:]
    valL=valL[indices,:]

    indices=np.random.permutation(Xtest.shape[0])
    Xtest=Xtest[indices,:]
    teL=teL[indices,:]

    #return Xtrain, Ytrain, Xval, Ytrain, Xtest, Ytest
    return Xtrain, trL, Xval, valL, Xtest, teL

def loadDataSentiment(corr=0.8):
    dataPath = '../data/background_npy/npy_'+str(int(corr*10))+'/'
    #dataPath = '../../../data/background_npy/npy_'+str(int(corr*10))+'/'
    # np.random.seed(0)
    Xtrain = np.load(dataPath + 'trainData_small.npy').astype(np.float32)
    # np.random.shuffle(Xtrain)
    Xval = np.load(dataPath + 'valData_small.npy').astype(np.float32)
    Xtest = np.load(dataPath + 'testData_small.npy').astype(np.float32)
    Ytrain = np.load(dataPath + 'trainLabel_small_onehot.npy').astype(np.float32)
    Yval = np.load(dataPath + 'valLabel_small_onehot.npy').astype(np.float32)
    Ytest = np.load(dataPath + 'testLabel_small_onehot.npy').astype(np.float32)
   
    return Xtrain, Ytrain, Xval, Yval, Xtest, Ytest


def loadDataMNIST():
    f = open('../data/mnist_uniform.pkl', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    return training_data[0], oneHotRepresentation(training_data[1]), validation_data[
        0], oneHotRepresentation(validation_data[1]), test_data[0], oneHotRepresentation(test_data[1])


if __name__ == '__main__':
    
    corr=10
    #loadDataSentimentDiffLabel(corr=0.0)
    #loadDataSentiment(corr=0.8)
    """
    while corr<=10:
        filePath='/Users/hzxue/Desktop/CMU/project/artificial-pattern/data/background_'+str(corr)+'/'
        organizeSentimentData(filePath=filePath)
        corr+=1
    """    
