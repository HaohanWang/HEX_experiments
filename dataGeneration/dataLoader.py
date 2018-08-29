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


def organizeSentimentData(loadFilePath='/Users/hzxue/Desktop/CMU/project/artificial-pattern/data/background_8/', saveFilePath=None):
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
            filepath = loadFilePath + n + '/' + n + '_' + k + '/'
            for r, d, f in os.walk(filepath):
                for fn in f:
                    try:
                        l = np.zeros(len(sentiment_dic))
                        count += 1
                        if count % 1000 == 0:
                            print 'finish', count, 'images'
                        im = cv2.imread(filepath + fn, cv2.IMREAD_UNCHANGED)
                        im=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
                        im=cv2.resize(im,(28,28),interpolation=cv2.INTER_CUBIC)

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
        np.save(saveFilePath + 'trainData_small', train)
        np.save(saveFilePath + 'trainLabel_small_onehot', trL)
        np.save(saveFilePath + 'valData_small', val)
        np.save(saveFilePath + 'valLabel_small_onehot', valL)
        np.save(saveFilePath + 'testData_small', test)
        np.save(saveFilePath + 'testLabel_small_onehot', teL)
    else:
        np.save(saveFilePath + 'trainData', train)
        np.save(saveFilePath + 'trainLabel', trL)
        np.save(saveFilePath + 'valData', val)
        np.save(saveFilePath + 'valLabel', valL)
        np.save(saveFilePath + 'testData', test)
        np.save(saveFilePath + 'testLabel', teL)

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

def loadDataOffice_features(testCase=0):
    webcam_surf = np.load('../data/office/webcam_surf.npy')
    webcam_glgcm = np.load('../data/office/webcam_glgcm.npy')
    webcam_label = np.load('../data/office/webcam_label.npy')
    amazon_surf = np.load('../data/office/amazon_surf.npy')
    amazon_glgcm = np.load('../data/office/amazon_glgcm.npy')
    amazon_label = np.load('../data/office/amazon_label.npy')
    dslr_surf = np.load('../data/office/dslr_surf.npy')
    dslr_glgcm = np.load('../data/office/dslr_glgcm.npy')
    dslr_label = np.load('../data/office/dslr_label.npy')

    webcam_glgcm = webcam_glgcm.reshape([webcam_glgcm.shape[0], webcam_glgcm.shape[2]])
    amazon_glgcm = amazon_glgcm.reshape([amazon_glgcm.shape[0], amazon_glgcm.shape[2]])
    dslr_glgcm = dslr_glgcm.reshape([dslr_glgcm.shape[0], dslr_glgcm.shape[2]])

    if testCase == 0:
        print 'DSLR to test'
        trainVal_surf = np.append(webcam_surf, amazon_surf, 0)
        trainVal_glgcm = np.append(webcam_glgcm, amazon_glgcm, 0)
        trainVal_label = np.append(webcam_label, amazon_label)
        test_surf = dslr_surf
        test_glgcm = dslr_glgcm
        test_label = dslr_label
    elif testCase == 1:
        print 'AMAZON to test'
        trainVal_surf = np.append(webcam_surf, dslr_surf, 0)
        trainVal_glgcm = np.append(webcam_glgcm, dslr_glgcm, 0)
        trainVal_label = np.append(webcam_label, dslr_label)
        test_surf = amazon_surf
        test_glgcm = amazon_glgcm
        test_label = amazon_label
    else:
        print 'Webcam to test'
        trainVal_surf = np.append(amazon_surf, dslr_surf, 0)
        trainVal_glgcm = np.append(amazon_glgcm, dslr_glgcm, 0)
        trainVal_label = np.append(amazon_label, dslr_label)
        test_surf = webcam_surf
        test_glgcm = webcam_glgcm
        test_label = webcam_label

    n_trainVal = trainVal_surf.shape[0]
    n_train = int(n_trainVal*0.6)
    indices = np.random.permutation(n_trainVal)
    train_surf = trainVal_surf[indices[:n_train], :]
    val_surf = trainVal_surf[indices[n_train:], :]
    train_glgcm = trainVal_glgcm[indices[:n_train], :]
    val_glgcm = trainVal_glgcm[indices[n_train:], :]
    train_label = trainVal_label[indices[:n_train]]
    val_label= trainVal_label[indices[n_train:]]

    train_label = oneHotRepresentation(train_label, 31)
    val_label = oneHotRepresentation(val_label, 31)
    test_label = oneHotRepresentation(test_label, 31)

    return train_surf, train_glgcm, train_label, val_surf, val_glgcm, val_label, test_surf, test_glgcm, test_label

def loadDataMNIST():
    f = open('../data/mnist_uniform.pkl', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    return training_data[0], oneHotRepresentation(training_data[1]), validation_data[
        0], oneHotRepresentation(validation_data[1]), test_data[0], oneHotRepresentation(test_data[1])


if __name__ == '__main__':
    
    corr=0
    #loadDataSentimentDiffLabel(corr=0.0)
    #loadDataSentiment(corr=0.8)
    while corr<=10:
        print corr
        loadPath='/media/haohanwang/Data/SentimentImages/background_'+str(corr) + '/'
        savePath='../data/background_npy/npy_'+str(corr)+'/'
        organizeSentimentData(loadFilePath=loadPath, saveFilePath=savePath)
        corr+=1
    # loadPath='/media/haohanwang/Info/SentimentImages/background_'+str(corr) + '/'
    # savePath='../data/background_npy/npy_'+str(corr)+'/'
    # organizeSentimentData(loadFilePath=loadPath, saveFilePath=savePath)
