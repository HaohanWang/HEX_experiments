# change dataloader to load npy with corr=0.0~1.0
import numpy as np
import cPickle as pickle
import cv2
import gzip
import scipy.io as scio

f=open('debug.txt','w')

def oneHotRepresentation(y, num=10):
    r = []
    for i in range(y.shape[0]):
        l = np.zeros(num)
        #print y[i]
        if y[i]==10:
            y[i]=0
        l[int(y[i])] = 1
        r.append(l)
    return np.array(r)

def check(img,label):
    print label
    print img
    cv2.imshow('1',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def loadData(seed, n, p, group):
    dataPath = '../data/' + str(seed) + '_' + str(n) + '_' + str(p) + '_' + str(group) + '_'

    Xtrain = np.load(dataPath + 'Xtrain.npy').astype(np.float32)
    Xval = np.load(dataPath + 'Xval.npy').astype(np.float32)
    Xtest = np.load(dataPath + 'Xtest.npy').astype(np.float32)
    Ytrain = np.load(dataPath + 'Ytrain.npy').astype(np.float32)
    Yval = np.load(dataPath + 'Yval.npy').astype(np.float32)
    Ytest = np.load(dataPath + 'Ytest.npy').astype(np.float32)

    return Xtrain, Ytrain, Xval, Yval, Xtest, Ytest



def loadDataOriginalMNIST():
    #f = gzip.open('../../../data/mnist.pkl.gz', 'rb')
    #f = gzip.open('../../../data/mnist.pkl.gz', 'rb')
    f=gzip.open('../data/MNIST/mnist.pkl.gz','rb')
    training_data, validation_data, test_data = pickle.load(f)
    xtr=training_data[0] # (50000, 784)
    xtrain=np.zeros([xtr.shape[0],28,28,3])
    for i in range(xtr.shape[0]):
        img = xtr[i].reshape(28,28)
        xtrain[i] = np.expand_dims(img, axis=2)
    ytrain=training_data[1] # (50000, 1)
    xva=validation_data[0] # (10000, 784)
    xval=np.zeros([xva.shape[0],28,28,3])
    for i in range(xva.shape[0]):
        img = xva[i].reshape(28,28)
        xval[i] = np.expand_dims(img, axis=2)
    yval=validation_data[1] # (10000, 1)
    xte=test_data[0] # (10000, 784)
    xtest=np.zeros([xte.shape[0],28,28,3])
    for i in range(xte.shape[0]):
        img = xte[i].reshape(28,28)
        xtest[i] = np.expand_dims(img, axis=2)    
     
    ytest=test_data[1] # (10000, 1)
    #check(xval[100],yval[100])
    #print xval[100]
    #print xtrain.shape, ytrain.shape, xval.shape, yval.shape, xtest.shape, ytest.shape
    ytrain=oneHotRepresentation(ytrain,10)
    yval=oneHotRepresentation(yval,10)
    ytest=oneHotRepresentation(ytest,10)  
    print xtrain.shape, ytrain.shape, xval.shape, yval.shape, xtest.shape, ytest.shape
    print "End loading MNIST-O"
    
    return xtrain, ytrain, xval, yval, xtest, ytest

def loadDataMNIST_M():
    
    f=open('../data/MNIST/mnistm_data.pkl', 'r')
    #f=open('/Users/hzxue/Desktop/CMU/project/artificial-pattern/data/mnistm_data.pkl', 'r')
    data=pickle.load(f)
    xtrain=data['train']/255.0
    ytrain=data['ytrain']
    xval=data['valid']/255.0
    yval=data['yval']
    xtest=data['test']/255.0
    ytest=data['ytest'] 
    ytrain=oneHotRepresentation(ytrain,10)
    yval=oneHotRepresentation(yval,10)
    ytest=oneHotRepresentation(ytest,10)   
    print xtrain.shape, ytrain.shape, xval.shape, yval.shape, xtest.shape, ytest.shape
    print "End loading MNIST-M"
    
    return xtrain, ytrain, xval, yval, xtest, ytest
    
# def loadDataMNIST_R():
#     #path='../../../data/MNIST-r/npy/'
#     path='/media/student/Data/zexue/mnist-r/npy/'
#     xtr=np.load(path+'xtrain.npy')
#     xtrain=np.zeros([xtr.shape[0],28,28,3])
#     for i in range(xtr.shape[0]):
#         img = xtr[i].reshape(28,28)
#         xtrain[i] = np.expand_dims(img, axis=2)
#
#     ytrain=np.load(path+'ytrain.npy')
#
#     xva=np.load(path+'xval.npy')
#     xval=np.zeros([xva.shape[0],28,28,3])
#     for i in range(xva.shape[0]):
#         img = xva[i].reshape(28,28)
#         xval[i] = np.expand_dims(img, axis=2)
#     yval=np.load(path+'yval.npy')
#
#     xte=np.load(path+'xtest.npy')
#     xtest=np.zeros([xte.shape[0],28,28,3])
#     for i in range(xte.shape[0]):
#         img = xte[i].reshape(28,28)
#         xtest[i] = np.expand_dims(img, axis=2)
#     ytest=np.load(path+'ytest.npy')
#     ytrain=oneHotRepresentation(ytrain,10)
#     yval=oneHotRepresentation(yval,10)
#     ytest=oneHotRepresentation(ytest,10)
#     #check(xval[100],yval[100])
#     print xtrain.shape, ytrain.shape, xval.shape, yval.shape, xtest.shape, ytest.shape
#     print "End loading MNIST-R"
#     return xtrain, ytrain, xval, yval, xtest, ytest

    
def loadDataSVHN():
    #path='../../../data/SVHN/'
    path='../data/SVHN/'
    train = scio.loadmat(path+'train_32x32.mat')
    test = scio.loadmat(path+'test_32x32.mat')

    l=train['X'].shape[3]
    tr_l=int(l*0.75)

    xtr=np.copy(train['X'][:,:,:, 0:tr_l]) # (32, 32, 3, 73257)
    ytr=np.copy(train['y'][0:tr_l, :]) # (73257,1)
    ytr.reshape(ytr.shape[0],1)
   

    xva=np.copy(train['X'][:,:,:,tr_l:l])
    yva=np.copy(train['y'][tr_l:l, :])
    yva.reshape(yva.shape[0],1)

    xte=test['X'] # (32, 32, 3, 26032)
    yte=test['y'] # (26032, 1)
    yte.reshape(yte.shape[0],1)
    
    xtrain=np.zeros([xtr.shape[3],28,28,3])
    ytrain=oneHotRepresentation(ytr)
    for i in range(xtr.shape[3]):
        #img=cv2.cvtColor(xtr[:,:,:,i],cv2.COLOR_RGB2GRAY)
        img=cv2.resize(xtr[:,:,:,i],(28,28),interpolation=cv2.INTER_CUBIC)
        #xtrain[i]=img.reshape(28*28)
        xtrain[i]=img/255.0
    
    xval=np.zeros([xva.shape[3],28,28,3])
    yval=oneHotRepresentation(yva)
    for i in range(xva.shape[3]):
        #img=cv2.cvtColor(xva[:,:,:,i],cv2.COLOR_RGB2GRAY)
        img=cv2.resize(xva[:,:,:,i],(28,28),interpolation=cv2.INTER_CUBIC)
        #xval[i]=img.reshape(28*28)
        xval[i]=img/255.0

    xtest=np.zeros([xte.shape[3],28,28,3])
    ytest=oneHotRepresentation(yte)
    for i in range(xte.shape[3]):
        #img=cv2.cvtColor(xte[:,:,:,i],cv2.COLOR_RGB2GRAY)
        img=cv2.resize(xte[:,:,:,i],(28,28),interpolation=cv2.INTER_CUBIC)
        #xtest[i]=img.reshape(28*28)
        xtest[i]=img/255.0
    #check(xval[100],yval[100])
    #print xval[100]
    print xtrain.shape, ytrain.shape, xval.shape, yval.shape, xtest.shape, ytest.shape
    print "end loading SVHN"
    return xtrain, ytrain,xval,yval,xtest,ytest

def loadDataUSPS():
    #path='../../../data/USPS/'
    path='../data/USPS/'
    data = scio.loadmat(path+'MNIST_vs_USPS.mat')
   
    x=data['X_tar'] # (256, 2000)
    y=data['Y_tar'] # (2000,1)
    l=x.shape[1]
    tr_l=int(l*0.8)
    val_l=int(l*0.1)+tr_l
    xtr=np.copy(x[:, 0: tr_l])
    ytr=np.copy(y[0: tr_l, :])
    ytr-=1
    ytr.reshape(ytr.shape[0],1)
    
    xva=np.copy(x[:, tr_l: val_l])
    yva=np.copy(y[tr_l: val_l, :])
    yva-=1
    yva.reshape(yva.shape[0],1)

    xte=np.copy(x[:, val_l: l])
    yte=np.copy(y[val_l:l, :])
    yte-=1
    yte.reshape(yte.shape[0],1)
    

    xtrain=np.zeros([xtr.shape[1],28,28,3])
    ytrain=oneHotRepresentation(ytr)
    for i in range(xtr.shape[1]):
        img=xtr[:,i].reshape(16,16)
        img=cv2.resize(img,(28,28),interpolation=cv2.INTER_CUBIC)
        img = np.expand_dims(img, axis=2)
        #xtrain[i]=img.reshape(28*28)
        xtrain[i]=img
    
    xval=np.zeros([xva.shape[1],28,28,3])
    yval=oneHotRepresentation(yva)
    for i in range(xva.shape[1]):
        #img=cv2.cvtColor(xva[:,:,:,i],cv2.COLOR_RGB2GRAY)
        img=xva[:,i].reshape(16,16)
        img=cv2.resize(img,(28,28),interpolation=cv2.INTER_CUBIC) 
        img = np.expand_dims(img, axis=2)
        #xval[i]=img.reshape(28*28)
        xval[i]=img

    xtest=np.zeros([xte.shape[1],28,28,3])
    ytest=oneHotRepresentation(yte)
    for i in range(xte.shape[1]):
        #img=cv2.cvtColor(xte[:,:,:,i],cv2.COLOR_RGB2GRAY)
        img=xte[:,i].reshape(16,16)
        img=cv2.resize(img,(28,28),interpolation=cv2.INTER_CUBIC) 
        img = np.expand_dims(img, axis=2)

        #xtest[i]=img.reshape(28*28)
        xtest[i]=img
    #check(xval[100],yval[100])
    #print xval
    print xtrain.shape, ytrain.shape, xval.shape, yval.shape, xtest.shape, ytest.shape
    print "End loading USPS"
    return xtrain, ytrain,xval,yval,xtest,ytest


def loadDigitClassification():
    '''
    :param test: 0 for MNIST-M, 1 for SVHN, 2 for USPS, 3 vanilla for MNIST
    :return:
    '''
    m_xtrain, m_ytrain,m_xval,m_yval,m_xtest,m_ytest=loadDataOriginalMNIST()

    s_xtrain, s_ytrain,s_xval,s_yval,s_xtest,s_ytest=loadDataSVHN()

    u_xtrain, u_ytrain,u_xval,u_yval,u_xtest,u_ytest=loadDataUSPS()

    mm_xtrain, mm_ytrain,mm_xval,mm_yval,mm_xtest,mm_ytest=loadDataMNIST_M()

    xtest = u_xtest
    ytest = u_ytest

    ztest = np.zeros([xtest.shape[0], 4])
    ztest[:, 2] = 1

    xtrain = np.append(m_xtrain, s_xtrain, 0)
    ytrain = np.append(m_ytrain, s_ytrain, 0)
    xtrain = np.append(xtrain, u_xtrain, 0)
    ytrain = np.append(ytrain, u_ytrain, 0)
    xtrain = np.append(xtrain, mm_xtrain, 0)
    ytrain = np.append(ytrain, mm_ytrain, 0)

    ztrain = np.zeros([xtrain.shape[0],4])
    ztrain[0:m_xtrain.shape[0],0] = 1
    ztrain[m_xtrain.shape[0]:m_xtrain.shape[0]+s_xtrain.shape[0],1] = 1
    ztrain[m_xtrain.shape[0]+s_xtrain.shape[0]:m_xtrain.shape[0]+s_xtrain.shape[0]+u_xtrain.shape[0],2] = 1
    ztrain[m_xtrain.shape[0]+s_xtrain.shape[0]+u_xtrain.shape[0]:,3] = 1

    xval = u_xval
    yval = u_yval

    zval = np.zeros([xval.shape[0], 4])
    zval[:, 2] = 1

    indices = np.random.permutation(xtrain.shape[0])
    xtrain=xtrain[indices,:]
    ytrain=ytrain[indices,:]
    ztrain=ztrain[indices,:]

    return xtrain, ytrain,xval,yval,xtest,ytest,ztrain,zval,ztest

if __name__ == '__main__':
   pass