#### Libraries
# Standard library
import cPickle
import time
import gzip

# Third-party libraries
import numpy as np

def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

def change_image_uniform():
    traning_data,validation_data,test_data=load_data()
    np.random.seed(int(time.time()))
    a=np.random.rand(10)
    changed_traning_data=[]
    images=traning_data[0]
    labels=traning_data[1]
    for i in range(0,50000):
        #training_data: <image array, label>
        image_array=images[i]
        label=labels[i]
        image_array = [k * a[label] for k in image_array]
        changed_traning_data.append((image_array,label))

    changed_validation_data=[]
    images=validation_data[0]
    labels=validation_data[1]
    for i in range(0,10000):
        image_array=images[i]
        label=labels[i]
        #print label   
        image_array = [k * a[label] for k in image_array]
        changed_validation_data.append((image_array,label))

    changed_test_data=[]
    images=test_data[0]
    labels=test_data[1]
    for i in range(0,10000):
        image_array=images[i]
        label=labels[i]
        image_array = [k * a[label] for k in image_array]
        changed_test_data.append((image_array,label))
    output = open('../data/mnist_uniform.pkl', 'w')
    # Pickle dictionary using protocol 0.
    cPickle.dump((changed_traning_data,changed_validation_data,changed_test_data),output)
    #g = gzip.GzipFile(filename="", mode="wb", fileobj=open('../data/mnist_uniform.pkl.gz', 'wb'))
    #g.write(open('../data/mnist_uniform.pkl').read())
    #g.close()   

change_image_uniform()