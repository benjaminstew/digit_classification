'''loading and pre-processing of MNIST dataset'''
import pickle 
import gzip 
import numpy as np 

def load_data():
    '''data loaded as tuple: 
    first element = 50k-d array of image vectors, 
    second element = 50k-d array of 0->9 digit labels'''
    f = gzip.open('../data/mnist.pkl.gz', 'rb') #decompress dataset file 
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1') #unpickle datset into original objects   
    f.close()
    return (training_data, validation_data, test_data) 

def process_data():
    '''pair up images with vectorised versions of digit labels as (x,y)'''
    training_data, validation_data, test_data = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in training_data[0]] #reshape images as column vectors (784,1) 
    training_labels = [vectorised_digit(y) for y in training_data[1]] #vectorise training labels 
    training_data = list(zip(training_inputs, training_labels))
    validation_inputs = [np.reshape(x, (784, 1)) for x in validation_data[0]]
    validation_data = list(zip(validation_inputs, validation_data[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
    test_data = list(zip(test_inputs, test_data[1]))
    #leave validation and test labels as int digits 
    return (training_data, validation_data, test_data)

def vectorised_digit(digit):
    '''turn digit into its (10,1) unit vector representation'''
    e = np.zeros((10,1))
    e[digit] = 1.0
    return e 



