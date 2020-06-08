'''This file loads the data from already grouped patches.
   First ~80% data is taken for train+validation and rest ~20% data
   is used for final testing. Since, the testing is only done on group level,
   the testing is done separately (this is not be concerened in this file.)
'''

import h5py 
import numpy as np
from sklearn.cross_validation import train_test_split

def group_extraction():
    path = '/data4/gopal.sharma/image_quality/test_data/'
    X_train = np.zeros((0, 1, 48, 48))
    y_train = np.zeros((0,))
    X_test  = np.zeros((0, 1, 48, 48))
    y_test  = np.zeros((0,))
    for i in range(11):
        with h5py.File(path + 'grouping%d.hdf5'%(i), 'r') as hf:
            print('List of arrays in this file: \n', hf.keys())
            data = hf.get('y')
            y = np.array(data)
            print('Shape of the array dataset_1: \n', X_train.shape)
            data = hf.get('X')
            X = np.array(data)            
        X = np.expand_dims(X, axis = 1)
        print (X.shape)
        X_train = np.concatenate((X_train, X), axis = 0)
        y_train = np.concatenate((y_train, y), axis = 0)

    for i in range(12, 15):
        with h5py.File(path + 'grouping%d.hdf5'%(i), 'r') as hf:
            print('List of arrays in this file: \n', hf.keys())
            data = hf.get('y')
            y = np.array(data)
            print('Shape of the array dataset_1: \n', X_train.shape)
            data = hf.get('X')
            X = np.array(data)
        X = np.expand_dims(X, axis = 1)
        X_test = np.concatenate((X_test, X), axis = 0)
        y_test = np.concatenate((y_test, y), axis = 0)

    return X_train, X_test, y_train, y_test
