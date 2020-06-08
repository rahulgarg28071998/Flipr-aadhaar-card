'''This file deals with testing the PCC and SROCC on document level.
   We first load the test data. Then we build and load the learned model.
   Testing is done on document level.
'''
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Lambda
import pool
import h5py
import numpy as np
from scipy import stats

batch_size = 256
path = '/data4/gopal.sharma/image_quality/train_data/groups/'
X_test  = np.zeros((0, 1, 48, 48))
y_test  = np.zeros((0,))
for i in range(0, 25):
    with h5py.File(path + "grouping%d.hdf5" % i, 'r') as hf:
        data = hf.get('y')
        y = np.array(data)
        data = hf.get('X')
        X = np.array(data)
    X = np.expand_dims(X, axis = 1)
    X_test = np.concatenate((X_test, X), axis = 0)
    y_test = np.concatenate((y_test, y), axis = 0)

y_test = np.expand_dims(y_test, axis = 1)

model = Sequential()
model.add(Convolution2D(40, 5, 5, input_shape=(1, 48, 48)))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Convolution2D(80, 5, 5))
model.add(Lambda(pool.min_max_pool2d, output_shape=pool.min_max_pool2d_output_shape))
model.add(Flatten())
model.add(Dense(1024))
model.add(Dense(1024))
model.add(Dense(1))

model.load_weights('weights_cvit_data.hdf5')
print (y_test.shape)
model.compile(loss='mse', optimizer = 'adam')
print (model.summary())

prediction = model.predict(X_test, verbose=1)
print (prediction.shape)
llc = stats.pearsonr(prediction, y_test)
print ('LLC for the test data is: ', llc)

srocc = stats.spearmanr(prediction, y_test)
print ('SROCC for the test data is: ', srocc)


### This is whole new game ####
# For each group, I will sample the image patches for each image
# then I will find the score for each patch, average it over an image.
# Then I will find the score with respect to the document score

# Let us first do this for test images extracted above
# I have the file indices_test.txt

# I have already extracted patches of the test data given by Prof. Vineet
result = []
correct_label = []
file = open(path + 'indices.txt', 'r')
size_f = 0
for f in file:
    lower, upper = f.split(',')
    lower = int(lower)
    upper = int(upper)
    size_f = size_f + upper -lower + 1

for i in range(size_f):
	correct_label += [np.mean(y_test[i*850:(i+1)*850,:])]
	prediction = model.predict(X_test[i*850:(i+1)*850,:, :, :], verbose=1)
	result += [np.mean(prediction)]
result = np.asarray(result)
correct_label = np.asarray(correct_label)
print (result.shape)
print (correct_label.shape)

llc = stats.pearsonr(result, correct_label)
print ('LLC for the test data is: ', llc)

srocc = stats.spearmanr(result, correct_label)
print ('SROCC for the test data is: ', srocc)
