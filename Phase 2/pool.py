from keras import backend as K

def min_max_pool2d(x):
    max_x = K.pool2d(x, pool_size=(7, 7), strides=(2, 2))
    min_x = -K.pool2d(-x, pool_size=(7, 7), strides=(2, 2))
    return K.concatenate([max_x, min_x], axis=1)  # concatenate on channel

def min_max_pool2d_output_shape(input_shape):
    shape = list(input_shape)
    shape[1] *= 2
    shape[2] = 1
    shape[3] = 1
    return tuple(shape)