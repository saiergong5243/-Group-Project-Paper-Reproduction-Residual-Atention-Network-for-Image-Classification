import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Add

'''
This file constructs the residual unit using pre-activation structure.
'''

def residual_unit(input, filters, s): 
    
    '''
    inputs:
    parameter input: Input data with shape (batch, height, width, channels).
    parameter filters: a vector of length 3, [num_filter_1, num_filter_2,num_filter_3].
                       represent the number of output filters at each convolution layer in residual unit.
    parameter s: value of stride for the second convolution layer.
    parameter mean and stddev: the mean and standard deviation of the normal distribution which we use to initialize weights
    
    outputs:
    X: Output data with shape (batch, new_height, new_width, num_filter_3).
    
    '''
    
    F1, F2, F3 = filters

    if s==1:
        shortcut = input
    else:
        shortcut = BatchNormalization()(input)
        shortcut = Activation('relu')(shortcut)
        shortcut = Conv2D(filters = F3, kernel_size = (1, 1), padding = 'same', strides = (s, s),
                                        bias_initializer = 'zeros',
                                        kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = tf.sqrt(2/(F3*1*1))))(shortcut)

  #res
    X = BatchNormalization()(input)
    X = Activation('relu')(X)
    X = Conv2D(filters = F1, kernel_size = (1, 1), padding = 'valid', strides = (1, 1), 
             bias_initializer = 'zeros',
             kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = tf.sqrt(2/(F1*1*1))))(X)

    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(filters = F2, kernel_size = (3, 3), padding = 'same', strides = (s, s), 
             bias_initializer = 'zeros',
             kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = tf.sqrt(2/(F2*3*3))))(X)


    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(filters = F3, kernel_size = (1, 1), padding = 'same', strides = (1, 1), 
             bias_initializer = 'zeros',
             kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = tf.sqrt(2/(F3*1*1))))(X)

  #ADD
    X = Add()([X, shortcut])

    return X







