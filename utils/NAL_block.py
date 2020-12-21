import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Multiply
from utils.residual_unit_CIFAR import residual_unit

'''
This file builds up the three attention modules used in the naive attention learning model.
'''

def NAL_stage_1(input, filters):
    
    '''
    The first attention module in the naive attention learning model using mixed attention with two skip connections in mask branch.
    
    inputs:
    parameter input: Input data.
    parameter filters: a vector of length 3, representing the number of output filters used in the funtion of residual unit.
    
    outputs:
    X: Output data.
    '''
    
    F1, F2, F3 = filters
    
    #p = 1
    X = residual_unit(input, filters, s = 1)
    
    #t = 2 trunk branch
    trunk = residual_unit(X, filters, s = 1)
    trunk = residual_unit(trunk, filters, s = 1)
    
    #soft mask branch   ### r = 1
    ###maxpooling
    X = MaxPooling2D((3,3), strides=(2,2), padding = 'same')(X)
    X = residual_unit(X, filters, s = 1)
    
    X_down_1 = residual_unit(X, filters, s = 1)

    X = MaxPooling2D((3,3), strides=(2,2), padding = 'same')(X)
    X = residual_unit(X, filters, s = 1)
    
    X_down_2 = residual_unit(X, filters, s = 1)
    
    X = MaxPooling2D((3,3), strides=(2,2), padding = 'same')(X)
    X = residual_unit(X, filters, s = 1)
    X = residual_unit(X, filters, s = 1)
    X = UpSampling2D()(X)
    
    X = Add()([X, X_down_2])
    
    X = residual_unit(X, filters, s = 1)
    X = UpSampling2D()(X)
    
    X = Add()([X, X_down_1])
    
    X = residual_unit(X, filters, s = 1)
    X = UpSampling2D()(X)
    
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1, 1), padding = 'valid',
               bias_initializer = 'zeros',
               kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = tf.sqrt(2/(F3*1*1))))(X)
    
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1, 1), padding = 'valid',
               bias_initializer = 'zeros',
               kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = tf.sqrt(2/(F3*1*1))))(X)
    
    X = Activation('sigmoid')(X)
    
    #
    X = Multiply()([X, trunk])
    
    # p = 1
    X = residual_unit(X, filters, s = 1)
    
    return X

def NAL_stage_2(input, filters):
    
    '''
    The second attention module in the naive attention learning model using mixed attention with one skip connection in mask branch.
    
    inputs:
    parameter input: Input data.
    parameter filters: a vector of length 3, representing the number of output filters used in the funtion of residual unit.
    
    outputs:
    X: Output data.
    '''
    
    F1, F2, F3 = filters
    
    #p = 1
    X = residual_unit(input, filters, s = 1)
    
    #t = 2 trunk branch
    trunk = residual_unit(X, filters, s = 1)
    trunk = residual_unit(trunk, filters, s = 1)
    
    #soft mask branch   ### r = 1
    ###maxpooling
    X = MaxPooling2D((3,3), strides=(2,2), padding = 'same')(X)
    X = residual_unit(X, filters, s = 1)
    
    X_down = residual_unit(X, filters, s = 1)
    
    X = MaxPooling2D((3,3), strides=(2,2), padding = 'same')(X)
    X = residual_unit(X, filters, s = 1)
    X = residual_unit(X, filters, s = 1)
    X = UpSampling2D()(X)
    
    X = Add()([X, X_down])
    
    X = residual_unit(X, filters, s = 1)
    X = UpSampling2D()(X)
    
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1, 1), padding = 'valid',
               bias_initializer = 'zeros',
               kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = tf.sqrt(2/(F3*1*1))))(X)
    
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1, 1), padding = 'valid',
               bias_initializer = 'zeros',
               kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = tf.sqrt(2/(F3*1*1))))(X)
    
    X = Activation('sigmoid')(X)
    
    #
    X = Multiply()([X, trunk])
    
    # p = 1
    X = residual_unit(X, filters, s = 1)
    
    return X

def NAL_stage_3(input, filters):
    
    '''
    The second attention module in the naive attention learning model using mixed attention without skip connection in mask branch.
    
    inputs:
    parameter input: Input data.
    parameter filters: a vector of length 3, representing the number of output filters used in the funtion of residual unit.
    
    outputs:
    X: Output data.
    '''
    
    F1, F2, F3 = filters
    
    #p = 1
    X = residual_unit(input, filters, s = 1)
    
    #t = 2 trunk branch
    trunk = residual_unit(X, filters, s = 1)
    trunk = residual_unit(trunk, filters, s = 1)
    
    #soft mask branch   ### r = 1
    ###maxpooling
    X = MaxPooling2D((3,3), strides=(2,2), padding = 'same')(X)
    X = residual_unit(X, filters, s = 1)
    X = residual_unit(X, filters, s = 1)
    X = UpSampling2D()(X)
    
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1, 1), padding = 'valid',
               bias_initializer = 'zeros',
               kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = tf.sqrt(2/(F3*1*1))))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1, 1), padding = 'valid',
               bias_initializer = 'zeros',
               kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = tf.sqrt(2/(F3*1*1))))(X)
    
    X = Activation('sigmoid')(X)
    
    #
    X = Multiply()([X, trunk])
    
    # p = 1
    X = residual_unit(X, filters, s = 1)
    
    return X


