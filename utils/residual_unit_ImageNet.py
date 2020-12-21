import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Add

'''
Two kinds of residual units for ImageNet model. 
res_conv: one convolutional layer in shortcut
res_identity: identity shortcut
'''

def res_conv(input, filters, s): # s means stride
    F1, F2, F3 = filters


    shortcut = BatchNormalization()(input)
    shortcut = Activation('relu')(shortcut)
    shortcut = Conv2D(filters = F3, kernel_size = (1, 1), padding = 'same', strides = (s, s),
                                    bias_initializer = 'zeros',
                                    kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.01))(shortcut)

  #res
    X = BatchNormalization()(input)
    X = Activation('relu')(X)
    X = Conv2D(filters = F1, kernel_size = (1, 1), padding = 'valid', strides = (1, 1),
             bias_initializer = 'zeros',
             kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.01))(X)

    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(filters = F2, kernel_size = (3, 3), padding = 'same', strides = (s, s),
             bias_initializer = 'zeros',
             kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.01))(X)


    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(filters = F3, kernel_size = (1, 1), padding = 'same', strides = (1, 1),
             bias_initializer = 'zeros',
             kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.01))(X)

  #ADD
    X = Add()([X, shortcut])

    return X


def res_identity(input, filters, s): # s means stride
    F1, F2, F3 = filters


    shortcut = input

  #res
    X = BatchNormalization()(input)
    X = Activation('relu')(X)
    X = Conv2D(filters = F1, kernel_size = (1, 1), padding = 'valid', strides = (1, 1),
             bias_initializer = 'zeros',
             kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.01))(X)

    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(filters = F2, kernel_size = (3, 3), padding = 'same', strides = (s, s),
             bias_initializer = 'zeros',
             kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.01))(X)


    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(filters = F3, kernel_size = (1, 1), padding = 'same', strides = (1, 1),
             bias_initializer = 'zeros',
             kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.01))(X)

  #ADD
    X = Add()([X, shortcut])

    return X




