import tensorflow as tf
from utils.attention_block import attention_stage_1, attention_stage_2, attention_stage_3
from utils.NAL_block import NAL_stage_1, NAL_stage_2, NAL_stage_3
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation, GlobalAveragePooling2D
from utils.residual_unit_CIFAR import residual_unit

'''
This file builds up the attention56 model and attention92 model for CIFAR-100 dataset using mixed attention module.

attention56 model has structure:
input >> Conv(3,3)
      >> BN >> ReLU >> MaxPool(3,3)
      >> Residual([4, 4, 16],s=1) >> Attention(4, 4, 16)
      >> Residual([8, 8, 32],s=2) >> Attention(8, 8, 32)
      >> Residual([16, 16, 64],s=2) >> Attention(16, 16, 64)
      >> Residual([16, 16, 64],s=1) >> Residual([16, 16, 64],s=1) >> Residual([16, 16, 64],s=1)
      >> BN >> ReLU >> GlobalAveragePooling >> Flatten >> Dense(100)
      >> output
      
attention92 model has structure:
input >> Conv(3,3)
      >> BN >> ReLU >> MaxPool(3,3)
      >> Residual([4, 4, 16],s=1) >> Attention(4, 4, 16)
      >> Residual([8, 8, 32],s=2) >> Attention(8, 8, 32) >> Attention(8, 8, 32)
      >> Residual([16, 16, 64],s=2) >> Attention(16, 16, 64) >> Attention(16, 16, 64) >> Attention(16, 16, 64)
      >> Residual([16, 16, 64],s=1) >> Residual([16, 16, 64],s=1) >> Residual([16, 16, 64],s=1)
      >> BN >> ReLU >> GlobalAveragePooling >> Flatten >> Dense(100)
      >> output
'''

def AttentionResNet56(X):
    
    '''
    Attention56 model using mixed attention module.
    
    inputs:
    parameter X: Input data with shape (batch_size, 32, 32, 3).
    
    outputs:
    X: Output data with shape (batch_size, 100), each 100 values for one batch represents the probability for the image to be recognized as the digits.
    '''
    
  X = Conv2D(filters = 16, kernel_size = 3, padding = 'same',
             bias_initializer = 'zeros',
             kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = tf.sqrt(2/(16*3*3))))(X)
  X = BatchNormalization()(X)
  X = Activation('relu')(X)
  X = MaxPooling2D(pool_size = (3, 3), strides=1, padding = 'same')(X)

  X = residual_unit(X, [4, 4, 16], s = 1)
  X = attention_stage_1(X, [4, 4, 16])  # 32*32*16

  X = residual_unit(X, [8, 8, 32], s = 2) # 16*16*32
  X = attention_stage_2(X, [8, 8, 32])

  X = residual_unit(X, [16, 16, 64], s = 2) # 8*8*64
  X = attention_stage_3(X, [16, 16, 64])

  X = residual_unit(X, [16, 16, 64], s = 1)
  X = residual_unit(X, [16, 16, 64], s = 1)
  X = residual_unit(X, [16, 16, 64], s = 1)

  X = BatchNormalization()(X)
  X = Activation('relu')(X)
  X = GlobalAveragePooling2D()(X)
  X = Flatten()(X)

  output = Dense(100, activation = 'softmax', kernel_initializer=tf.keras.initializers.RandomNormal(mean = 0.0, stddev = tf.sqrt(2/100)))(X)

  return output


def AttentionResNet92(X):
    
    '''
    Attention92 model using mixed attention module.
    
    inputs:
    parameter X: Input data with shape (batch_size, 32, 32, 3).
    
    outputs:
    X: Output data with shape (batch_size, 100), each 100 values for one batch represents the probability for the image to be recognized as the digits.
    '''
    
    X = Conv2D(filters = 16, kernel_size = 3, padding = 'same',
               bias_initializer = 'zeros',
               kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = tf.sqrt(2/(16*3*3))))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size = (3, 3), strides=1, padding = 'same')(X)
    
    X = residual_unit(X, [4, 4, 16], s = 1)
    X = attention_stage_1(X, [4, 4, 16])  # 32*32*16
    
    X = residual_unit(X, [8, 8, 32], s = 2) # 16*16*32
    X = attention_stage_2(X, [8, 8, 32])
    X = attention_stage_2(X, [8, 8, 32])
    
    X = residual_unit(X, [16, 16, 64], s = 2) # 8*8*64
    X = attention_stage_3(X, [16, 16, 64])
    X = attention_stage_3(X, [16, 16, 64])
    X = attention_stage_3(X, [16, 16, 64])
    
    X = residual_unit(X, [16, 16, 64], s = 1)
    X = residual_unit(X, [16, 16, 64], s = 1)
    X = residual_unit(X, [16, 16, 64], s = 1)
    
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = GlobalAveragePooling2D()(X)
    X = Flatten()(X)
    
    output = Dense(100, activation = 'softmax', kernel_initializer=tf.keras.initializers.RandomNormal(mean = 0.0, stddev = tf.sqrt(2/10)))(X)

    return output



