import tensorflow as tf
from utils.attention_block import attention_stage_1, attention_stage_2, attention_stage_3
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation, GlobalAveragePooling2D
from utils.residual_unit_ImageNet import res_conv, res_identity



def AttentionResNet56(X):
    X = Conv2D(filters = 64, kernel_size = 3, padding = 'same',
             bias_initializer = 'zeros',
             kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = tf.sqrt(2/(16*3*3))))(X)  # different from the original paper, use stride=1
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size = (3, 3), strides=2, padding = 'same')(X)

    X = res_conv(X, [16, 16, 64], s = 1)
    X = attention_stage_1(X, [16, 16, 64])  

    X = res_conv(X, [32, 32, 128], s = 2) 
    X = attention_stage_2(X, [32, 32, 128])

    X = res_conv(X, [64, 64, 256], s = 2) 
    X = attention_stage_3(X, [64, 64, 256])

    X = res_conv(X, [128, 128, 512], s = 2)
    X = res_identity(X, [128, 128, 512], s = 1)
    X = res_identity(X, [128, 128, 512], s = 1)

    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = GlobalAveragePooling2D()(X)
    X = Flatten()(X)
    X = Dense(1024, activation='relu')(X) #Add one more dense layer compared to the original paper
    output = Dense(200, activation = 'softmax', kernel_initializer=tf.keras.initializers.RandomNormal(mean = 0.0, stddev = tf.sqrt(2/10)))(X)

    return output






def AttentionResNet_92():
    X = Conv2D(filters = 64, kernel_size = 7, padding = 'same',
               bias_initializer = 'zeros',
               kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = tf.sqrt(2/(64*7*7))))(X)
    X = BatchNormalization(axis=-1, name='bn_1')(X)
    X = Activation('relu', name='relu_1')(X)
    X = MaxPooling2D(pool_size = (3, 3), strides=2, padding = 'same')(X)
    
    X = res_conv(X, [64, 64, 256], s = 1)
    X = attention_stage_1(X, [64, 64, 256])
    
    X = res_conv(X, [128, 128, 512], s = 2)
    X = attention_stage_2(X, [128, 128, 512])
    X = attention_stage_2(X, [128, 128, 512])
    
    X = res_conv(X, [256,256,1024], s = 2)
    X = attention_stage_3(X, [256,256,1024])
    X = attention_stage_3(X, [256,256,1024])
    X = attention_stage_3(X, [256,256,1024])
    
    X = res_conv(X, [512,512,2048], s = 2)
    X = res_identity(X, [512,512,2048], s = 1)
    X = res_identity(X, [512,512,2048], s = 1)
    
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = GlobalAveragePooling2D()(X)
    #X = Flatten()(X)
    
    output = Dense(200, activation = 'softmax', kernel_initializer=tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.01))(X)
    
    return output

