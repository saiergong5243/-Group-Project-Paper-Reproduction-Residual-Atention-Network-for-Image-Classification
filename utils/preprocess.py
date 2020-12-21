# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# CIFAR
# padding and random crop
def CIFAR_preprocess(image):
    # padding
    pad_image = np.zeros([40, 40, 3])
    pad_image[4:36, 4:36] = image
    # Random crop
    crop_image = tf.image.random_crop(pad_image, [32, 32, 3])
    return crop_image

# Noise label robustness
def add_noise(x_train, y_train, NOISE_LEVEL):
    num_class = np.unique(y_train).shape[0]
    select_ratio = NOISE_LEVEL/(num_class-1)*num_class

    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=select_ratio)
    _, selected_idx = next(iter(stratified_split.split(x_train, y_train)))

    new_label = np.random.choice(num_class, size=len(selected_idx), replace=True)
    y_train_noise = y_train.copy()
    y_train_noise[selected_idx] = new_label.reshape([-1,1])
    return x_train, y_train_noise


# For ImageNet
def standard_color_aug(x, eigenvalue, eigenvector):
    alpha = np.random.normal(0, 0.1, 1)
    add_quantity = np.reshape(eigenvector.T @ (eigenvalue*alpha), [1,1,3])
    return x + add_quantity