#! unzip '/content/drive/MyDrive/Colab Notebooks/NN Project/tiny-imagenet-200.zip' -d '/content/drive/MyDrive/Colab Notebooks/NN Project/'

import os
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
from sklearn import preprocessing

IMAGE_SIZE = 64
NUM_CHANNELS = 3
NUM_CLASSES = 200
NUM_IMAGES_PER_CLASS = 500
NUM_IMAGES = int(NUM_CLASSES * NUM_IMAGES_PER_CLASS * 0.1)


"""
Randomly load part of the training images for calculating the mean, std, eigenvalues, eigenvectors for preprocessing and standard color augmentation.
"""
def load_training_images(image_dir):

    image_index = 0
    
    images = np.ndarray(shape=(NUM_IMAGES, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

    # Loop through all the types directories
    for type in os.listdir(image_dir):
        if os.path.isdir(image_dir + type + '/images/'):
            type_images = os.listdir(image_dir + type + '/images/')
            
            # Loop through part of the images of a type directory
            index = np.random.choice(a=NUM_IMAGES_PER_CLASS, size= int(0.1* NUM_IMAGES_PER_CLASS), replace=False)
            for i in index:
                image = type_images[i]
                image_file = os.path.join(image_dir, type + '/images/', image)

                # reading the images as they are; no normalization, no color editing
                image_data = mpimg.imread(image_file) 

                if (image_data.shape == (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)):
                    images[image_index, :] = image_data.reshape([1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
                    image_index += 1
    
    images = images[:image_index]
    
    return images


if __name__ =='__main__':
    TRAINING_IMAGES_DIR = '/content/drive/MyDrive/Colab Notebooks/NN Project/tiny-imagenet-200/train/'
    TRAIN_SIZE = NUM_IMAGES

    training_images, training_labels = load_training_images(TRAINING_IMAGES_DIR)
    training_images = training_images[:len(training_labels)]

    le = preprocessing.LabelEncoder()
    training_le = le.fit(training_labels)
    training_labels_encoded = training_le.transform(training_labels)

    np.savez('/content/drive/MyDrive/Colab Notebooks/NN Project/train', x=training_images, y=training_labels_encoded)

    """Load validation data"""

    VAL_IMAGES_DIR = '/content/drive/MyDrive/Colab Notebooks/NN Project/tiny-imagenet-200/val/'

    val_data = pd.read_csv(VAL_IMAGES_DIR + 'val_annotations.txt', sep='\t', header=None, names=['File', 'Class', 'X', 'Y', 'H', 'W'])
    val_data = val_data.set_index('File')

    val_images, val_labels = load_validation_images(VAL_IMAGES_DIR, val_data)
    val_images = val_images[:len(val_labels)]

    le = preprocessing.LabelEncoder()
    val_le = le.fit(val_labels)
    val_labels_encoded = val_le.transform(val_labels)

    np.savez('/content/drive/MyDrive/Colab Notebooks/NN Project/test', x=val_images, y=val_labels_encoded)