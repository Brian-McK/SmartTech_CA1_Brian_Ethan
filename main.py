import time
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import random
import requests
from PIL import Image
import cv2
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import pickle
import pandas as pd

# declare constants
NUM_CLASSES = 200
NUM_IMAGES_PER_CLASS = 500
IMG_SIZE = 64


# directories & files paths
test_directory = "tiny-imagenet-200/test"
train_directory = "tiny-imagenet-200/train"
val_directory = "tiny-imagenet-200/val"
wnids_file = "tiny-imagenet-200/wnids.txt"
words_file = "tiny-imagenet-200/words.txt"


# process grey scale
def grey_scale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


# process gaussian blur
def gaussian_blur(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img


# process equaliser
def equalise(img):
    img = cv2.equalizeHist(img)
    return img


# combine processes
def pre_process_img(img):
    img = grey_scale(img)
    img = gaussian_blur(img)
    img = equalise(img)
    img = img / 255
    return img


# Following solution from the reference with modifications:
# ***************************************************************************************/
# *    Title: Train_ResNet_On_Tiny_ImageNet
# *    Author: Giri, S
# *    Date: Mar 29, 2019
# *    Code version: 1.0
# *    Availability: https://github.com/sonugiri1043/Train_ResNet_On_Tiny_ImageNet
# ***************************************************************************************/
# (Version 1.0) [Source code]. https://github.com/sonugiri1043/Train_ResNet_On_Tiny_ImageNet/blob/master/Train_ResNet_On_Tiny_ImageNet.ipynb


# get class ids from text file
def get_id_dictionary():
    id_dict = {}
    for i, line in enumerate(open(wnids_file, 'r')):
        id_dict[line.replace('\n', '')] = i
    return id_dict


def get_class_to_id_dict():
    id_dict = get_id_dictionary()
    all_classes = {}
    result = {}
    for i, line in enumerate(open(words_file, 'r')):
        n_id, word = line.split('\t')[:2]
        all_classes[n_id] = word
    for key, value in id_dict.items():
        result[value] = (key, all_classes[key])
    return result


def get_data(id_dict):
    print('starting loading data')
    train_data, test_data = [], []
    train_labels, test_labels = [], []
    t = time.time()
    for key, value in id_dict.items():
        train_data += [plt.imread(train_directory + '/{}/images/{}_{}.JPEG'.format(key, key, str(i))) for i in
                       range(NUM_CLASSES)]
        train_labels_ = np.array([[0] * NUM_CLASSES] * NUM_IMAGES_PER_CLASS)
        train_labels_[:, value] = 1
        train_labels += train_labels_.tolist()

    for line in open(val_directory + '/val_annotations.txt'):
        img_name, class_id = line.split('\t')[:2]
        test_data.append(plt.imread(val_directory + '/images/{}'.format(img_name)))
        test_labels_ = np.array([[0] * NUM_CLASSES])
        test_labels_[0, id_dict[class_id]] = 1
        test_labels += test_labels_.tolist()

    print('finished loading data, in {} seconds'.format(time.time() - t))
    return np.array(train_data), np.array(train_labels), np.array(test_data), np.array(test_labels)


train_data, train_labels, test_data, test_labels = get_data(get_id_dictionary())

train_data = np.array(list(map(pre_process_img, train_data)))
test_data = np.array(list(map(pre_process_img, test_data)))

assert (train_data.shape[0] == train_labels.shape[
    0]), "The training set does not have the same number of data points and labels"
assert (train_data.shape[0] == train_labels.shape[0]), "The test set does not have the same number of data points and labels"


# test image display
displayImage = train_data[0]
plt.imshow(displayImage)
plt.axis("off")
plt.show()

print(displayImage.shape)

# print out shapes
print("train data shape: ", train_data.shape)
print("train label shape: ", train_labels.shape)
print("test data shape: ", test_data.shape)
print("test labels.shape: ", test_labels.shape)

# Generate data augmentation and fit it to the training data
datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1,
                             rotation_range=10)
datagen.fit(train_data)

# One-hot encode labels
train_labels = to_categorical(train_labels, NUM_CLASSES)
test_labels = to_categorical(test_labels, NUM_CLASSES)
