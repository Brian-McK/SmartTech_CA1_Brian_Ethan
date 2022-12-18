import csv
import time

import imageio
import numpy as np
import tensorflow as tf
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
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import pandas as pd
import cv2
import os
import ntpath
from matplotlib.image import imread
from os import listdir
from os.path import isfile, join

def gray_scale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalise(img):
    img = cv2.equalizeHist(img)
    return img


def preprocess(img):
    img = gray_scale(img)
    img = equalise(img)
    img = img/255
    return img


def le_net_model():
    model = Sequential()
    model.add(Conv2D(60, (5, 5), input_shape=(64, 64, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def modified_model():
    model = Sequential()
    model.add(Conv2D(60, (5, 5), input_shape=(64, 64, 1), activation='relu'))
    model.add(Conv2D(60, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(200, activation='softmax'))
    model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

#load the images from the file
def get_image(data_dir):
    onlyfiles = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
    images = []
    for n in range(0, len(onlyfiles)):
        images.append(cv2.imread(join(data_dir, onlyfiles[n])))
    image_arr = np.array(images)
    return image_arr

#load the images from the file with the file path parameter
def get_image_size(data_dir):
    onlyfiles = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
    images= np.empty(len(onlyfiles), dtype=object)
    for n in range(0, len(onlyfiles)):
        images[n] = cv2.imread(join(data_dir, onlyfiles[n]))
    return images



#get the values from the wnids.tst as the id of the class category
def get_id_from_dictionary(path):
    id_dict = {}
    for i, line in enumerate(open(path + 'wnids.txt', 'r')):
        id_dict[line.replace('\n', '')] = i
    return id_dict

#get the category name from the words.txt and put them into the relevant key id
def get_class_to_id_dict(path):
    id_dict = get_id_from_dictionary(path)
    all_classes = {}
    result = {}
    for i, line in enumerate(open(path + 'words.txt', 'r')):
        n_id, word = line.split('\t')[:2]
        all_classes[n_id] = word
    for key, value in id_dict.items():
        result[value] = (key, all_classes[key])
    return result

def get_data(id_dict,path):
    print('starting loading data')
    X_train, X_valid = [], []
    y_train, y_valid = [], []
    t = time.time()
    for key, value in id_dict.items():
        X_train += [imageio.imread(path + 'train\\{}\\images\\{}_{}.JPEG'.format(key, key, str(i)), pilmode='RGB') for i in
                       range(500)]
        train_labels_ = np.array([[0] * 200] * 500)
        train_labels_[:, value] = 1
        y_train += train_labels_.tolist()

    for line in open(path + 'val\\val_annotations.txt'):
        img_name, class_id = line.split('\t')[:2]
        X_valid.append(imageio.imread(path + 'val\\images\\{}'.format(img_name), pilmode='RGB'))
        test_labels_ = np.array([[0] * 200])
        test_labels_[0, id_dict[class_id]] = 1
        y_valid += test_labels_.tolist()

    print('finished loading data, in {} seconds'.format(time.time() - t))
    return np.array(X_train), np.array(y_train), np.array(X_valid), np.array(y_valid)

train_data_dir = "D:\\OneDrive - Dundalk Institute of Technology\\Documents\\smart\\tiny-imagenet-200\\train\\n01443537\\images"
validation_data_dir = "D:\\OneDrive - Dundalk Institute of Technology\\Documents\\smart\\tiny-imagenet-200\\val\\images"
validation_data_dir2 = "D:\\OneDrive - Dundalk Institute of Technology\\Documents\\smart\\tiny-imagenet-200\\val"
test_data_dir = "D:\\OneDrive - Dundalk Institute of Technology\\Documents\\smart\\tiny-imagenet-200\\test\\images"
dat_dir = "D:\\OneDrive - Dundalk Institute of Technology\\Documents\\smart\\tiny-imagenet-200\\"
dat_dir2 = "D:\\OneDrive - Dundalk Institute of Technology\\Documents\\smart\\tiny-imagenet-200\\train"


wnids_data = os.path.join(dat_dir, 'wnids.txt')
words_data = os.path.join(dat_dir, 'words.txt')

X_train, y_train,X_val, y_val = get_data(get_id_from_dictionary(dat_dir),dat_dir)

# validation_data = os.path.join(validation_data_dir2,'val_annotations.txt')
# validation_txt = pd.read_csv(validation_data,sep="\t", header=None)
# validation_txt.columns =['Images', 'Class', 'Pixels', 'Pixels2', 'Pixels3', 'Pixels4']
# print(validation_txt)
# validation_txt.to_csv (r'D:\\OneDrive - Dundalk Institute of Technology\\Documents\\smart\\tiny-imagenet-200\\val\\val.csv', index=None)

train_data = os.path.join(validation_data_dir2,'val_annotations.txt')
# validation_txt = pd.read_csv(validation_data,sep="\t", header=None)
# validation_txt.columns =['Images', 'Class', 'Pixels', 'Pixels2', 'Pixels3', 'Pixels4']
# print(validation_txt)
# validation_txt.to_csv (r'D:\\OneDrive - Dundalk Institute of Technology\\Documents\\smart\\tiny-imagenet-200\\val\\val.csv', index=None)

train_txt_data = []
y_labels =[]
train_files = get_load_train_file(dat_dir2)
# t = time.time()
# for key, value in get_id_dictionary(dat_dir).items():
#     y_labels = np.array([0]*200*500)
#     y_labels[:, value] = 1
#     y_labels += y_labels.tolist()
# for file in train_files:
#     file_path = dat_dir2+'\\'+file+'\\'+file+'_boxes.txt'
#     with open(file_path) as fp:
#         line = fp.readline()
#         cnt = 1
#         while line:
#             # print("Line {}: {}".format(cnt, line.strip()))
#             line = fp.readline()
#             train_txt_data.append(line)
#             cnt += 1
# train_tx_data = np.array(train_txt_data, dtype='<U9')

filename = 'D:\\OneDrive - Dundalk Institute of Technology\\Documents\\smart\\tiny-imagenet-200\\val\\val.csv'


X_test, y_test = get_image(test_data_dir), get_image_size(test_data_dir)

print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)
print(X_test.shape)
print(y_test.shape)

# read_file = pd.read_csv(r'D:\\OneDrive - Dundalk Institute of Technology\\Documents\\smart\\tiny-imagenet-200\\words.txt',index_col = False,sep="\t")
# read_file.to_csv (r'D:\\OneDrive - Dundalk Institute of Technology\\Documents\\smart\\tiny-imagenet-200\\words.csv', index=None)

get_wnids_data = pd.read_csv(wnids_data, sep="\t", header=None)
get_words_data = pd.read_csv(words_data,sep="\t", header=None)


data = pd.read_csv('D:\\OneDrive - Dundalk Institute of Technology\\Documents\\smart\\tiny-imagenet-200\\words.csv', header=None)

assert(X_train.shape[0] == y_train.shape[0]), "The training set does not have the same number of data points and labels"
assert(X_val.shape[0] == y_val.shape[0]), "The validation set does not have the same number of data points and labels"
assert(X_test.shape[0] == y_test.shape[0]), "The test set does not have the same number of data points and labels"


X_train = np.array(list(map(preprocess, X_train)))
X_val = np.array(list(map(preprocess, X_val)))
X_test = np.array(list(map(preprocess, X_test)))

X_train = X_train.reshape(100000, 64, 64, 1)
X_val = X_val.reshape(10000, 64, 64, 1)
X_test = X_test.reshape(10000, 64, 64, 1)



datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1, rotation_range=10)
datagen.fit(X_train)

test_datagen = ImageDataGenerator( rescale = 1.0/255. )



# batches = datagen.flow(X_train, y_train, batch_size=20)
# X_batch, y_batch = next(batches)
# fig, axs = plt.subplots(1, 20, figsize=(20,5))
# for i in range(20):
#     axs[i].imshow(X_batch[i].reshape(32, 32))
#     axs[i].axis("off")
# plt.show()
num_classes = 200


# One hot encode all labels
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)
# y_test = to_categorical(y_test, num_classes)

model = modified_model()
print(model.summary())

y_train = np.argmax(y_train,axis=1)

# Train the model and evaluate its performance
h = model.fit(datagen.flow(X_train, y_train, batch_size=500), steps_per_epoch=X_train.shape[0]/500, epochs=10, validation_data=(X_val, y_val),  verbose=1)
plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

# score = model.evaluate(X_test, y_test, verbose=1)
# print('Test score: ', score[0])
# print('Test accuracy: ', score[1])