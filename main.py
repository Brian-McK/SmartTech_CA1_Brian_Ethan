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


# def le_net_model():
#     model = Sequential()
#     model.add(Conv2D(90, (5, 5), input_shape=(64, 64, 1), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Conv2D(90, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Flatten())
#     model.add(Dense(500, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(num_classes, activation='softmax'))
#     model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
#     return model
#
# def modified_model():
#     model = Sequential()
#     model.add(Conv2D(90, (5, 5), input_shape=(64, 64, 1), activation='relu'))
#     model.add(Conv2D(9, (5, 5), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Conv2D(60, (3, 3), activation='relu'))
#     model.add(Conv2D(60, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     # model.add(Dropout(0.5))
#     model.add(Flatten())
#     model.add(Dense(500, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(num_classes, activation='softmax'))
#     model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

# def path_leaf(path):
#    head, tail = ntpath.split(path)
#    return tail


# def input_model_function(filename):
#     csv_filename =[filename]
#     dataset = tf.data.TextLineDataset(csv_filename)
#     dataset = dataset.map(_parse_function)
#     dataset = dataset.batch(20)# you can use any number of batching
#     iterator = dataset.make_one_shot_iterator()
#     sess = tf.Session()
#     batch_images, batch_labels = sess.run(iterator.get_next())
#     return {'x':batch_images}, batch_labels
#
# def _parse_function(line):
#     image, labels= tf.decode_csv(line,record_defaults=[[""], [0]])
#     # Decode the raw bytes so it becomes a tensor with type.
#     image = imread(image)# give full path name of image
#     return image, labels


def get_image(data_dir):
    onlyfiles = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
    images = []
    for n in range(0, len(onlyfiles)):
        images.append(cv2.imread(join(data_dir, onlyfiles[n])))
    image_arr = np.array(images)
    return image_arr

def get_image_size(data_dir):
    onlyfiles = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
    images= np.empty(len(onlyfiles), dtype=object)
    for n in range(0, len(onlyfiles)):
        images[n] = cv2.imread(join(data_dir, onlyfiles[n]))
    return images


def get_load_train_file(file_directory):
    entries = os.listdir(file_directory)
    train_data_files_name = []
    for entry in entries:
        train_data_files_name.append(entry)
    train_files_np = np.array(train_data_files_name)
    return train_files_np

def get_image2(train_dir,file_dir):
    images = []
    for files in file_dir:
        file_paths = train_dir+'\\'+files+'\\'+'images'
        im = get_image(file_paths)
        for i in im:
            images.append(i)
    image_arr = np.array(images)
    return image_arr



train_data_dir = "D:\\OneDrive - Dundalk Institute of Technology\\Documents\\smart\\tiny-imagenet-200\\train\\n01443537\\images"
validation_data_dir = "D:\\OneDrive - Dundalk Institute of Technology\\Documents\\smart\\tiny-imagenet-200\\val\\images"
test_data_dir = "D:\\OneDrive - Dundalk Institute of Technology\\Documents\\smart\\tiny-imagenet-200\\test\\images"
dat_dir = "D:\\OneDrive - Dundalk Institute of Technology\\Documents\\smart\\tiny-imagenet-200"
dat_dir2 = "D:\\OneDrive - Dundalk Institute of Technology\\Documents\\smart\\tiny-imagenet-200\\train"


wnids_data = os.path.join(dat_dir, 'wnids.txt')
words_data = os.path.join(dat_dir, 'words.txt')


train_txt_data = []
train_files = get_load_train_file(dat_dir2)
for file in train_files:
    file_path = dat_dir2+'\\'+file+'\\'+file+'_boxes.txt'
    with open(file_path) as fp:
        line = fp.readline()
        cnt = 1
        while line:
            # print("Line {}: {}".format(cnt, line.strip()))
            line = fp.readline()
            train_txt_data.append(line)
            cnt += 1
train_tx_data = np.array(train_txt_data)

X_train, y_train = get_image2(dat_dir2,train_files), train_tx_data
X_val, y_val = get_image(validation_data_dir), get_image_size(validation_data_dir)
X_test, y_test = get_image(test_data_dir), get_image_size(test_data_dir)

print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)
print(X_test.shape)
print(y_test.shape)

# read_file = pd.read_csv(r'C:\\Users\\admin\\Documents\\smart\\tiny-imagenet-200\\words.txt',index_col = False,sep="\t")
# read_file.to_csv (r'C:\\Users\\admin\\Documents\\smart\\tiny-imagenet-200\\words.csv', index=None)

get_wnids_data = pd.read_csv(wnids_data, sep="\t", header=None)
get_words_data = pd.read_csv(words_data,sep="\t", header=None)
#
print("data shape ",get_words_data.shape,type(get_words_data))

data = pd.read_csv('D:\\OneDrive - Dundalk Institute of Technology\\Documents\\smart\\tiny-imagenet-200\\words.csv', header=None)
print(data)


assert(X_train.shape[0] == y_train.shape[0]), "The training set does not have the same number of data points and labels"
assert(X_val.shape[0] == y_val.shape[0]), "The validation set does not have the same number of data points and labels"
assert(X_test.shape[0] == y_test.shape[0]), "The test set does not have the same number of data points and labels"

num_of_samples = []
cols = 5
num_classes = 10
fig, axs = plt.subplots(nrows = num_classes, ncols=cols, figsize=(5, 50))
fig.tight_layout()
for i in range(cols):

    for j, row in data.iterrows():
        #print(y_train[j])
        #print(row.get(1))
        #y=y_train[j]
        #print(y[0:9])
        #x_selected = X_train[y_train[0:9] == row.get(0)]
        x_selected = X_train[y_train == row]
        # print("selected",x_selected)
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected)-1), :, :], cmap = plt.get_cmap('gray'))
        axs[j][i].axis('off')
        if i==2:
            axs[j][i].set_title(str(j))
            num_of_samples.append(len(x_selected))
plt.show()

