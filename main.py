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
