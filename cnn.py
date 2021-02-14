import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
#from pyimagesearch.lenet import LeNet
#from imutils import paths
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# load local data
def listdir_fullpath(d):
    return np.sort([os.path.join(d, f) for f in os.listdir(d)])

def load_images(paths):
    data = []
    for imagePath in paths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        image = img_to_array(image)
        data.append(image)
    data = np.array(data, dtype="float") / 255.0
    return data

test_input = listdir_fullpath("./road_segmentation/testing/input/")
test_output = listdir_fullpath("./road_segmentation/testing/output/")
train_input = listdir_fullpath("./road_segmentation/training/input/")
train_output = listdir_fullpath("./road_segmentation/training/output/")

test_input_data = load_images(test_input)
test_output_data = load_images(test_output)
train_input_data = load_images(train_input)
train_output_data = load_images(train_output)