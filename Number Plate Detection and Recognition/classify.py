import os
import re
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import keras
import cv2
import tensorflow as tf

import time


# Define constants
TARGET_WIDTH = 64
TARGET_HEIGHT = 64

# path to the model
MODEL_PATH = './trained_models/classification_model'

model = tf.saved_model.load(MODEL_PATH)

# Labels
labels = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'ba', 'cha', 'kha', 'ko', 'me', 'pa', 'pra'
]


#####_______________PREPROCESS IMAGE____________________
def preprocess_image(original_image):
    # ?Process the image if want to convert to binary
    # image = cv2.resize(original_image, (TARGET_WIDTH, TARGET_HEIGHT))
    # original_image = cv2.imread(filename)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    gray_image = cv2.resize(gray_image, (TARGET_HEIGHT,TARGET_WIDTH))

    ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

    # Filter out small blobs
    min_blob_area = 400  # Minimum area threshold for blobs
    filtered_labels = labels.copy()
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_blob_area:
            filtered_labels[labels == label] = 0

    # Create the filtered image
    final_image = np.where(filtered_labels > 0, 255, 0).astype(np.uint8)
  
  
#####________________________Extracting the characters____________________

def classify_character(image):
    
    ## ?Shape of our image should be 64x64 and 3 channelss 1 batch
    image = cv2.resize(image, (TARGET_WIDTH, TARGET_HEIGHT))

    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    
    
    ## ? if image is eroded/grayscale
    
    ## ? if image is eroded/grayscale it will be in (1,64,64,1) shape so we need to convert it to (1,64,64,3) for our model
    black_and_white = np.concatenate((image, image, image), axis=-1)
    
    # predicting
    prob = model(black_and_white, training=False)[0]
    max_prob = np.max(prob)
    # print(max_prob)
    if max_prob > 0.9:
        idx = np.argmax(prob)
        predicted_label = labels[idx]
    else:
        predicted_label = "X"

    return predicted_label