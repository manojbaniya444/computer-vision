import os
import re
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import keras
import cv2
import tensorflow as tf

# Define constants
TARGET_WIDTH = 128
TARGET_HEIGHT = 128

# path to the model
MODEL_PATH = './trained_model'

IMAGE_FOLDER = './characters'

model = tf.saved_model.load(MODEL_PATH)

# Labels
labels = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'ba', 'cha', 'pa'
]

def getLicenseCharacters(license_image):
    image = cv2.resize(license_image, (TARGET_WIDTH, TARGET_HEIGHT))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # Classify the input image then find the index of the class with the *largest* probability
    # print("[INFO] Classifying image:", filename)
    # prob = model.predict(image)[0]
    prob = model(image, training=False)[0]
    max_prob = np.max(prob)
    print(max_prob)
    if max_prob > 0.8:
        idx = np.argmax(prob)
        predicted_label = labels[idx]
    else:
        predicted_label = "X"

    return predicted_label


def getAllLicenseCharacters():
        # Initialize a list to store predicted labels
    predicted_labels = []

    # Define a function to extract numeric part from filename
    def extract_numeric_part(filename):
        return int(re.search(r'\d+', filename).group())

    # Get sorted list of image filenames in the folder based on numeric order
    image_files = sorted([filename for filename in os.listdir(IMAGE_FOLDER) if filename.endswith(('.jpg', '.jpeg', '.png'))],
                        key=extract_numeric_part)

    # Iterate over the images in ascending order
    for filename in image_files:
        # Load the image
        image_path = os.path.join(IMAGE_FOLDER, filename)
        original_image = cv2.imread(image_path)
        
        # Preprocess the image
        image = cv2.resize(original_image, (TARGET_WIDTH, TARGET_HEIGHT))
        
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        # Classify the input image then find the index of the class with the *largest* probability
        # print("[INFO] Classifying image:", filename)
        # prob = model.predict(image)[0]
        prob = model(image, training=False)[0]
        max_prob = np.max(prob)
        print(max_prob)
        if max_prob > 0.8:
            idx = np.argmax(prob)
            predicted_label = labels[idx]
            # print(predicted_label)
        else:
            predicted_label = "X"

        predicted_labels.append(predicted_label)

    predicted_labels_string = ' '.join(predicted_labels)

    return predicted_labels_string