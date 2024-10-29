import os
import re
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import keras
import cv2
import tensorflow as tf

import time


# Record the start time
start_time = time.time()

# Define constants
TARGET_WIDTH = 128
TARGET_HEIGHT = 128

# path to the model
MODEL_PATH = './trained_model'

# Path to the folder containing images
IMAGE_FOLDER = './ch'  

# Load the trained convolutional neural network
print("[INFO] Loading my model...")
# model = load_model(MODEL_PATH, compile=False)
# model = load_model(MODEL_PATH)
# model = keras.layers.TFSMLayer('./trained_model', call_endpoint='serving_default')

model = tf.saved_model.load(MODEL_PATH)


# Labels
labels = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'ba', 'cha', 'pa'
]

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
    if max_prob > 0.9:
        idx = np.argmax(prob)
        predicted_label = labels[idx]
    else:
        predicted_label = "Unknown"

    predicted_labels.append(predicted_label)

# Print the predicted labels for all images
print("Predicted labels:", predicted_labels)


# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

# Print the elapsed time
print("Execution time:", elapsed_time, "seconds")

