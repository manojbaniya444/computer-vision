
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import tensorflow as tf


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


class ClassificationModel:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ClassificationModel, cls).__new__(cls)
            # loading the model
            cls._instance.model = tf.saved_model.load(MODEL_PATH)

        return cls._instance

    def preprocess_image(self, image):
        pass

    def classify_character(self, image):
                
        ## ?Shape of our image should be 64x64 and 3 channelss 1 batch
        # image = cv2.resize(image, (TARGET_WIDTH, TARGET_HEIGHT))

        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
    
    
        ## ? if image is eroded/grayscale
    
        ## ? if image is eroded/grayscale it will be in (1,64,64,1) shape so we need to convert it to (1,64,64,3) for our model
        black_and_white = np.concatenate((image, image, image), axis=-1)  #*For option2 this step is necessary
    
        # predicting
        prob = model(black_and_white, training=False)[0]
        max_prob = np.max(prob)
        # print(max_prob)
        if max_prob > 0.9:
            idx = np.argmax(prob)
            predicted_label = labels[idx]
        else:
            predicted_label = " "

        return predicted_label