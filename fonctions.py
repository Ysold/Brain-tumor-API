import os
import numpy as np
import cv2

from keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from keras.preprocessing.image import img_to_array, load_img

def prepare_image(image):
    # Load the image
    img = load_img(image, target_size=(224, 224))
    # Convert to array
    img_array = img_to_array(img)
    # Expand dimension to match model input_shape
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize pixel values
    img_array = img_array / 255.0
    return img_array


def read_uploaded_file(uploaded_file):
    return uploaded_file.read()