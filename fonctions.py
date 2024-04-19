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

def load_images_from_directory(directory):
    X = []
    y = []
    for class_name in os.listdir(directory):
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            for image_file in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_file)
                image = cv2.imread(image_path)
                if image is not None:
                    X.append(image)
                    y.append(class_name)
    return np.array(X), np.array(y)

def preprocess_images(images):
    processed_images = []
    for image in images:
        # Preprocess image (e.g., resize, normalize, etc.)
        processed_image = cv2.resize(image, (224, 224))
        processed_image = processed_image / 255.0
        processed_images.append(processed_image)
    return np.array(processed_images)

def define_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train):
    early_stopping = EarlyStopping(patience=5, monitor='val_accuracy', mode='max')
    history = model.fit(X_train, y_train, epochs=20, validation_split=0.2, callbacks=[early_stopping])
    return history