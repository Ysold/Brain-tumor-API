import io
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, HTTPException
import os
import numpy as np
from fastapi import FastAPI, UploadFile
import requests
from fastapi import FastAPI, HTTPException
from typing import List
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from pydantic import BaseModel
from dotenv import load_dotenv
from keras.models import load_model
from keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from keras.models import load_model
from dotenv import load_dotenv
from fastapi.responses import FileResponse
from openai import OpenAI
from fonctions import *

load_dotenv()

tags_metadata = [
    {
        "name": "Image",
        "description": "Operations with image data.",
    }
]

app = FastAPI(
    title="Brain Tumor Image Classifier",
    openapi_tags=tags_metadata,
    description="""
    # Brain Tumor Image Classifier
    This API predicts whether an uploaded image contains a brain tumor or not.
    """
)

model = load_model('model.h5')

@app.post("/predict", tags=["Prediction"], summary="Make a prediction",
description="""This route accepts png or jpeg data, passes it to a pre-trained 
machine learning model, and returns a prediction based on this data.""")

async def prediction(file: UploadFile = File(...)):
    content = await file.read()
    image = prepare_image(BytesIO(content))
    prediction = model.predict(image)
    
    if prediction[0] > 0.5:
        class_name = "Tumor"
    else:
        class_name = "No Tumor"

    return {"prediction": class_name, "probability": float(prediction[0])}

@app.post("/training", tags=["Training"], summary="Train a machine learning model",
description="""This route accepts a uploaded CSV file, uses it to train a machine learning model, and
returns a response with information about the training process.""")

async def train_model(file: UploadFile = File(...)):
    # Check if file is CSV
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV.")
    
    content = await file.read()
    
    file_like_object = io.BytesIO(content)
    
    df = pd.read_csv(file_like_object)
    
    # Assuming last column is target variable and all others are features
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Save trained model
    model_filename = "trained_model.pkl"
    joblib.dump(model, model_filename)
    
    return {"message": "Model trained successfully.", "accuracy": accuracy, "model_filename": model_filename}

@app.get("/download_model", tags=["Training"], summary="Download the trained machine learning model",
description="""This route returns the trained machine learning model as a downloadable file.""")
async def download_model():
    model_filename = "trained_model.pkl"
    return FileResponse(model_filename, media_type="application/octet-stream", filename=model_filename)


@app.post("/train_tensorflow", tags=["Training"], summary="Train a TensorFlow model", 
description="""This route accepts a CSV file, uses it to train a TensorFlow model, and returns a response with information about the training process.""")
async def train_tensorflow_model(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV.")
    
    content = await file.read()
    
    file_like_object = io.BytesIO(content)
    
    df = pd.read_csv(file_like_object)
    
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define model architecture
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Define early stopping
    early_stopping = EarlyStopping(patience=5, monitor='val_accuracy', mode='max')
    
    # Train model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, callbacks=[early_stopping])
    
    # Save trained model
    model_filename = "tensorflow_model.h5"
    model.save(model_filename)
    
    # Evaluate model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    
    return {"message": "TensorFlow model trained successfully.", "accuracy": test_acc, "model_filename": model_filename}
    
@app.get("/model", tags=["Model"], summary="Get model information",
description="""This route returns information about the machine learning model
 currently used by the API.""")
def generate_response():

    messages = [
        {"role": "system", "content": "Tu es un assistant médical spécialisé dans les tumeurs."},
        {"role": "user", "content": "Qu'est-ce qu'une tumeur ?"},
        {"role": "assistant", "content": "Une tumeur est une masse de tissu qui se forme lorsqu'une croissance cellulaire anormale se produit. Les tumeurs peuvent être bénignes (non cancéreuses) ou malignes (cancéreuses)."},
        {"role": "user", "content": "Quels sont les différents types de tumeurs ?"}
    ]

    try:
        client = OpenAI(api_key=os.getenv('OPEN_AI_KEY'))
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages= messages
        )

        return completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))