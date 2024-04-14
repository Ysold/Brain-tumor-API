from io import BytesIO
from fastapi import File
import os
from PIL import Image
import numpy as np
from openai import OpenAI
from fastapi import FastAPI, UploadFile
import requests
from fastapi import FastAPI, HTTPException
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer

from pydantic import BaseModel
from dotenv import load_dotenv
from keras.models import load_model
from keras.models import Sequential
from tensorflow.keras.layers import Input
from keras.applications import VGG16
from tensorflow.keras import layers
from keras.optimizers import RMSprop
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.vgg16 import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from keras.models import load_model


load_dotenv()


tags_metadata = [
    {
        "name": "Text",
        "description": "Operations with text.",
    },
    {
        "name": "Numbers",
        "description": "Operations with numbers.",
    },
]

app = FastAPI(
    title="Tumor API",
    openapi_tags=tags_metadata,
    description="""
# Title
This is a very fancy project, with auto docs for the API and everything"
""",
)

model = load_model('brain_tumor.h5')

def prepare_image(image):
    # Load the image
    img = load_img(image, target_size=(128, 128))
    # Convert to array
    img_array = img_to_array(img)
    # Expand dimension to match model input_shape
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize pixel values
    img_array = img_array / 255.0
    return img_array

@app.post("/predict")
async def prediction(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    
    # Read image as bytes
    content = await file.read()
    # Prepare image for prediction
    image = prepare_image(BytesIO(content))
    # Make prediction
    prediction = model.predict(image)
    
    # Determine class based on prediction probability
    if prediction[0] > 0.5:
        class_name = "Tumor"
    else:
        class_name = "No Tumor"

    return {"prediction": class_name, "probability": float(prediction[0])}




class ResponseModel(BaseModel):
    generated_text: str
    
@app.get("/model", tags=["Model"], response_model=List[ResponseModel], responses={200: {"model": List[ResponseModel], "description": "Successful Response"}, 500: {"description": "Internal Server Error"}})
def generate_response():

    messages = [
        {"role": "system", "content": "Tu es un assistant médical spécialisé dans les tumeurs."},
        {"role": "user", "content": "Qu'est-ce qu'une tumeur ?"},
        {"role": "assistant", "content": "Une tumeur est une masse de tissu qui se forme lorsqu'une croissance cellulaire anormale se produit. Les tumeurs peuvent être bénignes (non cancéreuses) ou malignes (cancéreuses)."},
        {"role": "user", "content": "Quels sont les différents types de tumeurs ?"}
    ]

    inputs = " ".join([message["content"] for message in messages])
    
    try:
        API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
        headers = {"Authorization": f"Bearer "+ os.getenv('HUGGING_FACE_KEY')}
        data = {
            "inputs": inputs,
            "options": {
                "use_cache": False
            }
        }

        response = requests.post(API_URL, headers=headers, json=data)
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))