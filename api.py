from io import BytesIO
from fastapi import File
import os
from PIL import Image
import numpy as np
from fastapi import FastAPI, UploadFile, HTTPException
import requests
from typing import List

from pydantic import BaseModel
from dotenv import load_dotenv
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input


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


model = load_model('model.h5')

def prepare_image(image, target):
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

def predict(image, model):
    predictions = model.predict(image)
    
    top_score = predictions.max()
    
    response = {"score": float(round(top_score, 3))}
    
    return response

class Prediction(BaseModel):
    filename: str
    content_type: str
    predictions: dict = []

@app.post("/predict", response_model=Prediction)
async def prediction(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    content = await file.read()
    image = Image.open(BytesIO(content)).convert("RGB")
    # preprocess the image and prepare it for classification
    image = prepare_image(image, target=(224, 224))
    response = predict(image, model)

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "predictions": response,
    }


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