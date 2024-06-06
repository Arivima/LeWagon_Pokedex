from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
from pokedex.model_logic.registry import load_model
# from backend.pokedex.model_logic.model_classification import initialize_model_15, initialize_model_150, compile_model, train_model
from pokedex.params import *

import numpy as np
import cv2
import io

app = FastAPI()

# # Allow all requests (optional, good for development purposes)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods
#     allow_headers=["*"],  # Allows all headers
# )

# 💡 Preload the model to accelerate the predictions
# We want to avoid loading the heavy Deep Learning model from MLflow at each `get("/predict")`
# The trick is to load the model in memory when the Uvicorn server starts
# and then store the model in an `app.state.model` global variable, accessible across all routes!
# This will prove very useful for the Demo Day
app.state.model15 = load_model()
app.state.model150 = load_model()
app.state.modelgan = load_model()


@app.get("/")
def index():
    return {"status": "ok"}


@app.post('/predict_type')
async def predict_type(img: UploadFile=File(...)):
    # Receive and decode the image
    contents = await img.read()
    nparr = np.fromstring(contents, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # type(cv2_img) => numpy.ndarray

    if cv2_img is None:
        raise HTTPException(status_code=400, detail="Invalid image format")

    # Preprocess the image for the model (this might change depending on your model's requirements)
    img_resized = cv2.resize(cv2_img, (224, 224))  # Resize to the size your model expects
    img_array = np.expand_dims(img_resized, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image
    
    # Make prediction
    model = app.state.model15
    assert model is not None
    
    predictions = model.predict(img_array)
    predicted_label = LABELS_TYPE[np.argmax(predictions, axis=1)]
    
    return {"predicted_type": int(predicted_label), "confidence": float(predictions[0][predicted_label])}



@app.post('/predict_name')
async def predict_name(img: UploadFile=File(...)):
    # Receive and decode the image
    contents = await img.read()
    nparr = np.fromstring(contents, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # type(cv2_img) => numpy.ndarray

    if cv2_img is None:
        raise HTTPException(status_code=400, detail="Invalid image format")

    # Preprocess the image for the model (this might change depending on your model's requirements)
    img_resized = cv2.resize(cv2_img, (224, 224))  # Resize to the size your model expects
    img_array = np.expand_dims(img_resized, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image
    
    # Make prediction
    model = app.state.model150
    assert model is not None
    
    predictions = model.predict(img_array)
    predicted_label = LABELS_NAME[np.argmax(predictions, axis=1)]
    
    return {"predicted_name": int(predicted_label), "confidence": float(predictions[0][predicted_label])}

    

@app.get("/generate")
def generate():
    """
    Generate new images of Pokemons
    """

    # model = app.state.modelgan
    # assert model is not None


    # ⚠️ fastapi only accepts simple Python data types as a return value
    # among them dict, list, str, int, float, bool
    # in order to be able to convert the api response to JSON
    return {'new_pokemon': "pokedex_error"}
