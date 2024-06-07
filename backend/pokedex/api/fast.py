''' API ENDPOINTS '''
# IMPORTS
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pokedex.model_logic.registry import load_model
from pokedex.params import *

import numpy as np
import cv2
import json


# Starting API
app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# MODEL UPLOAD
app.state.model_15 = load_model(stage='Production', model_type='15')
print("✅ model 15 loaded\n")

app.state.model_150 = load_model(stage='Production', model_type='150')
print("✅ model 150 loaded\n")

# app.state.model_GAN = load_model()
# print("✅ model GAN loaded\n")


# ENDPOINTS
@app.get("/")
def index():
    ''' test for root'''
    return {"status": "ok"}


@app.post('/predict_type')
async def predict_type(img: UploadFile=File(...)):
    '''
    returns the predicted type of the image given as input
    '''
    print("\n⭐️ API call : predict_type")
    # Receive and decode the image
    contents = await img.read()
    nparr = np.fromstring(contents, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # type(cv2_img) => numpy.ndarray

    if cv2_img is None:
        raise HTTPException(status_code=400, detail="Invalid image format")

    # Preprocess the image for the model (this might change depending on your model's requirements)
    new_size = (128, 128) # TODO
    img_resized = cv2.resize(cv2_img, new_size)  # Resize to the size your model expects
    img_array = np.expand_dims(img_resized, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image
    print("✅ images loaded and processed\n")

    # Make prediction
    model = app.state.model_15
    assert model is not None
    predictions = model.predict(img_array)

    # decode prediction
    labels_type_dict  = json.loads(LABELS_TYPE)
    predicted_index = np.argmax(predictions, axis=1)
    predicted_labels = [labels_type_dict[str(i)] for i in predicted_index]
    confidence_scores = np.max(predictions, axis=1)
    predicted_labels_confidence_dict = [[k, v] for k, v in zip(predicted_labels, confidence_scores)]

    print('predicted_labels_confidence_dict', {predicted_labels_confidence_dict[0][0] : f'{(round(predicted_labels_confidence_dict[0][1]*100, 2))}%'})
    print("✅ predict_type done \n")
    return {predicted_labels_confidence_dict[0][0] : f'{(round(predicted_labels_confidence_dict[0][1]*100, 2))}%'}


@app.post('/predict_name')
async def predict_name(img: UploadFile=File(...)):
    '''
    returns the predicted pokemon name of the image given as input
    '''
    print("\n⭐️ API call : predict_name")
    # Receive and decode the image
    contents = await img.read()
    nparr = np.fromstring(contents, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # type(cv2_img) => numpy.ndarray

    if cv2_img is None:
        raise HTTPException(status_code=400, detail="Invalid image format")

    # Preprocess the image for the model (this might change depending on your model's requirements)
    new_size = (128, 128) # TODO
    img_resized = cv2.resize(cv2_img, new_size)  # Resize to the size your model expects
    img_array = np.expand_dims(img_resized, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image
    print("✅ images loaded and processed\n")

    # Make prediction
    model = app.state.model_150
    assert model is not None
    predictions = model.predict(img_array)

    # decode prediction
    labels_type_dict  = json.loads(LABELS_NAME)
    predicted_index = np.argmax(predictions, axis=1)
    predicted_labels = [labels_type_dict[str(i)] for i in predicted_index]
    confidence_scores = np.max(predictions, axis=1)
    predicted_labels_confidence_dict = [[k, v] for k, v in zip(predicted_labels, confidence_scores)]

    print('predicted_labels_confidence_dict', {predicted_labels_confidence_dict[0][0] : f'{(round(predicted_labels_confidence_dict[0][1], 2))*10}%'})
    print("✅ predict_name done \n")
    return {predicted_labels_confidence_dict[0][0] : f'{(round(predicted_labels_confidence_dict[0][1]*100, 2))}%'}


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
