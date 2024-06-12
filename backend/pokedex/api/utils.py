
from fastapi import HTTPException
from pokedex.model_logic.registry import load_model_from_gcs
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

def load_model(model_type, app_state, must_raise : bool = False, force_reload : bool = False):
    '''
    loads model to the app
    if force reloading skip to gcs
    - loads from app.state, if non existent
    - loads from gcs
    '''
    print(f'Loading model : {model_type} | must_raise : {must_raise} | force_reload : {force_reload}')
    # load from app.state
    if force_reload is True:
        model = None
    else:
        if model_type == '15':
            model = app_state.model_15
        elif model_type == '150':
            model = app_state.model_150
        elif model_type == 'GAN':
            model = app_state.model_GAN
        else:
            raise HTTPException(status_code=500, detail="Internal server error")

    # load from gcs
    if model is None:
        model = load_model_from_gcs(model_type=model_type)
        if model:
            print(f"✅ model {model_type} loaded to app\n")
        else:
            print(f"! model {model_type} NOT loaded to app\n")
            if must_raise:
                raise HTTPException(status_code=500, detail="Internal server error")
            return None
    else:
        print('Using model already loaded to app')
    return model



async def process_img(image):
    '''
    receives the user image and process it to be accepted by the model
    '''
    # Receive and decode the image
    contents = await image.read()

    with Image.open(BytesIO(contents)) as img:
        img = img.resize((128, 128))
        # img.save('image.jpeg')
        print(type(img))
        img_array = np.array(img)
        print(type(img_array), img_array.shape)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image format")

    # Preprocess the image for the model
    img_array = np.expand_dims(img_array, axis=0)
    print(type(img_array), img_array.shape)

    print("✅ images loaded and processed\n")
    return img_array



def decode_pred(predictions, decoder):
    '''
    takes an array with all probabilities and return the top probability and associated label
    '''
    # decode prediction
    print(type(predictions), predictions.shape, '\n', predictions)
    print(type(decoder), len(decoder), '\n', decoder)

    all_predictions_with_labels = []
    for pred in predictions:
        for i, prob in enumerate(pred):
            label = decoder[str(i)]
            all_predictions_with_labels.append((label, prob))
    print('all_predictions_with_labels\n')
    [print(i, '.\t', labelled_pred[0], '\t', labelled_pred[1]) for i, labelled_pred in enumerate(all_predictions_with_labels)]

    # Get the top prediction index
    predicted_index = np.argmax(predictions, axis=1)
    print('predicted_index', predicted_index)

    # Get the associated label for the top prediction
    predicted_label = [decoder[str(i)] for i in predicted_index]
    print('predicted_label', predicted_label)

    # Get associated confidence score for the top prediction
    top_score = np.max(predictions, axis=1)
    print('top_score', top_score)

    # create a dict
    response = {
        'label' : predicted_label[0],
        'confidence' : f'{round(top_score[0] * 100)}%'
        }
    print('response', response)

    print("✅ decode_pred done \n")
    return response
