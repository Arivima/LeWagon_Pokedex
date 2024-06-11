''' API ENDPOINTS '''
# IMPORTS
from fastapi import FastAPI, UploadFile, File, HTTPException
from pokedex.model_logic.registry import load_model_from_gcs
from pokedex.params import *

import numpy as np
import cv2
from tensorflow import keras

LABELS_TYPE={"0": "bug", "1": "dragon", "2": "electric", "3": "fighting", "4": "fire", "5": "flying", "6": "ghost", "7": "grass", "8": "ground", "9": "ice", "10": "normal", "11": "poison", "12": "psychic", "13": "rock", "14": "water"}
LABELS_NAME={"0": "Abra", "1": "Aerodactyl", "2": "Alakazam", "3": "Arbok", "4": "Arcanine", "5": "Articuno", "6": "Beedrill", "7": "Bellsprout", "8": "Blastoise", "9": "Bulbasaur", "10": "Butterfree", "11": "Caterpie", "12": "Chansey", "13": "Charizard", "14": "Charmander", "15": "Charmeleon", "16": "Clefable", "17": "Clefairy", "18": "Cloyster", "19": "Cubone", "20": "Dewgong", "21": "Diglett", "22": "Ditto", "23": "Dodrio", "24": "Doduo", "25": "Dragonair", "26": "Dragonite", "27": "Dratini", "28": "Drowzee", "29": "Dugtrio", "30": "Eevee", "31": "Ekans", "32": "Electabuzz", "33": "Electrode", "34": "Exeggcute", "35": "Exeggutor", "36": "Farfetchd", "37": "Fearow", "38": "Flareon", "39": "Gastly", "40": "Gengar", "41": "Geodude", "42": "Gloom", "43": "Golbat", "44": "Goldeen", "45": "Golduck", "46": "Golem", "47": "Graveler", "48": "Grimer", "49": "Growlithe", "50": "Gyarados", "51": "Haunter", "52": "Hitmonchan", "53": "Hitmonlee", "54": "Horsea", "55": "Hypno", "56": "Ivysaur", "57": "Jigglypuff", "58": "Jolteon", "59": "Jynx", "60": "Kabuto", "61": "Kabutops", "62": "Kadabra", "63": "Kakuna", "64": "Kangaskhan", "65": "Kingler", "66": "Koffing", "67": "Krabby", "68": "Lapras", "69": "Lickitung", "70": "Machamp", "71": "Machoke", "72": "Machop", "73": "Magikarp", "74": "Magmar", "75": "Magnemite", "76": "Magneton", "77": "Mankey", "78": "Marowak", "79": "Meowth", "80": "Metapod", "81": "Mew", "82": "Mewtwo", "83": "Moltres", "84": "MrMime", "85": "Muk", "86": "Nidoking", "87": "Nidoqueen", "88": "Nidorina", "89": "Nidorino", "90": "Ninetales", "91": "Oddish", "92": "Omanyte", "93": "Omastar", "94": "Onix", "95": "Paras", "96": "Parasect", "97": "Persian", "98": "Pidgeot", "99": "Pidgeotto", "100": "Pidgey", "101": "Pikachu", "102": "Pinsir", "103": "Poliwag", "104": "Poliwhirl", "105": "Poliwrath", "106": "Ponyta", "107": "Porygon", "108": "Primeape", "109": "Psyduck", "110": "Raichu", "111": "Rapidash", "112": "Raticate", "113": "Rattata", "114": "Rhydon", "115": "Rhyhorn", "116": "Sandshrew", "117": "Sandslash", "118": "Scyther", "119": "Seadra", "120": "Seaking", "121": "Seel", "122": "Shellder", "123": "Slowbro", "124": "Slowpoke", "125": "Snorlax", "126": "Spearow", "127": "Squirtle", "128": "Starmie", "129": "Staryu", "130": "Tangela", "131": "Tauros", "132": "Tentacool", "133": "Tentacruel", "134": "Vaporeon", "135": "Venomoth", "136": "Venonat", "137": "Venusaur", "138": "Victreebel", "139": "Vileplume", "140": "Voltorb", "141": "Vulpix", "142": "Wartortle", "143": "Weedle", "144": "Weepinbell", "145": "Weezing", "146": "Wigglytuff", "147": "Zapdos", "148": "Zubat"}


# Starting API
app = FastAPI()

# MODEL UPLOAD
app.state.model_15 = load_model_from_gcs(model_type='15')
if app.state.model_15 :
    print("✅ model 15 loaded\n")
else:
    print("! model 15 NOT loaded\n")

app.state.model_150 = load_model_from_gcs(model_type='150')
if app.state.model_150 :
    print("✅ model 150 loaded\n")
else:
    print("! model 150 NOT loaded\n")

app.state.model_GAN = load_model_from_gcs(model_type='GAN')
if app.state.model_GAN :
    print("✅ model GAN loaded\n")
else:
    print("! model GAN NOT loaded\n")


# ENDPOINTS
@app.get("/")
def index():
    ''' test for root'''
    return {"status": "OK!"}


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
    if model is None:
        return {'class': "pokedex_error"}

    predictions = model.predict(img_array)

    # decode prediction
    predicted_index = np.argmax(predictions, axis=1)
    predicted_labels = [LABELS_TYPE[str(i)] for i in predicted_index]
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
    if model is None:
        return {'class': "pokedex_error"}

    predictions = model.predict(img_array)

    # decode prediction
    predicted_index = np.argmax(predictions, axis=1)
    predicted_labels = [LABELS_NAME[str(i)] for i in predicted_index]
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

    model = app.state.model_GAN
    if model is None:
        return {'new_pokemon': "pokedex_error"}

    return {'new_pokemon': "pokedex_error"}
