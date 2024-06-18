''' API ENDPOINTS '''
# IMPORTS
from pathlib import Path
import tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import os

from tensorflow import random
import tensorflow.keras as keras

import matplotlib.pyplot as plt
import numpy as np


from pokedex.api.utils import load_model, process_img, decode_pred


# Global variables
LABELS_TYPE={"0": "bug", "1": "dragon", "2": "electric", "3": "fighting", "4": "fire", "5": "flying", "6": "ghost", "7": "grass", "8": "ground", "9": "ice", "10": "normal", "11": "poison", "12": "psychic", "13": "rock", "14": "water"}
# dict dataset raw
# LABELS_NAME={'0': 'Abra', '1': 'Aerodactyl', '2': 'Alakazam', '3': 'Alolan Sandslash', '4': 'Arbok', '5': 'Arcanine', '6': 'Articuno', '7': 'Beedrill', '8': 'Bellsprout', '9': 'Blastoise', '10': 'Bulbasaur', '11': 'Butterfree', '12': 'Caterpie', '13': 'Chansey', '14': 'Charizard', '15': 'Charmander', '16': 'Charmeleon', '17': 'Clefable', '18': 'Clefairy', '19': 'Cloyster', '20': 'Cubone', '21': 'Dewgong', '22': 'Diglett', '23': 'Ditto', '24': 'Dodrio', '25': 'Doduo', '26': 'Dragonair', '27': 'Dragonite', '28': 'Dratini', '29': 'Drowzee', '30': 'Dugtrio', '31': 'Eevee', '32': 'Ekans', '33': 'Electabuzz', '34': 'Electrode', '35': 'Exeggcute', '36': 'Exeggutor', '37': 'Farfetchd', '38': 'Fearow', '39': 'Flareon', '40': 'Gastly', '41': 'Gengar', '42': 'Geodude', '43': 'Gloom', '44': 'Golbat', '45': 'Goldeen', '46': 'Golduck', '47': 'Golem', '48': 'Graveler', '49': 'Grimer', '50': 'Growlithe', '51': 'Gyarados', '52': 'Haunter', '53': 'Hitmonchan', '54': 'Hitmonlee', '55': 'Horsea', '56': 'Hypno', '57': 'Ivysaur', '58': 'Jigglypuff', '59': 'Jolteon', '60': 'Jynx', '61': 'Kabuto', '62': 'Kabutops', '63': 'Kadabra', '64': 'Kakuna', '65': 'Kangaskhan', '66': 'Kingler', '67': 'Koffing', '68': 'Krabby', '69': 'Lapras', '70': 'Lickitung', '71': 'Machamp', '72': 'Machoke', '73': 'Machop', '74': 'Magikarp', '75': 'Magmar', '76': 'Magnemite', '77': 'Magneton', '78': 'Mankey', '79': 'Marowak', '80': 'Meowth', '81': 'Metapod', '82': 'Mew', '83': 'Mewtwo', '84': 'Moltres', '85': 'MrMime', '86': 'Muk', '87': 'Nidoking', '88': 'Nidoqueen', '89': 'Nidorina', '90': 'Nidorino', '91': 'Ninetales', '92': 'Oddish', '93': 'Omanyte', '94': 'Omastar', '95': 'Onix', '96': 'Paras', '97': 'Parasect', '98': 'Persian', '99': 'Pidgeot', '100': 'Pidgeotto', '101': 'Pidgey', '102': 'Pikachu', '103': 'Pinsir', '104': 'Poliwag', '105': 'Poliwhirl', '106': 'Poliwrath', '107': 'Ponyta', '108': 'Porygon', '109': 'Primeape', '110': 'Psyduck', '111': 'Raichu', '112': 'Rapidash', '113': 'Raticate', '114': 'Rattata', '115': 'Rhydon', '116': 'Rhyhorn', '117': 'Sandshrew', '118': 'Sandslash', '119': 'Scyther', '120': 'Seadra', '121': 'Seaking', '122': 'Seel', '123': 'Shellder', '124': 'Slowbro', '125': 'Slowpoke', '126': 'Snorlax', '127': 'Spearow', '128': 'Squirtle', '129': 'Starmie', '130': 'Staryu', '131': 'Tangela', '132': 'Tauros', '133': 'Tentacool', '134': 'Tentacruel', '135': 'Vaporeon', '136': 'Venomoth', '137': 'Venonat', '138': 'Venusaur', '139': 'Victreebel', '140': 'Vileplume', '141': 'Voltorb', '142': 'Vulpix', '143': 'Wartortle', '144': 'Weedle', '145': 'Weepinbell', '146': 'Weezing', '147': 'Wigglytuff', '148': 'Zapdos', '149': 'Zubat'}
# dict dataset clean
# LABELS_NAME={"0": "Abra", "1": "Aerodactyl", "2": "Alakazam", "3": "Arbok", "4": "Arcanine", "5": "Articuno", "6": "Beedrill", "7": "Bellsprout", "8": "Blastoise", "9": "Bulbasaur", "10": "Butterfree", "11": "Caterpie", "12": "Chansey", "13": "Charizard", "14": "Charmander", "15": "Charmeleon", "16": "Clefable", "17": "Clefairy", "18": "Cloyster", "19": "Cubone", "20": "Dewgong", "21": "Diglett", "22": "Ditto", "23": "Dodrio", "24": "Doduo", "25": "Dragonair", "26": "Dragonite", "27": "Dratini", "28": "Drowzee", "29": "Dugtrio", "30": "Eevee", "31": "Ekans", "32": "Electabuzz", "33": "Electrode", "34": "Exeggcute", "35": "Exeggutor", "36": "Farfetchd", "37": "Fearow", "38": "Flareon", "39": "Gastly", "40": "Gengar", "41": "Geodude", "42": "Gloom", "43": "Golbat", "44": "Goldeen", "45": "Golduck", "46": "Golem", "47": "Graveler", "48": "Grimer", "49": "Growlithe", "50": "Gyarados", "51": "Haunter", "52": "Hitmonchan", "53": "Hitmonlee", "54": "Horsea", "55": "Hypno", "56": "Ivysaur", "57": "Jigglypuff", "58": "Jolteon", "59": "Jynx", "60": "Kabuto", "61": "Kabutops", "62": "Kadabra", "63": "Kakuna", "64": "Kangaskhan", "65": "Kingler", "66": "Koffing", "67": "Krabby", "68": "Lapras", "69": "Lickitung", "70": "Machamp", "71": "Machoke", "72": "Machop", "73": "Magikarp", "74": "Magmar", "75": "Magnemite", "76": "Magneton", "77": "Mankey", "78": "Marowak", "79": "Meowth", "80": "Metapod", "81": "Mew", "82": "Mewtwo", "83": "Moltres", "84": "MrMime", "85": "Muk", "86": "Nidoking", "87": "Nidoqueen", "88": "Nidorina", "89": "Nidorino", "90": "Ninetales", "91": "Oddish", "92": "Omanyte", "93": "Omastar", "94": "Onix", "95": "Paras", "96": "Parasect", "97": "Persian", "98": "Pidgeot", "99": "Pidgeotto", "100": "Pidgey", "101": "Pikachu", "102": "Pinsir", "103": "Poliwag", "104": "Poliwhirl", "105": "Poliwrath", "106": "Ponyta", "107": "Porygon", "108": "Primeape", "109": "Psyduck", "110": "Raichu", "111": "Rapidash", "112": "Raticate", "113": "Rattata", "114": "Rhydon", "115": "Rhyhorn", "116": "Sandshrew", "117": "Sandslash", "118": "Scyther", "119": "Seadra", "120": "Seaking", "121": "Seel", "122": "Shellder", "123": "Slowbro", "124": "Slowpoke", "125": "Snorlax", "126": "Spearow", "127": "Squirtle", "128": "Starmie", "129": "Staryu", "130": "Tangela", "131": "Tauros", "132": "Tentacool", "133": "Tentacruel", "134": "Vaporeon", "135": "Venomoth", "136": "Venonat", "137": "Venusaur", "138": "Victreebel", "139": "Vileplume", "140": "Voltorb", "141": "Vulpix", "142": "Wartortle", "143": "Weedle", "144": "Weepinbell", "145": "Weezing", "146": "Wigglytuff", "147": "Zapdos", "148": "Zubat"}
# dict PH dataset clean
LABELS_NAME={'43': 'Golbat', '6': 'Beedrill', '11': 'Caterpie', '16': 'Clefable', '110': 'Raichu', '117': 'Sandslash', '80': 'Metapod', '28': 'Drowzee', '91': 'Oddish', '13': 'Charizard', '131': 'Tauros', '106': 'Ponyta', '108': 'Primeape', '126': 'Spearow', '77': 'Mankey', '103': 'Poliwag', '67': 'Krabby', '113': 'Rattata', '133': 'Tentacruel', '47': 'Graveler', '66': 'Koffing', '147': 'Zapdos', '5': 'Articuno', '109': 'Psyduck', '7': 'Bellsprout', '68': 'Lapras', '10': 'Butterfree', '145': 'Weezing', '0': 'Abra', '85': 'Muk', '18': 'Cloyster', '107': 'Porygon', '38': 'Flareon', '57': 'Jigglypuff', '112': 'Raticate', '137': 'Venusaur', '20': 'Dewgong', '54': 'Horsea', '114': 'Rhydon', '92': 'Omanyte', '34': 'Exeggcute', '60': 'Kabuto', '22': 'Ditto', '49': 'Growlithe', '81': 'Mew', '33': 'Electrode', '139': 'Vileplume', '120': 'Seaking', '35': 'Exeggutor', '32': 'Electabuzz', '12': 'Chansey', '74': 'Magmar', '51': 'Haunter', '90': 'Ninetales', '17': 'Clefairy', '50': 'Gyarados', '130': 'Tangela', '78': 'Marowak', '125': 'Snorlax', '87': 'Nidoqueen', '52': 'Hitmonchan', '31': 'Ekans', '116': 'Sandshrew', '58': 'Jolteon', '61': 'Kabutops', '69': 'Lickitung', '99': 'Pidgeotto', '122': 'Shellder', '124': 'Slowpoke', '101': 'Pikachu', '105': 'Poliwrath', '37': 'Fearow', '75': 'Magnemite', '53': 'Hitmonlee', '71': 'Machoke', '104': 'Poliwhirl', '76': 'Magneton', '21': 'Diglett', '136': 'Venonat', '63': 'Kakuna', '30': 'Eevee', '56': 'Ivysaur', '24': 'Doduo', '146': 'Wigglytuff', '44': 'Goldeen', '2': 'Alakazam', '128': 'Starmie', '48': 'Grimer', '102': 'Pinsir', '132': 'Tentacool', '82': 'Mewtwo', '23': 'Dodrio', '64': 'Kangaskhan', '4': 'Arcanine', '27': 'Dratini', '1': 'Aerodactyl', '39': 'Gastly', '41': 'Geodude', '73': 'Magikarp', '148': 'Zubat', '95': 'Paras', '70': 'Machamp', '138': 'Victreebel', '142': 'Wartortle', '93': 'Omastar', '79': 'Meowth', '88': 'Nidorina', '9': 'Bulbasaur', '36': 'Farfetchd', '111': 'Rapidash', '121': 'Seel', '8': 'Blastoise', '135': 'Venomoth', '55': 'Hypno', '45': 'Golduck', '86': 'Nidoking', '134': 'Vaporeon', '26': 'Dragonite', '94': 'Onix', '98': 'Pidgeot', '72': 'Machop', '83': 'Moltres', '118': 'Scyther', '84': 'MrMime', '19': 'Cubone', '40': 'Gengar', '65': 'Kingler', '29': 'Dugtrio', '42': 'Gloom', '96': 'Parasect', '97': 'Persian', '46': 'Golem', '119': 'Seadra', '127': 'Squirtle', '89': 'Nidorino', '14': 'Charmander', '59': 'Jynx', '25': 'Dragonair', '3': 'Arbok', '143': 'Weedle', '100': 'Pidgey', '62': 'Kadabra', '115': 'Rhyhorn', '144': 'Weepinbell', '15': 'Charmeleon', '129': 'Staryu', '140': 'Voltorb', '123': 'Slowbro', '141': 'Vulpix'}


# Starting API
app = FastAPI()
app.state.model_15 = app.state.model_150 = app.state.model_GAN = None

# MODEL UPLOAD
app.state.model_15 = load_model(model_type='15', app_state=app.state)
app.state.model_150 = load_model(model_type='150', app_state=app.state)
app.state.model_GAN = load_model(model_type='GAN', app_state=app.state)


# ENDPOINTS
@app.get("/")
def index():
    ''' test for root'''
    return {"status": "OK!"}

@app.get("/force_reload")
def force_reload():
    ''' forces the reload of models from the gcs buckets'''
    print("\n⭐️ API call : force_reload")

    app.state.model_15 = load_model(model_type='15', app_state=app.state, force_reload=True)
    app.state.model_150 = load_model(model_type='150', app_state=app.state, force_reload=True)
    app.state.model_GAN = load_model(model_type='GAN', app_state=app.state, force_reload=True)

    return {"status": "OK!"}


@app.post('/predict')
async def predict(img: UploadFile=File(...)):
    '''
    returns the predicted type and pokemon name of the image given as input
    '''
    print("\n⭐️ API call : predict")
    # Receive and decode the image
    img_array = await process_img(image=img)

    # Load model
    model_15 = load_model(model_type='15', app_state=app.state, must_raise=True)
    model_150 = load_model(model_type='150', app_state=app.state, must_raise=True)

    # Make predictions
    pokemon_name = model_150.predict(img_array)
    pokemon_type = model_15.predict(img_array)

    print('prediction proba name', pokemon_name)
    print('prediction proba type', pokemon_type)

    # decode prediction
    name_index = np.argmax(pokemon_name, axis=1)
    name_label = [LABELS_NAME[str(i)] for i in name_index]

    type_index = np.argmax(pokemon_type, axis=1)
    type_label = [LABELS_TYPE[str(i)] for i in type_index]

    # create a dict
    response = {
        'Pokemon' : name_label[0],
        'type' : type_label[0]
        }

    print('response', response)
    print("✅ predict_type done \n")
    return response


@app.get("/generate")
async def generate():
    """
    Generate new images of Pokemons
    """
    print("\n⭐️ API call : generate")
    # Load model
    model = load_model(model_type='GAN', app_state=app.state, must_raise=True)

    # generating images
    latent_dim = 100
    seed = random.normal([25, latent_dim])
    generated_images = model(seed)

    # rescaling images to [0, 255]
    generated_images = (generated_images * 255).numpy()

    # creating a temporary file to save the image
    output_dir = os.path.join(os.getcwd(), '..', 'output_gan')
    os.makedirs(output_dir, exist_ok=True)
    print(output_dir)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpeg") as tmpfile:
        print('Saving image in tmpfile')
        plt.figure(figsize=(10, 10))
        dim = (5, 5)
        for i in range(generated_images.shape[0]):
            plt.subplot(dim[0], dim[1], i+1)
            img = keras.preprocessing.image.array_to_img(generated_images[i])
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout()
        plt.savefig(tmpfile.name)
        output_file_path = os.path.join(output_dir, os.path.basename(tmpfile.name))
        print(output_file_path)
        plt.savefig(output_file_path)
        print(f"Gan generated image saved at {output_file_path}")
        plt.close()

        # return the path of the temporary file as a response
        return FileResponse(tmpfile.name)


@app.post('/predict_type')
async def predict_type(img: UploadFile=File(...)):
    '''
    returns the predicted type of the image given as input
    '''
    print("\n⭐️ API call : predict_type")
    # decode image
    img_array = await process_img(image=img)

    # Load model
    model = load_model(model_type='15', app_state=app.state, must_raise=True)

    # Make prediction
    predictions = model.predict(img_array)

    # decode prediction
    response = decode_pred(predictions=predictions, decoder=LABELS_TYPE)
    print("✅ predict_type done \n")
    return response


@app.post('/predict_name')
async def predict_name(img: UploadFile=File(...)):
    '''
    returns the predicted pokemon name of the image given as input
    '''
    print("\n⭐️ API call : predict_name")
    # decode image
    img_array = await process_img(image=img)

    # Load model
    model = load_model(model_type='150', app_state=app.state, must_raise=True)

    # Make prediction
    predictions = model.predict(img_array)

    # decode prediction
    response = decode_pred(predictions=predictions, decoder=LABELS_NAME)
    print("✅ predict_type done \n")
    return response
