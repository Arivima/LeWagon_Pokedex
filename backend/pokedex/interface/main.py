''' package interface '''

import os
import numpy as np
import tensorflow as tf
from colorama import Fore, Style
from tensorflow import random
import tempfile
import matplotlib.pyplot as plt
import tensorflow.keras as keras

## Classification import
from pokedex.params import *
from pokedex.model_logic.data import load_images_from_folders
from pokedex.model_logic.preprocessing import encode_target
from pokedex.model_logic.model_classification import initialize_model_15, initialize_model_150
from pokedex.model_logic.model_classification import compile_model, train_model, evaluate_model
from pokedex.model_logic.registry import load_model_from_local, load_model_from_gcs, save_model, save_results
from pokedex.model_logic.registry import compare_vs_production
from sklearn.model_selection import train_test_split

## GAN import
    ## data process, aug
from pokedex.model_logic.model_GAN import gan_process, DiffAugment, rand_brightness, rand_saturation, rand_contrast, rand_translation, rand_cutout

    ##  model , loss , optimizer
from pokedex.model_logic.model_GAN import initialize_discriminator, initialize_generator
from pokedex.model_logic.model_GAN import generator_loss, discriminator_loss
from pokedex.model_logic.model_GAN import initialize_gen_optimizer, initialize_disc_optimizer

    ## train_step
from pokedex.model_logic.model_GAN import train_step, train_gan


LABELS_TYPE={"0": "bug", "1": "dragon", "2": "electric", "3": "fighting", "4": "fire", "5": "flying", "6": "ghost", "7": "grass", "8": "ground", "9": "ice", "10": "normal", "11": "poison", "12": "psychic", "13": "rock", "14": "water"}
# LABELS_NAME={"0": "Abra", "1": "Aerodactyl", "2": "Alakazam", "3": "Arbok", "4": "Arcanine", "5": "Articuno", "6": "Beedrill", "7": "Bellsprout", "8": "Blastoise", "9": "Bulbasaur", "10": "Butterfree", "11": "Caterpie", "12": "Chansey", "13": "Charizard", "14": "Charmander", "15": "Charmeleon", "16": "Clefable", "17": "Clefairy", "18": "Cloyster", "19": "Cubone", "20": "Dewgong", "21": "Diglett", "22": "Ditto", "23": "Dodrio", "24": "Doduo", "25": "Dragonair", "26": "Dragonite", "27": "Dratini", "28": "Drowzee", "29": "Dugtrio", "30": "Eevee", "31": "Ekans", "32": "Electabuzz", "33": "Electrode", "34": "Exeggcute", "35": "Exeggutor", "36": "Farfetchd", "37": "Fearow", "38": "Flareon", "39": "Gastly", "40": "Gengar", "41": "Geodude", "42": "Gloom", "43": "Golbat", "44": "Goldeen", "45": "Golduck", "46": "Golem", "47": "Graveler", "48": "Grimer", "49": "Growlithe", "50": "Gyarados", "51": "Haunter", "52": "Hitmonchan", "53": "Hitmonlee", "54": "Horsea", "55": "Hypno", "56": "Ivysaur", "57": "Jigglypuff", "58": "Jolteon", "59": "Jynx", "60": "Kabuto", "61": "Kabutops", "62": "Kadabra", "63": "Kakuna", "64": "Kangaskhan", "65": "Kingler", "66": "Koffing", "67": "Krabby", "68": "Lapras", "69": "Lickitung", "70": "Machamp", "71": "Machoke", "72": "Machop", "73": "Magikarp", "74": "Magmar", "75": "Magnemite", "76": "Magneton", "77": "Mankey", "78": "Marowak", "79": "Meowth", "80": "Metapod", "81": "Mew", "82": "Mewtwo", "83": "Moltres", "84": "MrMime", "85": "Muk", "86": "Nidoking", "87": "Nidoqueen", "88": "Nidorina", "89": "Nidorino", "90": "Ninetales", "91": "Oddish", "92": "Omanyte", "93": "Omastar", "94": "Onix", "95": "Paras", "96": "Parasect", "97": "Persian", "98": "Pidgeot", "99": "Pidgeotto", "100": "Pidgey", "101": "Pikachu", "102": "Pinsir", "103": "Poliwag", "104": "Poliwhirl", "105": "Poliwrath", "106": "Ponyta", "107": "Porygon", "108": "Primeape", "109": "Psyduck", "110": "Raichu", "111": "Rapidash", "112": "Raticate", "113": "Rattata", "114": "Rhydon", "115": "Rhyhorn", "116": "Sandshrew", "117": "Sandslash", "118": "Scyther", "119": "Seadra", "120": "Seaking", "121": "Seel", "122": "Shellder", "123": "Slowbro", "124": "Slowpoke", "125": "Snorlax", "126": "Spearow", "127": "Squirtle", "128": "Starmie", "129": "Staryu", "130": "Tangela", "131": "Tauros", "132": "Tentacool", "133": "Tentacruel", "134": "Vaporeon", "135": "Venomoth", "136": "Venonat", "137": "Venusaur", "138": "Victreebel", "139": "Vileplume", "140": "Voltorb", "141": "Vulpix", "142": "Wartortle", "143": "Weedle", "144": "Weepinbell", "145": "Weezing", "146": "Wigglytuff", "147": "Zapdos", "148": "Zubat"}

LABELS_NAME={'43': 'Golbat', '6': 'Beedrill', '11': 'Caterpie', '16': 'Clefable', '110': 'Raichu', '117': 'Sandslash', '80': 'Metapod', '28': 'Drowzee', '91': 'Oddish', '13': 'Charizard', '131': 'Tauros', '106': 'Ponyta', '108': 'Primeape', '126': 'Spearow', '77': 'Mankey', '103': 'Poliwag', '67': 'Krabby', '113': 'Rattata', '133': 'Tentacruel', '47': 'Graveler', '66': 'Koffing', '147': 'Zapdos', '5': 'Articuno', '109': 'Psyduck', '7': 'Bellsprout', '68': 'Lapras', '10': 'Butterfree', '145': 'Weezing', '0': 'Abra', '85': 'Muk', '18': 'Cloyster', '107': 'Porygon', '38': 'Flareon', '57': 'Jigglypuff', '112': 'Raticate', '137': 'Venusaur', '20': 'Dewgong', '54': 'Horsea', '114': 'Rhydon', '92': 'Omanyte', '34': 'Exeggcute', '60': 'Kabuto', '22': 'Ditto', '49': 'Growlithe', '81': 'Mew', '33': 'Electrode', '139': 'Vileplume', '120': 'Seaking', '35': 'Exeggutor', '32': 'Electabuzz', '12': 'Chansey', '74': 'Magmar', '51': 'Haunter', '90': 'Ninetales', '17': 'Clefairy', '50': 'Gyarados', '130': 'Tangela', '78': 'Marowak', '125': 'Snorlax', '87': 'Nidoqueen', '52': 'Hitmonchan', '31': 'Ekans', '116': 'Sandshrew', '58': 'Jolteon', '61': 'Kabutops', '69': 'Lickitung', '99': 'Pidgeotto', '122': 'Shellder', '124': 'Slowpoke', '101': 'Pikachu', '105': 'Poliwrath', '37': 'Fearow', '75': 'Magnemite', '53': 'Hitmonlee', '71': 'Machoke', '104': 'Poliwhirl', '76': 'Magneton', '21': 'Diglett', '136': 'Venonat', '63': 'Kakuna', '30': 'Eevee', '56': 'Ivysaur', '24': 'Doduo', '146': 'Wigglytuff', '44': 'Goldeen', '2': 'Alakazam', '128': 'Starmie', '48': 'Grimer', '102': 'Pinsir', '132': 'Tentacool', '82': 'Mewtwo', '23': 'Dodrio', '64': 'Kangaskhan', '4': 'Arcanine', '27': 'Dratini', '1': 'Aerodactyl', '39': 'Gastly', '41': 'Geodude', '73': 'Magikarp', '148': 'Zubat', '95': 'Paras', '70': 'Machamp', '138': 'Victreebel', '142': 'Wartortle', '93': 'Omastar', '79': 'Meowth', '88': 'Nidorina', '9': 'Bulbasaur', '36': 'Farfetchd', '111': 'Rapidash', '121': 'Seel', '8': 'Blastoise', '135': 'Venomoth', '55': 'Hypno', '45': 'Golduck', '86': 'Nidoking', '134': 'Vaporeon', '26': 'Dragonite', '94': 'Onix', '98': 'Pidgeot', '72': 'Machop', '83': 'Moltres', '118': 'Scyther', '84': 'MrMime', '19': 'Cubone', '40': 'Gengar', '65': 'Kingler', '29': 'Dugtrio', '42': 'Gloom', '96': 'Parasect', '97': 'Persian', '46': 'Golem', '119': 'Seadra', '127': 'Squirtle', '89': 'Nidorino', '14': 'Charmander', '59': 'Jynx', '25': 'Dragonair', '3': 'Arbok', '143': 'Weedle', '100': 'Pidgey', '62': 'Kadabra', '115': 'Rhyhorn', '144': 'Weepinbell', '15': 'Charmeleon', '129': 'Staryu', '140': 'Voltorb', '123': 'Slowbro', '141': 'Vulpix'}


def preprocess(
    classification_type : str = CLASSIFICATION_TYPE,
    sample :str = '300',
    img_new_size : tuple = (128, 128),
    ) -> tuple:
    """
    - load the raw dataset
    - process the raw dataset
    - returns a tuple holding processed features and encoded target
    """
    print(Fore.MAGENTA + "\n⭐️ Starting preprocessing" + Style.RESET_ALL)
    print('classification_type', classification_type, 'sample', sample, 'img_new_size', img_new_size)


    # Load raw data
    if classification_type == '15':
        dataset = load_images_from_folders(
            DATASET_TYPE_PATH,
            sample=sample,
            new_size=img_new_size
            )
    elif classification_type == '150':
        dataset =  load_images_from_folders(
            DATASET_NAME_PATH,
            sample=sample,
            new_size=img_new_size
            )
    else:
        raise ValueError("classification_type should be '15' or '150'")

    # display_images(dataset) # debug
    print('dataset.shape', dataset.shape)

    # define features and target
    X = np.stack(dataset['image'].values)
    # print('X', type(X), X.shape, X[0].shape)
    y = dataset[["label"]].values


    # Process features
    #TODO refactor from load_images_from_folders to preprocess_features
    # X_processed = preprocess_features(X)
    X_processed = X

    # Encode target
    y_cat = encode_target(y)
    print('X_processed', X_processed.shape)
    print('y_cat', y_cat.shape)
    # TODO refactor to save preprocessed data to root registry

    print("✅ preprocess() done \n")
    return (X_processed, y_cat)

def train(
        X_y : tuple,
        classification_type : str = '15', # '15 types' or '150 pokemon'
        test_size=0.25,
        learning_rate=0.01,
        momentum=0.9,
        nesterov=True,
        batch_size = 32,
        monitor='val_loss',
        patience = 2,
        epochs=100,
        validation_data=None, # overrides validation_split
        validation_split=0.3,
        verbose=1
    ) -> float:
    """
    - creates train test subsets
    - Train on the preprocessed dataset

    => Return accuracy as a float
    """
    print(Fore.MAGENTA + "\n⭐️ Starting training " + Style.RESET_ALL)

    # load dataset    # TODO refactor to load preprocessed data from root registry
    X_processed, y_cat = X_y

    # create train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_cat, test_size=test_size, random_state=42)

    print('train test split', X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # Initialize model
    if classification_type == '15':
        model = initialize_model_15(image_size=X_train.shape[1:])
    elif classification_type == '150':
        model = initialize_model_150(image_size=X_train.shape[1:])
    else:
        raise ValueError("classification_type should be '15 types' or '150 pokemon'")

    # Compile model
    model = compile_model(
        model,
        learning_rate=learning_rate,
        momentum=momentum,
        nesterov=nesterov
        )

    # Train model
    model, history = train_model(
        model=model,
        X= X_train,
        y= y_train,
        batch_size=batch_size,
        monitor=monitor,
        patience=patience,
        epochs=epochs,
        validation_data=validation_data, # overrides validation_split
        validation_split=validation_split,
        verbose=verbose
    )

    min_val_loss_index = np.argmin(history.history['val_loss'])
    metrics = {
        'accuracy':history.history['accuracy'][min_val_loss_index],
        'loss':history.history['val_loss'][min_val_loss_index],
    }

    # print(Fore.PINK +"Hyperparamètres du modèle :" + Style.RESET_ALL)
    # for key, value in model.get_config().items():
    #     print(f"{key} : {value}")

    total_parameters = sum(p.shape.num_elements() for layer in model.layers for p in layer.trainable_variables)
    print(Fore.GREEN + "Total parameters:", "{:,}".format(total_parameters), Style.RESET_ALL)

    # save results and model
    params = dict(
        context="train",
        model=CLASSIFICATION_TYPE,
        nb_images=len(X_train),
        img_size=X_train.shape[1:3]
        )
    # Save results on the hard drive using pokedex.model_logic.registry
    save_results(params=params, metrics=metrics, context='train')

    # Save model weight on the hard drive
    save_model(model=model, context='train', metrics=metrics)

    print("✅ train() done \n")
    return metrics['accuracy'], X_test, y_test


def evaluate(
        X_test,
        y_test,
        batch_size : int =32,
        verbose : bool = 1,
    ) -> float:
    '''
    Evaluate the performance of the latest production model on processed data
    '''
    print(Fore.MAGENTA + "\n⭐️ Use case: evaluate" + Style.RESET_ALL)

    # load latest model
    model = load_model_from_local(stage="Staging", model_type=CLASSIFICATION_TYPE)
    assert model is not None

    # evaluate model
    metrics_dict = evaluate_model(
        model=model,
        X=X_test,
        y=y_test,
        batch_size=batch_size,
        verbose = verbose
        )

    # save results
    params = dict(
        context="evaluate", # Package behavior
        model=CLASSIFICATION_TYPE,
        nb_images=len(X_test),
        img_size=X_test.shape[1:3]
    )

    save_results(params=params, metrics=metrics_dict, context='evaluate')

    print("✅ evaluate() done \n")
    compare_vs_production()

    return metrics_dict["accuracy"]

def pred_both(test=False) -> np.ndarray:
    ''' predicts both 15 and 150 '''
    final_dict_15, df_15 = pred('15', test=test)
    final_dict_150, df_150 = pred('150', test=test)
    sorted_dict_15 = dict(sorted(final_dict_15.items(), key=lambda item: item[1], reverse=True))
    sorted_dict_150 = dict(sorted(final_dict_150.items(), key=lambda item: item[1], reverse=True))
    print(sorted_dict_15)
    print(sorted_dict_150)

    print(df_15.shape, df_15.columns) # (92, 3)
    print(df_150.shape, df_150.columns) # (2, 3)
    merged_df = pd.merge(df_15, df_150, on='i', how='inner', suffixes=('_15', '_150'))

    print(merged_df.shape, merged_df.columns) #(1, 5)
    return final_dict_15, final_dict_150


import pandas as pd

def pred(model_type : str = '15', test=False) -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """
    print(f"\n⭐️ Use case: predict {model_type}")
    if test is False:
        images_path = os.path.join('..', 'all_prediction_images')
        dataset = load_images_from_folders(images_path)
    else:
        if str(model_type) == '15':
            dataset = load_images_from_folders(DATASET_TYPE_PATH)
        elif str(model_type) == '150':
            dataset =  load_images_from_folders(DATASET_NAME_PATH)
        else:
            raise ValueError("model_type should be '15' or '150'")

    # display_images(dataset) # debug
    print('dataset.shape', dataset.shape)

    # define features and target
    X = np.stack(dataset['image'].values)
    print('X', type(X), X.shape, X[0].shape)
    y = dataset[["label"]].values
    print(y.shape)
    y_cat = encode_target(y)
    print("✅ images loaded and processed\n")

    # load model in production
    model = load_model_from_gcs(model_type=model_type)
    print('model', model_type, model.summary())
    print("✅ model loaded\n")

    # Evaluate model
    # metrics_dict = evaluate_model(model=model, X=X, y=y_cat)
    # print('metrics evaluate', metrics_dict)

    # make prediction
    y_pred = model.predict(X)
    # print('y_pred', y_pred)

    # decode prediction
    labels_dict = LABELS_TYPE if (str(model_type) == '15') else LABELS_NAME
    # print('labels_dict', labels_dict)

    predicted_index = np.argmax(y_pred, axis=1)
    # print('predicted_index', predicted_index)

    predicted_labels = [labels_dict[str(i)] for i in predicted_index]
    # print('predicted_labels', predicted_labels)

    # all_predictions_with_labels = []
    # for pred in y_pred:
    #     for i, prob in enumerate(pred):
    #         label = labels_dict[str(i)]
    #         all_predictions_with_labels.append((label, prob))
    # all_predictions_with_labels = [{labels_dict[str(i)] : prob } for (i, prob) in enumerate(pred) for pred in y_pred]
    # print('all_predictions_with_labels', all_predictions_with_labels)

    # Get the confidence scores for each prediction
    top_score = np.max(y_pred, axis=1)

    # Create a dictionary with predicted labels as keys and confidence scores as values
    predicted_labels_confidence_dict = [[k, v] for k, v in zip(predicted_labels, top_score)]
    # print('predicted_labels_confidence_dict', predicted_labels_confidence_dict)


    final_dict = {}
    for entry in predicted_labels_confidence_dict:
        if entry[0] in final_dict.keys():
            final_dict[entry[0]] += 1
        else:
            final_dict[entry[0]] = 1

    # print(final_dict)
    # good_predictions = pd.DataFrame(columns=['i', 'y'])
    # print(type(X), type(y), type(predicted_labels))
    # <class 'numpy.ndarray'> <class 'numpy.ndarray'> <class 'list'>
    # print(X.shape, y.shape, len(predicted_labels))
    # (300, 128, 128, 3) (300, 1) 300

    predicted_labels = np.array(predicted_labels)
    y = y.flatten()

    # get a np array with all images for which the prediction is the same as label
    good_predictions = []
    for i, _ in enumerate(predicted_labels):
        if y[i] == predicted_labels[i]:
            # print(i, y[i], predicted_labels[i])
            good_predictions.append([i, y[i]])
            # good_predictions = pd.concat([good_predictions, pd.DataFrame.from_dict({'i': i, 'y': y[i]})], ignore_index=True)

    df_good_pred = pd.DataFrame(good_predictions, columns=['i', 'label'])
    print(df_good_pred.shape)

    print("✅ pred() done \n")
    return final_dict, df_good_pred


def generate():
    """
    Generate new images of Pokemons
    """
    print("\n⭐️ Use case : generate")
    # load model in production
    model = load_model_from_gcs(model_type="GAN")
    print('model', "GAN", model.summary())
    print("✅ model loaded\n")

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
        return None




def main():
    # try :
    print(Fore.MAGENTA + "\n ⭐️ ⭐️ ⭐️ Starting Pokedex ⭐️ ⭐️ ⭐️ " + Style.RESET_ALL)
    # Here only place to define hyperparams

    if CLASSIFICATION_TYPE == '15':
        settings = dict(
            classification_type = CLASSIFICATION_TYPE, # '15 types' or '150 pokemon'
            sampled_dataset = SAMPLED_DATASET,
            img_new_size = (128, 128),
            test_size=0.2,
            learning_rate=0.01,
            momentum=0.9,
            nesterov=True,
            batch_size = 64,
            monitor='val_loss',
            patience = 5,
            epochs=200,
            validation_split=0.3,
            verbose=0
        )
    else :
        settings = dict(
            classification_type = CLASSIFICATION_TYPE, # '15 types' or '150 pokemon'
            sampled_dataset = SAMPLED_DATASET,
            img_new_size = (128, 128),
            test_size=0.2,
            learning_rate=0.01,
            momentum=0.9,
            nesterov=True,
            batch_size = 64,
            monitor='val_loss',
            patience = 5,
            epochs=200,
            validation_split=0.3,
            verbose=0
        )

    print('Running with the following settings :')
    print(settings)
    print()

    dataset_processed = preprocess(
        classification_type = settings['classification_type'],
        sample = settings['sampled_dataset'],
        img_new_size = settings['img_new_size']
        )

    best_accuracy, X_test, y_test = train(
        dataset_processed,
        classification_type=settings['classification_type'],
        test_size=settings['test_size'],
        learning_rate=settings['learning_rate'],
        momentum=settings['momentum'],
        nesterov=settings['nesterov'],
        batch_size = settings['batch_size'],
        monitor=settings['monitor'],
        patience = settings['patience'],
        epochs=settings['epochs'],
        validation_split=settings['validation_split'],
        verbose=settings['verbose']
    )
    accuracy = evaluate(
        X_test=X_test,
        y_test=y_test,
        batch_size = settings['batch_size'],
        verbose=settings['verbose']
    )
    print(Fore.MAGENTA + "\n ⭐️ ⭐️ ⭐️ Closing Pokedex ⭐️ ⭐️ ⭐️ " + Style.RESET_ALL)

def main_gan():
    # try :
    print(Fore.MAGENTA + "\n ⭐️ ⭐️ ⭐️ Starting MagiGAN training ⭐️ ⭐️ ⭐️ " + Style.RESET_ALL)
    # Here only place to define hyperparams
    #VARIABLES :

    AUGMENT_FNS = {
        'color': [rand_brightness, rand_saturation, rand_contrast],
        'translation': [rand_translation],
        'cutout': [rand_cutout],
    }


    tf.keras.utils.set_random_seed(7)
    batch_size = 32
    latent_dim = 100
    epochs = 1500
    path = DATASET_NAME_PATH
    trained_models_folder = os.path.expanduser(os.path.join('~', '.lewagon', 'pokedex', 'gan','output_models'))
    generated_images_folder = os.path.expanduser(os.path.join('~', '.lewagon', 'pokedex', 'gan','output_images'))

    print(f"Preprocessing the data from {path}. \n")
    dataset = gan_process(path=path,batch_size=batch_size)
    print("✅ preprocess() done \n")
    print("✅ initialize_discriminator and initialize_generator done \n")
    print(f" ♾️ Begining training on {epochs} epochs ♾️\n")
    seed = tf.random.normal([25, latent_dim])
    train_gan(dataset,epochs,trained_models_folder,generated_images_folder,seed,batch_size,latent_dim,AUGMENT_FNS)
    print(Fore.MAGENTA + "\n ⭐️ ⭐️ ⭐️ MagiGAN training is over ⭐️ ⭐️ ⭐️ " + Style.RESET_ALL)




if __name__ == '__main__':
    if CLASSIFICATION_TYPE == 'GAN':
        main_gan()
    else:
        main()
