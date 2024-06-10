''' package interface '''

import os
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from colorama import Fore, Style

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




def preprocess(
    classification_type : str = CLASSIFICATION_TYPE, # '15 types' or '150 pokemon'
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
        raise ValueError("classification_type should be '15 types' or '150 pokemon'")

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



def pred(model_type : str = '15') -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """
    print("\n⭐️ Use case: predict")
    # Fetch images to predict
    images_path = os.path.join('..', 'all_prediction_images')
    img_new_size = (128, 128)
    images = load_images_from_folders(
        images_path,
        sample='all',
        new_size=img_new_size
        )
    print('images.shape', images.shape)

    # define features
    X = np.stack(images['image'].values)
    print('X', X.shape)
    print("✅ images loaded and processed\n")

    # load model in production
    model = load_model_from_gcs(model_type=model_type)
    print("✅ model loaded\n")

    # make prediction
    y_pred = model.predict(X)

    # decode prediction
    label_decoder = LABELS_TYPE if (str(model_type) == '15') else LABELS_NAME
    labels_type_dict  = json.loads(label_decoder)
    predicted_index = np.argmax(y_pred, axis=1)
    predicted_labels = [labels_type_dict[str(i)] for i in predicted_index]

    # Get the confidence scores for each prediction
    confidence_scores = np.max(y_pred, axis=1)

    # Create a dictionary with predicted labels as keys and confidence scores as values
    predicted_labels_confidence_dict = [[k, v] for k, v in zip(predicted_labels, confidence_scores)]

    print(predicted_labels_confidence_dict)
    print("✅ pred() done \n")
    return predicted_labels






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
    # TODO makefile
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
