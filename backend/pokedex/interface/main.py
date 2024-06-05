''' package interface '''

import os
import pandas as pd
import numpy as np

from pokedex.params import *
from pokedex.model_logic.data import load_images_from_folders, display_images
from pokedex.model_logic.preprocessing import preprocess_features, encode_target
from pokedex.model_logic.model_classification import initialize_model_15, initialize_model_150
from pokedex.model_logic.model_classification import compile_model, train_model, evaluate_model
from pokedex.model_logic.plotting import plot_loss_accuracy
from pokedex.model_logic.registry import load_model, save_model, save_results
from sklearn.model_selection import train_test_split

from colorama import Fore, Style, init



def preprocess(
    classification_type : str = '15', # '15 types' or '150 pokemon'
    sample :str = '50', #TODO set to all by default when debug done
    img_new_size : tuple = (128, 128),
    ) -> tuple:
    """
    - load the raw dataset
    - process the raw dataset
    - returns a tuple holding processed features and encoded target
    """

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

    # define features and target
    X = dataset[["image"]].to_numpy()
    y = dataset[["label"]].to_numpy().ravel()

    print('X', X.shape, 'y', y.shape)

    # Process features
    #TODO refactor from load_images_from_folders to preprocess_features
    # X_processed = preprocess_features(X)
    X_processed = X

    # Encode target
    y_cat = encode_target(y)

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

    # load dataset    # TODO refactor to load preprocessed data from root registry
    X_processed, y_cat = X_y

    # create train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_cat, test_size=test_size, random_state=42)

    # Initialize model
    model = load_model()
    if model is None:
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
    best_accuracy = np.max(history.history['accuracy'])

    # plot_loss_accuracy(history)    #TODO delete when refactor

    # save results and model
    params = dict(
        context="train",
        model=CLASSIFICATION_TYPE,
        nb_images=len(X_train),
        learning_rate=learning_rate,
        momentum=momentum,
        nesterov=nesterov,
        batch_size=batch_size,
        monitor=monitor,
        patience=patience,
        epochs=epochs,
        validation_data=validation_data, # overrides validation_split
        validation_split=validation_split,
        )
    # Save results on the hard drive using pokedex.model_logic.registry + MLFlow
    save_results(params=params, metrics=dict(best_accuracy=best_accuracy))

    # Save model weight on the hard drive + MLFlow
    save_model(model=model)

    # # The latest model should be moved to staging
    # if MODEL_TARGET == 'mlflow':
    #     mlflow_transition_model(current_stage="None", new_stage="Staging")

    print("✅ train() done \n")
    return best_accuracy, X_test, y_test


def evaluate(
        X_test,
        y_test,
        batch_size : int =32,
        verbose : bool = 1,
        stage: str = "Production"
    ) -> float:
    '''
    Evaluate the performance of the latest production model on processed data
    '''
    print(Fore.MAGENTA + "\n⭐️ Use case: evaluate" + Style.RESET_ALL)

    # load latest model
    model = load_model(stage=stage)
    assert model is not None

    # evaluate model
    metrics_dict = evaluate_model(
        model=model,
        X=X_test,
        y=y_test,
        batch_size=batch_size,
        verbose = verbose,
        )

    # save results
    params = dict(
        context="evaluate", # Package behavior
        model=CLASSIFICATION_TYPE,
        nb_images=len(X_test),
        batch_size=batch_size,
    )
    save_results(params=params, metrics=metrics_dict)

    print("✅ evaluate() done \n")
    return metrics_dict["accuracy"]



# def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
#     """
#     Make a prediction using the latest trained model
#     """

#     print("\n⭐️ Use case: predict")

#     # y_pred = np.ndarray()

#     print("✅ pred() done \n")
#     # return y_pred
#     pass




# LATER
#TODO refactor pipeline preproc
#TODO cascade up all hyperparameters, set default values to good model hyperparams
#TODO re-read and refactor
#TODO set up prefect + automatic production when better + differenciate 15 and 150 models
#TODO set-up prediction
#TODO

# before push VM
#TODO debug
#TODO check local storing ok
#TODO set up MLFlow + differenciate 15 and 150 models
#TODO reset default sample to 'all'

def main():
    # try :
    print(Fore.BLUE + f"\nstart of main" + Style.RESET_ALL)
    # Here only place to define hyperparams
    if CLASSIFICATION_TYPE == '15':
        settings = dict(
            classification_type = CLASSIFICATION_TYPE, # '15 types' or '150 pokemon'
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
    print('best_accuracy', best_accuracy)

    accuracy = evaluate(
        X_test=X_test,
        y_test=y_test,
        batch_size = settings['batch_size'],
        verbose=settings['verbose']
    )
    print('accuracy', accuracy)
    print(Fore.BLUE + f"\end of main" + Style.RESET_ALL)


# TODO
# pred()
    # except Exception as e:
    #     print(f"Error : {e}")



if __name__ == '__main__':
    main()
