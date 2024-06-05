import numpy as np

from tensorflow import keras
from keras import Model, Sequential, layers, regularizers, optimizers
from keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD

from colorama import Fore, Style
from typing import Tuple


def initialize_model_15(image_size: tuple) -> Model:
    """
    Initialize the model
    """
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=image_size),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(512, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(512, (3, 3), activation='relu'),
        BatchNormalization(),
        GlobalAveragePooling2D(),

        Dense(512, activation='relu', kernel_regularizer='l2'),
        Dropout(0.5),
        Dense(128, activation='relu', kernel_regularizer='l2'),
        Dropout(0.5),
        Dense(15, activation='softmax')
    ])

    print("✅ Model 15 initialized")
    return model

def initialize_model_150(image_size: tuple) -> Model:
    """
    Initialize the model
    """
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=image_size),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(512, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(512, (3, 3), activation='relu'),
        BatchNormalization(),
        GlobalAveragePooling2D(),

        Dense(512, activation='relu', kernel_regularizer='l2'),
        Dropout(0.5),
        Dense(128, activation='relu', kernel_regularizer='l2'),
        Dropout(0.5),
        Dense(150, activation='softmax')
    ])
    print("✅ Model 150 initialized")
    return model


def compile_model(model: Model, learning_rate=0.01, momentum=0.9, nesterov=True) -> Model:
    """
    Compile the model
    """
    optimizer = SGD(learning_rate=learning_rate, momentum=momentum, nesterov=nesterov)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    print("✅ Model compiled")
    # print(model.summary())
    return model

def train_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=32,
        monitor='val_loss',
        patience=5,
        epochs=100,
        validation_data=None, # overrides validation_split
        validation_split=0.3,
        verbose=0
    ) -> Tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    # Callbacks for learning rate adjustment
    callback = ReduceLROnPlateau(
        monitor=monitor,
        patience=patience,
        factor=0.2,
        min_lr=0.00001)
    # callback = EarlyStopping( monitor=monitor, patience=patience, restore_best_weights=True, verbose=1)

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    print( 'X', 'y', type(X), type(y),  X.shape, y.shape)

    # Training the model
    history = model.fit(
        datagen.flow(X, y, batch_size=64),
        validation_data=validation_data,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[callback],
        verbose=verbose
    )

    print(f"✅ Model trained on {len(X)} images with min val loss: {round(np.min(history.history[monitor]), 2)}")

    return model, history


def evaluate_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=32,
        verbose=0
    ) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on the dataset
    """

    print(Fore.BLUE + f"\nEvaluating model on {len(X)} images..." + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    metrics = model.evaluate(
        x=X, y=y,
        batch_size=batch_size,
        verbose=verbose,
        # callbacks=None,
        return_dict=True
    )

    print(f"✅ Model evaluated, accuracy: {round(metrics['accuracy'], 2)}")

    return metrics
