''' Preprocessing'''
import numpy as np
import pandas as pd
import cv2

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


def proc_to_bw_resized(
    img : np.ndarray ,
    new_shape : tuple = (128,128),
    gray : bool = False
    )-> np.ndarray :
    """
    Process an image:
    - Resize image to new_shape
    - Optionally convert to grayscale
    """
    # Convert RGB to BGR
    img_cv2 = img[...,::-1]

    # Resize
    resized_img = cv2.resize(img_cv2, dsize=new_shape)

    # Optionally converts to grayscale
    if gray:
        gray_image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        return gray_image

    return resized_img

#TODO resolve iteration on series vs image per image
def preprocess_features(
    X: pd.Series,
    new_shape: tuple = (128, 128),
    gray: bool = True
    ) -> pd.Series:
    ''' preprocess images '''

    X_processed = proc_to_bw_resized(X)

    return X_processed

def encode_target(y : pd.Series) -> pd.Series:
    ''' Encodes the target '''

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    return y_categorical
