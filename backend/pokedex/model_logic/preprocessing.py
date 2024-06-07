''' Preprocessing'''
import numpy as np
import pandas as pd
import cv2

from sklearn.preprocessing import OneHotEncoder
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
    # Resize
    resized_img = cv2.resize(img, dsize=new_shape)

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
    print('  ➡️ start preprocess_features')
    print('X.shape', X.shape, type(X), 'new_shape', new_shape, 'gray', gray)

    X_processed = proc_to_bw_resized(X)

    return X_processed

def encode_target(y : pd.Series) -> pd.Series:
    ''' Encodes the target '''

    label_encoder = OneHotEncoder(sparse_output=False)
    y_encoded = label_encoder.fit_transform(y)

    print('✅ Target encoded')
    return y_encoded
