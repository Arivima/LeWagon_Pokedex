# function load data

#Import
import os
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def load_images_from_folders(base_dir:str)-> pd.DataFrame:
    """
    Load image from folder and put it in a Dataframe columns = [ images : nd.array (224,224,3) , type ]
    """

    X = []
    y = []
    class_labels = [folder for folder in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, folder))]
    for label in class_labels:
        class_dir = os.path.join(base_dir, label)
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            try:
                with Image.open(file_path) as img:
                    img_array = np.array(img)
                    X.append(img_array)
                    y.append(label)
            except Exception as e:
                print(f"Error loading image {file_path}: {e}")
    X = np.array(X)
    # Convertir chaque ndarray en une liste
    data = [X[i] for i in range(X.shape[0])]
    # Créer un DataFrame à partir de la liste
    df = pd.Series(data, name='image').to_frame()
    df['type']=y
    return df

def display_images(df, num_images=10):
    plt.figure(figsize=(15, 15))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(df.image.loc[i])
        plt.title(df.type.loc[i])
        plt.axis("off")
    plt.show()
