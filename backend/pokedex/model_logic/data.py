''' function load data '''

#Import
import os
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#TODO delete after refactor
from pokedex.model_logic.preprocessing import proc_to_bw_resized

from pokedex.params import *

import google
from google.cloud import storage
from google.cloud.exceptions import NotFound
from io import BytesIO



def sample_dataframe(df, num_samples):
    """
    Randomly samples num_samples items from the given DataFrame.
    """
    sampled_df = df.sample(n=num_samples, random_state=42)
    return sampled_df


def load_images_from_bucket(
    base_dir: str,
    sample: str = '300',
    new_size: tuple = (128,128),
) -> pd.DataFrame:
    """
    Load images from a GCP bucket and put them in a DataFrame.
    columns = [ images : nd.array (128,128,3) , type ]
    """
    X = []
    y = []

    # Initialize a client
    client = storage.Client()
    print(client.project)

    # Get the bucket
    try:
        bucket = client.get_bucket(BUCKET_NAME)
        print(bucket.name)
        # Perform operations with the bucket

        # List all blobs in the bucket
        blobs = list(bucket.list_blobs(prefix=base_dir))
        print(len(blobs), blobs.count(), blobs.index(0))

        # Extract the subdirs labels from the directory structure
        subdirs = list(set([blob.name.split('/')[0] for blob in blobs if '/' in blob.name]))
        print(subdirs)

        for label in subdirs:
            subdir_blobs = [blob for blob in blobs if blob.name.startswith(label + '/')]
            print(subdir_blobs.count())
            for blob in subdir_blobs:
                try:
                    # Download the image as bytes
                    img_bytes = blob.download_as_bytes()

                    # Open the image
                    with Image.open(BytesIO(img_bytes)) as img:
                        img_array = np.array(img)
                        img_proc = proc_to_bw_resized(img_array, new_size)
                        X.append(img_proc)
                        y.append(label)
                except Exception as e:
                    print(f"Error loading image {blob.name}: {e}")

        X = np.array(X)

        # Convert each ndarray to a list
        data = [X[i] for i in range(X.shape[0])]

        # Create a DataFrame from the list
        df = pd.Series(data, name='image').to_frame()
        df['label'] = y
        print(df.describe())

        # If sample is not 'all', randomly sample X to reduce the size of the df
        if sample.isdigit() and int(sample) < len(df):
            df = df.sample(n=int(sample))
        df = sample_dataframe(df, 300)

        print(f"✅ Data loaded from bucket {BUCKET_NAME}{base_dir}")
        return df

    except google.cloud.exceptions.NotFound:
        print(f"Bucket {BUCKET_NAME} does not exist.")

def load_images_from_folders(
    base_dir : str,
    sample : str = '300',
    new_size : tuple = (128,128),
    ) -> pd.DataFrame:
    """
    Load images from folder and put it in a Dataframe
    columns = [ images : nd.array (224,224,3) , type ]
    """
    X = []
    y = []
    print('A')
    class_labels = [folder for folder in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, folder))]
    for label in class_labels:
        class_dir = os.path.join(base_dir, label)
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            try:
                print('B', )
                with Image.open(file_path) as img:
                    img_array = np.array(img)
                    print('C')
                    #TODO delete line proc_to_bw_resized when refactored
                    img_proc = proc_to_bw_resized(img_array, new_size)
                    X.append(img_proc)
                    y.append(label)
            except Exception as e:
                print(f"Error loading image {file_path}: {e}")
    X = np.array(X)

    # Convertir chaque ndarray en une liste
    data = [X[i] for i in range(X.shape[0])]

    # Créer un DataFrame à partir de la liste
    df = pd.Series(data, name='image').to_frame()
    df['label'] = y
    print(df.describe())

    # if sample is not 'all', randomly samples X to reduce the size of the df
    # if sample.isdigit() and int(sample) < len(df):
    #     df = df.sample(n=int(sample))
    df = sample_dataframe(df, 300)

    print(f"✅ Data loaded from {base_dir}")
    return df



def display_images(df : pd.DataFrame, num_images=10):
    '''
    displays a sample of images with associated label from the dataset
    '''
    plt.figure(figsize=(15, 15))
    # Ensure it doesn't try to display more images than available
    for i in range(min(num_images, len(df))):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(df.image.iloc[i])
        plt.title(df.label.iloc[i])
        plt.axis("off")
    plt.show()
