''' function load data '''

#Import
import os
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#TODO delete after refactor
from pokedex.model_logic.preprocessing import proc_to_bw_resized



def sample_dataframe(df, num_samples):
    """
    Randomly samples num_samples items from the given DataFrame.
    """
    sampled_df = df.sample(n=num_samples, random_state=42)
    return sampled_df



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
    class_labels = [folder for folder in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, folder))]
    for label in class_labels:
        class_dir = os.path.join(base_dir, label)
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            try:
                with Image.open(file_path) as img:
                    img_array = np.array(img)
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

    # if sample is not 'all', randomly samples X to reduce the size of the df
    # if sample.isdigit() and int(sample) < len(df):
    #     print(sample)
    df = sample_dataframe(df, int(sample))

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
