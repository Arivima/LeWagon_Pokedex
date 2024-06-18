import os

##################  VARIABLES  ##################
MODEL_TARGET = os.environ.get("MODEL_TARGET")
CLASSIFICATION_TYPE= os.environ.get("CLASSIFICATION_TYPE")
SAMPLED_DATASET= os.environ.get("SAMPLED_DATASET")
WHO= os.environ.get("WHO")

GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
INSTANCE = os.environ.get("INSTANCE")

##################  CONSTANTS  #####################
ML_DIR= os.path.join(os.path.expanduser('~'), ".lewagon", "pokedex")
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), ".lewagon", "pokedex", "data")
LOCAL_REGISTRY_PATH = os.path.join(os.path.expanduser('~'), ".lewagon", "pokedex", "training_outputs")
PRODUCTION_REGISTRY_PATH = os.path.join(os.getcwd(), 'pokedex', 'production_registry')
GCS_PATH = os.path.join(os.getcwd(), 'pokedex', 'production_registry', 'gcs')

DATASET_NAME_PATH = os.path.join('..', 'all_data_name_cleaned')
DATASET_TYPE_PATH = os.path.join('..', 'all_data_type_cleaned')
