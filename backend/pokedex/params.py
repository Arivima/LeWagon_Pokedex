import os

##################  VARIABLES  ##################
MODEL_TARGET = os.environ.get("MODEL_TARGET")
CLASSIFICATION_TYPE= os.environ.get("CLASSIFICATION_TYPE")
SAMPLED_DATASET= os.environ.get("SAMPLED_DATASET")

GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")
INSTANCE = os.environ.get("INSTANCE")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME")
# PREFECT_FLOW_NAME = os.environ.get("PREFECT_FLOW_NAME")
# PREFECT_LOG_LEVEL = os.environ.get("PREFECT_LOG_LEVEL")

##################  CONSTANTS  #####################
ML_DIR= os.path.join(os.path.expanduser('~'), ".lewagon", "pokedex")
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), ".lewagon", "pokedex", "data")
LOCAL_REGISTRY_PATH = os.path.join(os.path.expanduser('~'), ".lewagon", "pokedex", "training_outputs")

DATASET_NAME_PATH = os.path.join('..', 'all_data_name')
DATASET_TYPE_PATH = os.path.join('..', 'all_data_type')

# LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "data")
# LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "training_outputs")
