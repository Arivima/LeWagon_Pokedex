import glob
import os
import time
import pickle
import shutil

from colorama import Fore, Style
from tensorflow import keras

from pokedex.params import *

def save_results(params: dict, metrics: dict, context: str) -> None:
    """
    Persist params & metrics locally on the hard drive at
    "{LOCAL_REGISTRY_PATH}/params/{current_timestamp}.pickle"
    "{LOCAL_REGISTRY_PATH}/metrics/{current_timestamp}.pickle"
    """
    print(Fore.BLUE + "\nSaving results..." + Style.RESET_ALL)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f'{CLASSIFICATION_TYPE}_{WHO}_{context}_{timestamp}.pickle'
    print(filename)

    # Save params locally
    if params is not None:
        params_path = os.path.join(LOCAL_REGISTRY_PATH, "params", filename)
        with open(params_path, "wb") as file:
            pickle.dump(params, file)
    print(f"✅ Params saved locally {params_path}")

    # Save metrics locally
    if metrics is not None:
        metrics_path = os.path.join(LOCAL_REGISTRY_PATH, "metrics", filename)
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    print(f"✅ Metrics saved locally {metrics_path}")


def save_model(context : str, model: keras.Model = None) -> None:
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    """
    print(Fore.BLUE + f"\nSaving model ... - context {context}" + Style.RESET_ALL)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f'{CLASSIFICATION_TYPE}_{WHO}_{context}_{timestamp}.h5'
    print(filename)

    # Save model locally
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", filename)
    model.save(model_path)

    print(f"✅ Model saved locally at {model_path}")

    return None


def load_model(
    stage : str = "Production",
    include_filename : bool = False,
    model_type : str = CLASSIFICATION_TYPE
    ) -> keras.Model:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)

    Return None (but do not Raise) if no model is found

    """
    if MODEL_TARGET == "local":

        # Get the latest model version name by the timestamp on disk
        if stage == 'Production':
            registry_path = PRODUCTION_REGISTRY_PATH
        elif stage == 'Staging':
            registry_path = LOCAL_REGISTRY_PATH
        else:
            print(f'ERROR : unknown stage {stage}')
            return None

        print(Fore.BLUE + f"\nLoad latest model from local registry, stage {stage}" + Style.RESET_ALL)

        local_model_directory = os.path.join(registry_path, "models")
        pattern_filename = f'{model_type}_{WHO}*'
        local_model_paths = glob.glob(f"{local_model_directory}/{pattern_filename}")

        if not local_model_paths:
            print(f"No model {model_type} at stage {stage} on local disk")
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

        latest_model = keras.models.load_model(most_recent_model_path_on_disk)

        print('model :', os.path.basename(most_recent_model_path_on_disk))
        print("✅ Model loaded from local disk")
        if include_filename:
            model_filename = os.path.basename(most_recent_model_path_on_disk)
            return latest_model, model_filename

        return latest_model

    else:
        return None



def load_results(context : str, stage="Production", include_filename=False) -> keras.Model:
    """
    Return latest results (metrics, params):
    - locally (latest one in alphabetical order)
    Includes the filename if specified to True
    Return None (but do not Raise) if no results are found

    """
    if MODEL_TARGET == "local":
        if stage == "Production":
            registry_path = os.path.join(os.getcwd(), 'pokedex', 'production_registry')
        elif stage== 'Staging':
            registry_path = LOCAL_REGISTRY_PATH
        else:
            print(f'Stage {stage} not known')
            return None

        # Get the latest params version name by the timestamp on disk
        print(Fore.BLUE + f"\nLoad {stage} params from local registry..." + Style.RESET_ALL)
        local_params_directory = os.path.join(registry_path, "params")
        local_params_paths = glob.glob(f"{local_params_directory}/{CLASSIFICATION_TYPE}_{WHO}_{context}*")
        if not local_params_paths:
            print(f'No {stage} params on local disk at dir', local_params_directory)
            return None
        most_recent_params_path_on_disk = sorted(local_params_paths)[-1]
        with open(most_recent_params_path_on_disk, "rb") as file:
            latest_params = pickle.load(file)
        params_filename = os.path.basename(most_recent_params_path_on_disk)
        print(params_filename, latest_params)
        print("✅ Params loaded from local disk")

        # Get the latest metrics version name by the timestamp on disk
        print(Fore.BLUE + f"\nLoad {stage} metrics from local registry..." + Style.RESET_ALL)
        local_metrics_directory = os.path.join(registry_path, "metrics")
        local_metrics_paths = glob.glob(f"{local_metrics_directory}/{CLASSIFICATION_TYPE}_{WHO}_{context}*")
        if not local_metrics_paths:
            print(f'No {stage} metrics on local disk', local_metrics_directory)
            return None
        most_recent_metrics_path_on_disk = sorted(local_metrics_paths)[-1]
        with open(most_recent_metrics_path_on_disk, "rb") as file:
            latest_metrics = pickle.load(file)
        metrics_filename = os.path.basename(most_recent_metrics_path_on_disk)
        print(metrics_filename, latest_metrics)
        print("✅ metrics loaded from local disk")

        if include_filename:
            return latest_params, latest_metrics, params_filename, metrics_filename

        return latest_params, latest_metrics

    print('not in local mode')
    return None




def delete_production_folders_content(registry_path):
    '''
    removes all content of subdirectories : metrics, params, models in 'production' dir
    '''
    print(Fore.RED + f"\nDeleting content of production folders in {registry_path}..." + Style.RESET_ALL)

    folders_to_clear = ["params", "metrics", "models"]
    for folder in folders_to_clear:
        folder_path = os.path.join(registry_path, folder)

        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(Fore.RED + f'Failed to delete {file_path}. Reason: {e}' + Style.RESET_ALL)

    print(Fore.GREEN + "✅ Content of production folders deleted" + Style.RESET_ALL)



def compare_vs_production():
    '''
    - loads metrics from the last run
    - loads metrics from production stage
    - compare both metrics:
        - if production < last_run : stage last_run to production, archive other
        - if production >= last_run : do nothing
    '''
    print(Fore.BLUE + f"\nCompare latest model performance vs production..." + Style.RESET_ALL)

    if MODEL_TARGET == 'local':
        last_results = load_results(context='evaluate', stage='Staging', include_filename=True)
        # no params or metrice presnet on local registry
        if last_results is None:
            return None
        last_params, last_metrics, params_filename, metrics_filename = last_results
        if last_params is None or last_metrics is None:
            return None

        prod_results = load_results(context='evaluate', stage='Production', include_filename=False)

        if prod_results:
            production_params, production_metrics = prod_results

        # Stage current model to production
        if prod_results is None or \
            last_metrics['accuracy'] > production_metrics['accuracy']:

            print(Fore.BLUE + "\nStaging current model to production ..."+ Style.RESET_ALL)
            registry_path = os.path.join(os.getcwd(), 'pokedex', 'production_registry')

            # Delete content of production subfolders (metrics, params, models)
            delete_production_folders_content(registry_path)

            # Copy last run (params, metrics, and model) to production stage
            # Save params locally
            params_path = os.path.join(registry_path, "params", params_filename)
            print('Copying last run params to ', params_path)
            with open(params_path, "wb") as file:
                pickle.dump(last_params, file)

            # Save metrics locally
            metrics_path = os.path.join(registry_path, "metrics", metrics_filename)
            print('Copying last run metrics to ', metrics_path)
            with open(metrics_path, "wb") as file:
                pickle.dump(last_metrics, file)

            # Save model locally
            model, model_filename = load_model(stage='Staging', model_type=CLASSIFICATION_TYPE, include_filename=True)
            model_path = os.path.join(registry_path, "models", model_filename)
            print('Copying last run model to ', model_path)
            model.save(model_path)
            print("\n✅ Model, params, and metrics staged to production")

        else:
            print("\n✅ Keeping production model in production")

    return None
