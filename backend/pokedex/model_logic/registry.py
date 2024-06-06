import glob
import os
import time
import pickle
import shutil

from colorama import Fore, Style
from tensorflow import keras

from pokedex.params import *
import mlflow
from mlflow.tracking import MlflowClient

def save_results(params: dict, metrics: dict, context: str) -> None:
    """
    Persist params & metrics locally on the hard drive at
    "{LOCAL_REGISTRY_PATH}/params/{current_timestamp}.pickle"
    "{LOCAL_REGISTRY_PATH}/metrics/{current_timestamp}.pickle"
    - if MODEL_TARGET='mlflow', also persist them on MLflow
    """
    print(Fore.BLUE + "\nSaving results..." + Style.RESET_ALL)
    if MODEL_TARGET == "mlflow":
        if params is not None:
            mlflow.log_params(params)
        if metrics is not None:
            mlflow.log_metrics(metrics)
        print("✅ Results saved on MLflow")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f'{CLASSIFICATION_TYPE}_{context}_{timestamp}.pickle'
    print(filename)

    # Save params locally
    if params is not None:
        params_path = os.path.join(LOCAL_REGISTRY_PATH, "params", filename)
        with open(params_path, "wb") as file:
            pickle.dump(params, file)
    print("✅ Params saved locally")

    # Save metrics locally
    if metrics is not None:
        metrics_path = os.path.join(LOCAL_REGISTRY_PATH, "metrics", filename)
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    print("✅ Metrics saved locally")


def save_model(context : str, model: keras.Model = None) -> None:
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.keras"
    - if MODEL_TARGET='mlflow', also persist it on MLflow instead of GCS (for unit 0703 only)
    """
    print(Fore.BLUE + "\nSaving model..." + Style.RESET_ALL)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f'{CLASSIFICATION_TYPE}_{context}_{timestamp}.keras'
    print(filename)

    # Save model locally
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", filename)
    model.save(model_path)

    print("✅ Model saved locally")

    if MODEL_TARGET == "mlflow":
        project_name = f'{MLFLOW_MODEL_NAME}_{CLASSIFICATION_TYPE}'
        mlflow.tensorflow.log_model(
            model=model,
            artifact_path="model",
            registered_model_name=project_name
        )
        print("✅ Model saved to MLflow")
        return None

    return None


def load_model(stage="Production", include_filename=False) -> keras.Model:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from MLFLOW (by "stage") if MODEL_TARGET=='mlflow'

    Return None (but do not Raise) if no model is found

    """
    if MODEL_TARGET == "local":
        print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

        # Get the latest model version name by the timestamp on disk
        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
        local_model_paths = glob.glob(f"{local_model_directory}/{CLASSIFICATION_TYPE}*")

        if not local_model_paths:
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

        latest_model = keras.models.load_model(most_recent_model_path_on_disk)

        print('model :', os.path.basename(most_recent_model_path_on_disk))
        print("✅ Model loaded from local disk")
        if include_filename:
            model_filename = os.path.basename(most_recent_model_path_on_disk)
            return latest_model, model_filename

        return latest_model

    elif MODEL_TARGET == "mlflow":
        print(Fore.BLUE + f"\nLoad [{stage}] model from MLflow..." + Style.RESET_ALL)

        # Load model from MLflow
        model = None
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()

        project_name = f'{MLFLOW_MODEL_NAME}_{CLASSIFICATION_TYPE}'

        try:
            model_versions = client.get_latest_versions(name=project_name, stages=[stage])
            model_uri = model_versions[0].source

            assert model_uri is not None
        except:
            print(f"\n❌ No model found with name {project_name} in stage {stage}")

            return None

        model = mlflow.tensorflow.load_model(model_uri=model_uri)

        print("✅ Model loaded from MLflow")
        if include_filename:
            return latest_model, model_uri

        return model
    else:
        return None


def mlflow_transition_model(current_stage: str, new_stage: str) -> None:
    """
    Transition the latest model from the `current_stage` to the
    `new_stage` and archive the existing model in `new_stage`
    """
    if MODEL_TARGET == "mlflow":
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        client = MlflowClient()

        project_name = f'{MLFLOW_MODEL_NAME}_{CLASSIFICATION_TYPE}'

        version = client.get_latest_versions(name=project_name, stages=[current_stage])

        if not version:
            print(f"\n❌ No model found with name {project_name} in stage {current_stage}")
            return None

        client.transition_model_version_stage(
            name=project_name,
            version=version[0].version,
            stage=new_stage,
            archive_existing_versions=True
        )

        print(f"✅ Model {project_name} (version {version[0].version}) transitioned from {current_stage} to {new_stage}")

    return None


def mlflow_run(func):
    """
    Generic function to log params and results to MLflow along with TensorFlow auto-logging

    Args:
        - func (function): Function you want to run within the MLflow run
        - params (dict, optional): Params to add to the run in MLflow. Defaults to None.
        - context (str, optional): Param describing the context of the run. Defaults to "Train".
    """

    def wrapper(*args, **kwargs):
        if MODEL_TARGET == "mlflow":
            mlflow.end_run()
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(experiment_name=MLFLOW_EXPERIMENT)

            with mlflow.start_run():
                print('URI : ', MLFLOW_TRACKING_URI)
                print('EXPERIMENT : ',MLFLOW_EXPERIMENT)

                mlflow.tensorflow.autolog()
                results = func(*args, **kwargs)
            print("✅ mlflow_run auto-log done")
            return results
        else:
            results = func(*args, **kwargs)
        return results
    return wrapper



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
        local_params_paths = glob.glob(f"{local_params_directory}/{CLASSIFICATION_TYPE}_{context}*")
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
        local_metrics_paths = glob.glob(f"{local_metrics_directory}/{CLASSIFICATION_TYPE}_{context}*")
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




def load_latest_run(stage="Production") -> keras.Model:
    """
    Loads the latest run from MLFLOW (by "stage") if MODEL_TARGET=='mlflow'
    Return metrics
    """
    if MODEL_TARGET == "mlflow":
        print(Fore.BLUE + f"\nLoad latest [{stage}] run from MLflow..." + Style.RESET_ALL)

        # Load model from MLflow
        model = None
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()

        project_name = f'{MLFLOW_MODEL_NAME}_{CLASSIFICATION_TYPE}'

        try:
            model_versions = client.get_latest_versions(name=project_name, stages=[stage])
            model_uri = model_versions[0].source
            assert model_uri is not None

        except:
            print(f"\n❌ No model found with name {project_name} in stage {stage}")

            return None

        model = mlflow.tensorflow.load_model(model_uri=model_uri)
        run_id = model_uri.split('/')[1]
        run = mlflow.get_run(run_id)
        metrics = run.data.metrics
        params = run.data.params
        print(metrics, params)

        print(f"✅ Latest {stage} run loaded from MLflow")
        return metrics
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
            model, model_filename = load_model(include_filename=True)
            model_path = os.path.join(registry_path, "models", model_filename)
            print('Copying last run model to ', model_path)
            model.save(model_path)
            print("\n✅ Model, params, and metrics staged to production")

        else:
            print("\n✅ Keeping production model in production")

    #TODO
    elif MODEL_TARGET == 'mlflow':
        print('➡️ Staging to production')
        # If latest model more performant than latest production model, should be moved to production
        staging_model_metrics = load_latest_run(stage="Staging")
        print(staging_model_metrics)
        production_model_metrics = load_latest_run(stage="Production")
        print(production_model_metrics)

        if production_model_metrics is None:
            print('No model currently in production : Staged current model to production')
            mlflow_transition_model(current_stage="Staging", new_stage="Production")
        elif staging_model_metrics is None:
            print('ERROR : No model currently in production ')
        elif staging_model_metrics['best_accuracy'] > production_model_metrics['best_accuracy']:
            print('current model is better than production model : Staged current model to production')
            mlflow_transition_model(current_stage="Staging", new_stage="Production")
        else:
            print('production model is still better than current model : Production model stayed the same')

    return None
