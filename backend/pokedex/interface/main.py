import pandas as pd
import numpy as np
# test
def preprocess() -> pd.DataFrame:
    """
    - load the raw dataset
    - process the raw dataset
    - returns processed data as pd df
    """
    # data = load_data()
    # processed_data = preprocess(data)
    dataframe = pd.DataFrame()
    print("✅ preprocess() done \n")
    return dataframe

@mlflow_run
def train(
        dataset ,
        learning_rate=0.0005,
        batch_size = 32,
        patience = 2
    ) -> float:
    """
    - Train on the preprocessed dataset

    => Return accuracy as a float
    """
    # create train test split
    # train model with model.py

    print("✅ train() done \n")
    accuracy = 0
    return accuracy


@mlflow_run
def evaluate(
        stage: str = "Production"
    ) -> float:
    '''

    '''

    accuracy = 0

    print("✅ evaluate() done \n")
    return accuracy

def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    '''

    '''

    y_pred = np.ndarray()

    print("✅ pred() done \n")
    return y_pred



if __name__ == '__main__':
    preprocess()
    train()
    evaluate()
    pred()
