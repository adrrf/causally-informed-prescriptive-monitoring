from sklearn.preprocessing import LabelEncoder
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.GraphUtils import GraphUtils
from IPython.display import Image, display
from elp.splitters import TimeCaseSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import pprint
import joblib
import json
import os
import time


def label_encode(df, columns):
    """
    Label encode the columns in the dataframe
    :param df: dataframe
    :param columns: list of columns to encode
    :return: dataframe with label encoded columns
    """

    for column in columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
    return df


def causal_inference_fci(df, name):
    """
    Perform causal inference using the FCI algorithm
    :param df: dataframe
    :param name: name of the image to save
    :return: pydot graph
    """

    data = df.to_numpy()
    g, edges = fci(data)
    pdy = GraphUtils.to_pydot(g, labels=df.columns)
    os.makedirs("../images", exist_ok=True)
    pdy.write_png(f"../images/{name}/causalgraph.png")
    display(Image(f"../images/{name}/causalgraph.png"))
    return pdy


def run_experiment(
    df, target, name, experiment, columns=None, n_splits=5, random_state=33
):
    """
    Run the experiment, saves the ../models in ../models folder
    :param df: dataframe
    :param target: target column
    :param columns: columns selected for the X
    :param name: name of the dataset
    :param experiment: name of the experiment
    :param n_splits: number of splits
    """
    if columns is None:
        columns = df.columns.tolist()
    X = df[columns]
    y = df[target]
    X.drop(columns=[target], inplace=True)
    ids = df.index

    tcs = TimeCaseSplit(n_splits=n_splits)
    mse = []
    rmse = []
    results = []

    os.makedirs("../models", exist_ok=True)
    os.makedirs(f"../models/{name}", exist_ok=True)
    os.makedirs("../outputs", exist_ok=True)
    os.makedirs(f"../outputs/{name}", exist_ok=True)
    os.makedirs("../images", exist_ok=True)
    os.makedirs(f"../images/{name}", exist_ok=True)

    for i, (train_index, test_index) in enumerate(tcs.split(X, y, ids)):
        iter_results = {}
        iter_results["split"] = i

        print(f"Split {i}: {len(train_index)} train rows, {len(test_index)} test rows")

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        if not os.path.exists(f"../models/{name}/{experiment}_{i}.pkl"):
            model = RandomForestRegressor(n_jobs=-1, random_state=42)
            start = time.time()
            model.fit(X_train, y_train)
            end = time.time()
            iter_results["time"] = end - start
            joblib.dump(model, f"../models/{name}/{experiment}_{i}.pkl")
        else:
            model = joblib.load(f"../models/{name}/{experiment}_{i}.pkl")

        y_pred = model.predict(X_test)

        iter_results["mse"] = mean_squared_error(y_test, y_pred)
        iter_results["rmse"] = np.sqrt(iter_results["mse"])

        mse.append(iter_results["mse"])
        rmse.append(iter_results["rmse"])

        results.append(iter_results)
        pprint.pprint(iter_results)

        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, color="blue")
        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        plt.title(f"{name}-{experiment}: Split {i}")
        plt.savefig(f"../images/{name}/{experiment}_{i}.png")
        plt.show()

    scores = {"mse": np.mean(mse), "rmse": np.mean(rmse)}
    pprint.pprint(scores)
    results.append(scores)

    json.dump(results, open(f"../outputs/{name}/{experiment}.json", "w"))
    return results
