#Imports
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_squared_error,
    r2_score
)

import numpy as np

import argparse

import mlflow
import mlflow.sklearn

#Función para argumentos de línea de comandos
def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="Número de arboles"
    )

    parser.add_argument(
        "--max_depth",
        type=int,
        default=None,
        help="Profundidad máxima"
    )

    parser.add_argument(
        "--min_samples_split",
        type=int,
        default=2,
        help="Mínimo de muestras para dividir"
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Semilla aleatoria"
    )

    return parser.parse_args()

#Función para cargar datos
def load_data(filepath):

    df = pd.read_csv(filepath)

    return df

#Función de preparación
def prepare_data(df):

    X = df.drop("quality", axis=1)

    y = df["quality"]

    return X, y

#Función para dividir datos
def split_data(X, y, random_state):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test

#Función de entrenamineto
def train_model(
    X_train,
    y_train,
    n_estimators,
    max_depth,
    min_samples_split,
    random_state
):

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state
    )

    model.fit(
        X_train,
        y_train
    )

    return model

#Función de evaluación
def evaluate_model(
    model,
    X_test,
    y_test
):

    y_pred = model.predict(X_test)

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    print(f"RMSE: {rmse}")

    print(f"R2: {r2}")

    return rmse, r2

#Función principal
def main():

    args = parse_args()

    mlflow.set_tracking_uri(
        "sqlite:///mlflow.db"
    )

    mlflow.set_experiment(
        "Wine Quality Experiments"
    )

    with mlflow.start_run():

        df = load_data(
            "winequality_clean.csv"
        )

        X, y = prepare_data(df)

        X_train, X_test, y_train, y_test = split_data(
            X,
            y,
            args.random_state
        )

        model = train_model(
            X_train,
            y_train,
            args.n_estimators,
            args.max_depth,
            args.min_samples_split,
            args.random_state
        )

        rmse, r2 = evaluate_model(
            model,
            X_test,
            y_test
        )

        mlflow.log_param(
            "n_estimators",
            args.n_estimators
        )

        mlflow.log_param(
            "max_depth",
            args.max_depth
        )

        mlflow.log_param(
            "min_samples_split",
            args.min_samples_split
        )

        mlflow.log_param(
            "random_state",
            args.random_state
        )

        mlflow.log_metric(
            "rmse",
            rmse
        )

        mlflow.log_metric(
            "r2",
            r2
        )

        metrics_file = "metrics.txt"

        with open(metrics_file, "w") as file:

            file.write(
                f"RMSE: {rmse}\n"
            )

            file.write(
                f"R2: {r2}\n"
            )

        mlflow.log_artifact(
            metrics_file,
            artifact_path="evaluation"
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            name="wine_quality_model",
            serialization_format="skops",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

        print("Experimento registrado en MLflow")

if __name__ == "__main__":

    main()