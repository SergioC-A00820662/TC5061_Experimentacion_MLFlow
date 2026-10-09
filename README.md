# TC5051_Experimentacion_MLFlow

## Descripción

Este proyecto implementa un flujo reproducible de entrenamiento y experimentación para el dataset Wine Quality. El modelo utilizado es Random Forest Regressor y los experimentos son registrados utilizando MLflow Tracking.

El script permite modificar hiperparámetros desde la línea de comandos sin editar el código y registra automáticamente los parámetros, métricas, artefactos y modelo de cada ejecución.

## Requisitos

- Python 3.12
- Las dependencias indicadas en `requirements.txt`

## Instalación

Crear un ambiente virtual:

    python3 -m venv .venv

Activar el ambiente:

    source .venv/bin/activate

Instalar las dependencias:

    python -m pip install -r requirements.txt

## Ejecución

El script puede ejecutarse utilizando sus valores predeterminados:

    python train.py

También pueden especificarse los hiperparámetros desde línea de comandos:

    python train.py \
    --n_estimators 200 \
    --max_depth 10 \
    --min_samples_split 5 \
    --random_state 123

## Parámetros disponibles

- `--n_estimators`: número de árboles del Random Forest.
- `--max_depth`: profundidad máxima de los árboles.
- `--min_samples_split`: número mínimo de muestras requeridas para dividir un nodo.
- `--random_state`: semilla utilizada para reproducibilidad.

## MLflow Tracking

Cada ejecución registra en MLflow:

- Hiperparámetros del modelo.
- RMSE.
- R².
- Archivo de métricas como artefacto.
- Modelo entrenado.

Para iniciar la interfaz de MLflow:

    python -m mlflow ui --backend-store-uri sqlite:///mlflow.db

Después abrir:

    http://localhost:5000

Los experimentos se agrupan bajo:

    Wine Quality Experiments

## Reproducibilidad

La semilla aleatoria se controla mediante `random_state` y se utiliza tanto en la división de entrenamiento/prueba como en Random Forest Regressor. Ejecutar el script con los mismos argumentos y dependencias permite reproducir los resultados del experimento.
