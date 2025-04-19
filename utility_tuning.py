import os
import json
import logging
import numpy as np
import pandas as pd
from copy import deepcopy
from ultralytics import YOLO
from ray.tune.result_grid import ResultGrid
from utility_models import get_backbone_path
from config import MODEL_BACKBONE_LAYERS, SEARCH_SPACES

# Configurar el registro de errores
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


def create_tune_dict(models, datasets, optimizers, use_ray=False, use_freeze=False, extra_params=None):
    datasets_yaml_dir = os.path.abspath("datasets_yaml")

    tune_dict = {}

    for model_name in models:
        for dataset_yaml in datasets:
            # Path completo del archivo Yaml
            data_yaml = os.path.join(datasets_yaml_dir, dataset_yaml)

            for optimizer in optimizers:
                # Definir nombre del experimento
                caso = os.path.splitext(dataset_yaml)[0] + "_" + model_name + "_" + optimizer

                # Parámetros
                model_params = {"model": get_backbone_path(model_name), "task": "segment"}
                tuning_params = {"data":data_yaml, "optimizer": optimizer}
                if extra_params is not None:
                    tuning_params.update(extra_params)

                # Guardarlos en un diccionario por caso
                tune_dict[caso] = {"model_params": model_params, "tuning_params": tuning_params,
                                   "model_name": model_name, "use_ray": use_ray, "use_freeze": use_freeze, "done": False}

    return tune_dict


def save_tune_config(tune_dict, json_file):
    # Crea el directorio si no existe
    directorio = os.path.dirname(json_file)
    if directorio and not os.path.exists(directorio):
        os.makedirs(directorio)

    with open(json_file, 'w', encoding='utf-8') as archivo:
        json.dump(tune_dict, archivo, ensure_ascii=False, indent=4)


def load_tune_config(json_file) -> dict:
    with open(json_file, 'r', encoding='utf-8') as archivo:
        tune_dict = json.load(archivo)
    return tune_dict


def run_tuning_file(json_file):
    # Cargar el archivo de configuración para el tuning
    try:
        tune_dict = load_tune_config(json_file)
    except Exception as e:
        print(f"Error al cargar {json_file}: {e}")
        return None

    # Iterar para cada caso
    for case, values in tune_dict.items():
        if not values.get("done", False):
            try:
                # Realizar el tuning
                completed, best_metrics_dict = do_tuning(values)
                # Si se logró el tuning
                if completed:
                    # Si fue hecho con raytune
                    if best_metrics_dict is not None:
                        tune_dict[case].update(best_results=best_metrics_dict)
                    # Actualizar el diccionario y guardar los cambios
                    tune_dict[case].update(done=True)
                    try:
                        save_tune_config(tune_dict, json_file)
                    except Exception as e:
                        print(f"Error al guardar {json_file}: {e}")

            except Exception as e:
                print(f"Error al procesar {case}: {e}")
                continue


def do_tuning(values):
    try:
        # Cargar parámetros del diccionario
        model_params = values["model_params"]
        tuning_params = deepcopy(values["tuning_params"])
        model_name = values["model_name"]
        use_ray = values["use_ray"]
        use_freeze = values["use_freeze"]

        # Actualizar los parámetros de entrenamiento
        tuning_params.update(use_ray=use_ray, space=SEARCH_SPACES[use_ray])
        if use_freeze:
            tuning_params.update(freeze=MODEL_BACKBONE_LAYERS[model_name])
        if use_ray:
            tuning_params.update(gpu_per_trial=1)

        # Realizar tuning
        result_grid = thread_safe_tuning(model_params, tuning_params)

        # Si el tuning se realiza con ray_tune la respuesta es un objeto ResultGrid
        best_metrics_dict = None if result_grid is None else procesar_result_grid(result_grid)

        return True, best_metrics_dict

    except KeyError as e:
        logger.error(f"KeyError: {e} - Falta una clave en el diccionario de valores.")
        return False, None
    except Exception as e:
        logger.error(f"Error durante el tuning: {e}")
        return False, None


def thread_safe_tuning(model_params, tuning_params)  -> (ResultGrid | None):
    # Cargar el modelo
    local_model = YOLO(**model_params)
    # Realizar el tuning
    result_grid = local_model.tune(**tuning_params)
    # Borrar el modelo para asegurarse de guardar memoria y retornar resultados
    del local_model
    return result_grid


def raytune_filtar_metrics(df):
    # Lista de columnas específicas a incluir
    add_columnas = ["metrics/precision(M)", "metrics/recall(M)", "metrics/mAP50(M)", "metrics/mAP50-95(M)", "trial_id"]
    miss_columnas = ["config/data", "config/epochs"]

    # Filtrar columnas que comienzan con "config/" o están en add_columnas
    columnas_deseadas = [col for col in df.columns if (col.startswith("config/") or col in add_columnas) and col not in miss_columnas]
    df_filtrado = df[columnas_deseadas]

    # Seleccionar la fila con el máximo "metrics/mAP50(M)" usando idxmax
    idx_max = df_filtrado["metrics/mAP50(M)"].idxmax()
    fila_seleccionada = df_filtrado.loc[idx_max].copy()

    # Calcular F1_score(M) con manejo de división por cero
    precision = fila_seleccionada.get("metrics/precision(M)", 0)
    recall = fila_seleccionada.get("metrics/recall(M)", 0)
    fila_seleccionada["metrics/F1_score(M)"] = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

    return fila_seleccionada


def raytune_serie_a_diccionario(serie):
    """
    Convierte una serie de Pandas en un diccionario estructurado.

    - Los índices que comienzan con "config/" se agrupan en un subdiccionario "config".
    - Los índices que comienzan con "metrics/" se agrupan en un subdiccionario "metrics".
    - Los demás índices se añaden directamente al diccionario principal.

    Parámetros:
    - serie (pd.Series): La serie de entrada.

    Retorna:
    - dict: El diccionario estructurado.
    """
    resultado = {}
    config_dict = {}
    metrics_dict = {}

    for indice, valor in serie.items():
        # Convertir valores NumPy a tipos nativos
        if isinstance(valor, (np.floating, np.integer)):
            valor = valor.item()  # Convierte np.float64, np.int64, etc. a float/int
        elif isinstance(valor, np.bool_):
            valor = bool(valor)    # Convierte np.bool_ a bool nativo
        elif isinstance(valor, np.ndarray):
            valor = valor.tolist() # Convierte arrays NumPy a listas

        if indice.startswith("config/"):
            # Eliminar el prefijo "config/" y añadir al subdiccionario config
            clave_config = indice.replace("config/", "", 1)
            config_dict[clave_config] = valor
        elif indice.startswith("metrics/"):
            # Eliminar el prefijo "metrics/" y añadir al subdiccionario metrics
            clave_metrics = indice.replace("metrics/", "", 1)
            metrics_dict[clave_metrics] = valor
        else:
            # Añadir directamente al diccionario principal
            resultado[indice] = valor

    # Añadir los subdiccionarios al resultado si no están vacíos
    if config_dict:
        resultado["config"] = config_dict
    if metrics_dict:
        resultado["metrics"] = metrics_dict

    return resultado


def procesar_result_grid(result_grid):
    # Obtenemos el mejor caso de cada entrenamiento
    all_best_metrics_df = result_grid.get_dataframe(filter_metric="metrics/mAP50(M)", filter_mode="max")

    # Filtramos solo el mejor de ellos y tambien se filtran algunas columnas
    best_metrics_df = raytune_filtar_metrics(all_best_metrics_df)

    # Lo convertimos en dicionario y lo retornamos
    return raytune_serie_a_diccionario(best_metrics_df)


if __name__ == "__main__":
    # Definir los parametros para realizar el tuning
    models = ['yolov8l-seg', 'yolov8x-seg', 'yolov9e-seg', 'yolo11l-seg', 'yolo11x-seg']
    datasets = ['Deepfish.yaml']
    optimizers = ['SGD']
    use_ray = True      # Utilizar Raytune como tuner
    use_freeze = False  # Congelar backbone? (Transfer Learning)
    extra_training_params = {"epochs": 80, "iterations": 50, "batch": 8, "single_cls": True, "cos_lr": True}
    # batch cambiado manualmente a 6 para yolov9e

    # Archivo donde guardar la configuración
    config_file = "tuning_Deepfish.json"

    # Crear configuraciones y guardarlos en un archivo
    #tune_dict = create_tune_dict(models, datasets, optimizers, use_ray, use_freeze, extra_training_params)
    #save_tune_config(tune_dict, config_file)

    # Realizar tuning
    run_tuning_file(config_file)
