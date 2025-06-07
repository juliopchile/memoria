import os
import json
import logging
from typing import Any, cast
import numpy as np
from copy import deepcopy
from ultralytics import YOLO
from ray import tune
from ray.tune.result_grid import ResultGrid
from pandas import DataFrame, Series
from utility_models import get_backbone_path
from config import MODEL_BACKBONE_LAYERS

SEARCH_SPACES = {
        False: {
                'lr0': (1e-4, 0.01),
                'lrf': (0.01, 0.5),
                'momentum': (0.6, 0.98),
                'weight_decay': (0.0, 0.001),
                'warmup_epochs': (0.0, 5.0),
                'warmup_momentum': (0.0, 0.95)
        },
        True: {
                'lr0': tune.uniform(1e-4, 0.01),
                'lrf': tune.uniform(0.01, 0.5),
                'momentum': tune.uniform(0.6, 0.98),
                'weight_decay': tune.uniform(0.0, 0.001),
                'warmup_epochs': tune.uniform(0.0, 5.0),
                'warmup_momentum': tune.uniform(0.0, 0.95)
        }
}

# Configurar el registro de errores
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


def create_tune_dict(models: list[str], datasets: list[str], optimizers: list[str], use_ray: bool = False, use_freeze: bool = False,
                     extra_params: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """ Esta función crea diccionarios que definen los parámetros de configuración para realizar diferentes tunings con Ultralytics.

    :param list[str] models: Lista de nombres de modelos a tunear.
    :param list[str] datasets: Lista de datasets a usar.
    :param list[str] optimizers: Lista de optimizadores a usar.
    :param bool use_ray: Determina si usar tuning con RayTune, por defecto False.
    :param bool use_freeze: Determina si realizar congelamiento de capas, por defecto False.
    :param dict[str, Any] extra_params: Diccionario de parámetros extras a evaluar, por defecto None.
    :return dict[str, dict[str, Any]]: Diccionario con los distintos tunes definidos. Los items del diccionario son:
    
        - "model_params": Diccionario con {"model": model_path, "task": "segment"}.
        - "tuning_params": Diccionario con {"data": data_yaml, "optimizer": optimizer}.
        - "model_name": Nombre del modelo (str).
        - "use_ray": Si usar o no RayTune (bool).
        - "use_freeze": Si congelar o no el backbone (bool).
        - "done": Variable que se utilizará para indicar si el tuning se completó.
    """
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


def save_tune_config(tune_dict: dict[str, dict[str, Any]], json_file: str):
    """ Guarda la información de un diccionario de tuning en un archivo JSON.

    :param dict[str, dict[str, Any]] tune_dict: Diccionario de configuraciones de tuning.
    :param str json_file: Ruta donde guardar como archivo JSON.
    """
    # Crea el directorio si no existe
    directorio = os.path.dirname(json_file)
    if directorio and not os.path.exists(directorio):
        os.makedirs(directorio)

    with open(json_file, 'w', encoding='utf-8') as archivo:
        json.dump(tune_dict, archivo, ensure_ascii=False, indent=4)


def load_tune_config(json_file: str) -> dict[str, dict[str, Any]]:
    """ Carga las configuraciones para el tuning desde un arhivo JSON a un diccionario.

    :param str json_file: Ruta del archivo JSON.
    :return dict[str, dict[str, Any]]: Diccionario con las configuraciones de tuning.
    Los items del diccionario debiesen ser:
    
        - "model_params": Diccionario con {"model": model_path, "task": "segment"}.
        - "tuning_params": Diccionario con {"data": data_yaml, "optimizer": optimizer}.
        - "model_name": Nombre del modelo (str).
        - "use_ray": Si usar o no RayTune (bool).
        - "use_freeze": Si congelar o no el backbone (bool).
        - "done": Variable que se utiliza para indicar si el tuning ya se realizó o no.
    """
    with open(json_file, 'r', encoding='utf-8') as archivo:
        tune_dict = json.load(archivo)
    return tune_dict


def run_tuning_file(json_file: str):
    """
    Carga la configuración de tuning desde un archivo JSON y ejecuta los casos pendientes.

    Para cada caso que no esté marcado como 'done', se ejecuta el tuning y se actualiza el estado
    en el archivo. Si ocurre un error crítico al cargar el archivo, se lanza una excepción.

    :param str json_file: Ruta del archivo JSON de configuración.
    :raises RuntimeError: Si no se puede cargar el archivo de configuración.
    """
    # Cargar el archivo de configuración para el tuning
    try:
        tune_dict = load_tune_config(json_file)
    except Exception as e:
        raise RuntimeError(f"No se pudo cargar el archivo '{json_file}': {e}")

    # Iterar sobre cada caso
    for case, tune_config in tune_dict.items():
        if not tune_config.get("done", False):
            try:
                completed, best_metrics_dict = do_tuning(tune_config)

                if completed:
                    if best_metrics_dict is not None:
                        tune_dict[case].update(best_results=best_metrics_dict)
                    tune_dict[case].update(done=True)

                    try:
                        save_tune_config(tune_dict, json_file)
                    except Exception as e:
                        raise RuntimeError(f"Error al guardar cambios en '{json_file}': {e}")

            except Exception as e:
                print(f"[Advertencia] Error al procesar el caso '{case}': {e}")
                continue


def do_tuning(tune_config: dict[str, Any]) -> tuple[bool, dict | None]:
    """ Se realiza un tuning utilizando un diccionario de configuración.

    :param dict[str, Any] tune_config: Diccionario de configuración del tuning.
    :return tuple[bool, dict | None]: Tupla de valores que identifican los resultados del tuning.
    """
    try:
        # Cargar parámetros del diccionario
        model_params = tune_config["model_params"]
        tuning_params = deepcopy(tune_config["tuning_params"])
        model_name = tune_config["model_name"]
        use_ray = tune_config["use_ray"]
        use_freeze = tune_config["use_freeze"]

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


def thread_safe_tuning(model_params: dict[str, Any], tuning_params: dict[str, Any]) -> ResultGrid | None:
    """ Realiza la búsqueda de hiperparámetros por medio del tuning de un modelo. Se retornan los resultados si aplica.

    :param dict[str, Any] model_params: Diccionario con {"model": model_path, "task": "segment"}.
    :param dict[str, Any] tuning_params: Diccionario con los parámetros de configuració para realizar el tuning.
    :return ResultGrid | None: Retorna los resultados del tuning si se realizó con RayTune, en otro caso retorna None.
    """
    # Cargar el modelo
    local_model = YOLO(**model_params)
    # Realizar el tuning
    result_grid = local_model.tune(**tuning_params)
    # Borrar el modelo para asegurarse de guardar memoria y retornar resultados
    del local_model
    return result_grid


def raytune_filtar_metrics(df: DataFrame) -> Series:
    """ Se filtra un Datagrama de resultados de un tuning y se retorna una Serie.

    :param DataFrame df: Datagrama con los resultados del tuning.
    :return Series: Serie con los datos que se solicitan
    """
    # Lista de columnas específicas a incluir
    add_columnas = ["metrics/precision(M)", "metrics/recall(M)", "metrics/mAP50(M)", "metrics/mAP50-95(M)", "trial_id"]
    miss_columnas = ["config/data", "config/epochs"]

    # Filtrar columnas que comienzan con "config/" o están en add_columnas
    columnas_deseadas = [col for col in df.columns if (col.startswith("config/") or col in add_columnas) and col not in miss_columnas]
    df_filtrado = df[columnas_deseadas]

    # Seleccionar la fila con el máximo "metrics/mAP50(M)" usando idxmax
    idx_max = df_filtrado["metrics/mAP50(M)"].idxmax()
    fila_seleccionada = cast(Series, df_filtrado.loc[idx_max].copy())

    # Calcular F1_score(M) con manejo de división por cero
    precision = cast(float, fila_seleccionada.get("metrics/precision(M)", 0.0))
    recall = cast(float, fila_seleccionada.get("metrics/recall(M)", 0.0))
    fila_seleccionada["metrics/F1_score(M)"] = 2 * (precision * recall) / (precision + recall) if precision + recall > 0.0 else 0.0

    return fila_seleccionada


def raytune_serie_a_diccionario(serie: Series) -> dict[str, Any]:
    """ Convierte una Serie de Pandas en un diccionario estructurado.

    - Los índices que comienzan con "config/" se agrupan en un subdiccionario "config".
    - Los índices que comienzan con "metrics/" se agrupan en un subdiccionario "metrics".
    - Los demás índices se añaden directamente al diccionario principal.

    :param Series serie: La Serie de entrada.
    :return dict[str, Any]: El diccionario estructurado.
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

        if isinstance(indice, str):
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
        else:
            # Añadir directamente al diccionario principal
            resultado[indice] = valor

    # Añadir los subdiccionarios al resultado si no están vacíos
    if config_dict:
        resultado["config"] = config_dict
    if metrics_dict:
        resultado["metrics"] = metrics_dict

    return resultado


def procesar_result_grid(result_grid: ResultGrid) -> dict[str, Any]:
    """ Se procesa la grilla de resultados que retorna el realizar tuning con RayTune.

    :param ResultGrid result_grid: Grilla de resultados obtenida.
    :return dict[str, Any]: Diccionario con los resultadso del tuning.
    """
    # Obtenemos el mejor caso de cada entrenamiento
    all_best_metrics_df = result_grid.get_dataframe(filter_metric="metrics/mAP50(M)", filter_mode="max")

    # Filtramos solo el mejor de ellos y tambien se filtran algunas columnas
    best_metrics_df = raytune_filtar_metrics(all_best_metrics_df)

    # Lo convertimos en dicionario y lo retornamos
    return raytune_serie_a_diccionario(best_metrics_df)


if __name__ == "__main__":
    # Definir los parametros para realizar el tuning
    models = ['yolov8m-seg', 'yolov8l-seg', 'yolov9c-seg', 'yolo11m-seg', 'yolo11l-seg']
    datasets = ['Salmones.yaml']
    optimizers = ['SGD']
    use_ray = True      # Utilizar Raytune como tuner
    use_freeze = False  # Congelar backbone? (Transfer Learning)
    extra_training_params = {"epochs": 50, "iterations": 50, "batch": 8, "single_cls": True, "cos_lr": True}
    # batch cambiado manualmente en caso de requerirse

    # Archivo donde guardar la configuración
    config_file = "tuning_Salmones.json"

    # Crear configuraciones y guardarlos en un archivo
    # tune_dict = create_tune_dict(models, datasets, optimizers, use_ray, use_freeze, extra_training_params)
    # save_tune_config(tune_dict, config_file)

    # Realizar tuning
    run_tuning_file(config_file)
