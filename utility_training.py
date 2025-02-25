import os
import json
import copy
import pandas as pd
import numpy as np
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from ultralytics import YOLO

from utility_models import get_backbone_path, export_to_tensor_rt
from utility_datasets import get_export_yaml_path
from config import *


def thread_safe_train(model_path, params):
    """
    Entrena un modelo YOLO en un dataset especificado utilizando una instancia local del modelo de manera segura para hilos.
    
    Parámetros:
    -----------
    model_path : str
        Ruta del modelo pre-entrenado o backbone descargado.
    params : dict
        Diccionario con los parámetros e hiperparámetros de entrenamiento, el cual puede incluir:
          - data: ruta al archivo YAML del dataset.
          - optimizer: optimizador a utilizar (por ejemplo, "SGD" o "AdamW").
          - freeze: número de capas a congelar o None si no se congela el backbone.
          - project: directorio del proyecto para almacenar los resultados.
          - name: nombre del run de entrenamiento.
          - otros parámetros adicionales (por ejemplo, epochs, batch, etc.)
    
    Retorna:
    --------
    None
    """
    local_model = YOLO(model_path)
    local_model.train(**params)
    del local_model


def train_run(config_file, max_concurrent_threads=1):
    """
    Ejecuta entrenamientos de modelos basados en las configuraciones definidas en un archivo JSON, 
    limitando el número de entrenamientos concurrentes a 'max_concurrent_threads'.
    
    Para cada configuración de entrenamiento en el archivo JSON que no esté marcada como completada 
    ("done": False), se envía una tarea a un ThreadPoolExecutor con límite de hilos concurrentes. Una vez 
    finalizado el entrenamiento de una tarea, se actualiza su estado a True en el diccionario y se guarda 
    la actualización en el archivo JSON.
    
    Parámetros:
    -----------
    config_file : str
        Ruta al archivo JSON que contiene las configuraciones de entrenamiento. Cada entrada en el JSON debe tener la
        siguiente estructura:
        
            {
                "run_name": {
                    "model_name": <nombre del modelo>,
                    "hyperparam": { ... parámetros de entrenamiento ... },
                    "done": false
                },
                ...
            }
    
    max_concurrent_threads : int, opcional (por defecto = 1)
        Número máximo de entrenamientos a ejecutar de forma concurrente.
    
    Retorna:
    --------
    None
    """
    # Cargar configuraciones de entrenamiento desde el JSON
    with open(config_file, "r") as f:
        training_dict = json.load(f)
    
    # Lock para asegurar actualizaciones seguras al archivo JSON desde múltiples hilos
    json_lock = Lock()
    
    def training_wrapper(key, config):
        """
        Función wrapper que ejecuta el entrenamiento de un run y actualiza su estado en el JSON.
        
        Parámetros:
        -----------
        key : str
            Clave identificadora única para el run.
        config : dict
            Diccionario con la configuración del entrenamiento (incluye model_name, hyperparam y done).
        """
        model_name = config.get("model_name")
        # Obtener la ruta del modelo: se descarga o se recupera según la implementación de get_backbone_path
        model_path = get_backbone_path(model_name)
        # Recuperar los hiperparámetros de entrenamiento
        train_params = config.get("hyperparams", {})

        try:
            # Ejecutar el entrenamiento
            thread_safe_train(model_path, train_params)
            # Al finalizar, marcar el run como completado y actualizar el archivo JSON
            with json_lock:
                training_dict[key]["done"] = True
                with open(config_file, "w") as f_w:
                    json.dump(training_dict, f_w, indent=4)
            print(f"Entrenamiento completado para: {key}")
        except Exception as e:
            print(f"El entrenamiento para {key} falló con error: {e}")
    
    # Crear un ThreadPoolExecutor que limite el número de tareas concurrentes
    with ThreadPoolExecutor(max_workers=max_concurrent_threads) as executor:
        futures = []
        for key, config in training_dict.items():
            if not config.get("done", False):
                futures.append(executor.submit(training_wrapper, key, config))
        
        # Esperar a que todas las tareas finalicen
        for future in as_completed(futures):
            # Si deseas capturar excepciones globales podrías hacerlo aquí:
            try:
                future.result()
            except Exception as exc:
                print(f"Una tarea falló: {exc}")

    print(f"Todos los entrenamientos de {config_file} han sido completados.")


def create_training_json(dataset_yaml_list, model_name_list, optimizer_list, project_name, freeze_dict, json_file, extra_params):
    """
    Crea un archivo JSON con configuraciones de entrenamiento generadas a partir de listas de parámetros.
    
    Para cada combinación de dataset, modelo y optimizador, se genera una configuración que incluye:
      - La ruta al dataset YAML.
      - Los parámetros de entrenamiento como: optimizador, cantidad de capas a congelar (si aplica),
        directorio del proyecto, nombre del run, y parámetros adicionales.
    
    Parámetros:
    -----------
    dataset_yaml_list : list
        Lista de nombres de archivos YAML que definen los datasets (por ejemplo, ['Deepfish.yaml', 'Deepfish_LO.yaml']).
    model_name_list : list
        Lista de nombres de modelos (ejemplo: ['yolov8n-seg', 'yolov9c-seg']) que se utilizarán.
    optimizer_list : list
        Lista de optimizadores a usar (por ejemplo, ["SGD", "AdamW"]).
    project_name : str
        Nombre del proyecto (usado para definir el directorio donde se guardarán los resultados).
    freeze_dict : dict
        Diccionario que indica si se debe aplicar congelamiento del backbone para cada modelo.
        Cada clave es el nombre de un modelo y el valor es un booleano (True para congelar, False para no congelar).
    json_file : str
        Ruta del archivo JSON en el que se guardarán las configuraciones de entrenamiento.
    extra_params : dict
        Diccionario con otros parámetros adicionales de entrenamiento (por ejemplo, {"epochs": 80, "batch": 8}).
    
    Retorna:
    --------
    None
    """
    training_dict = {}
    for dataset in dataset_yaml_list:
        dataset_path = os.path.join(YAML_DIRECTORY, dataset)
        dataset_name, _ = os.path.splitext(dataset)
        
        if os.path.isfile(dataset_path):
            for model_name in model_name_list:
                for optimizer in optimizer_list:
                    project_dir = os.path.join(TRAINING_DIRECTORY, project_name)

                    # En caso de ser necesario, ajustar el nombre y número de capas a congelar
                    if freeze_dict.get(model_name, False) is True:
                        ast = "*"
                        freeze_num = MODEL_BACKBONE_LAYERS.get(model_name)
                    else:
                        ast = ""
                        freeze_num = None
                        
                    if optimizer in ["AdamW"]:
                        lr0 = 0.001
                    else:
                        lr0 = 0.005

                    # Crear el nombre del run y definir el diccionario de hiperparámetros
                    run_name = os.path.join(dataset_name, f"{model_name}{ast}_{optimizer}")
                    train_params = {"data": dataset_path, "optimizer": optimizer, "freeze": freeze_num, "lr0": lr0,
                                    "project": project_dir, "name": run_name, **extra_params}
                    
                    # Agregar la configuración al diccionario general de entrenamiento
                    training_dict[run_name] = {"model_name": model_name, "hyperparams": train_params,
                                               "done": False, "trt32": False, "trt16": False, "trt8": False}
    
    # Asegurarse de que el directorio donde se guardará el archivo JSON existe
    json_dir = os.path.dirname(json_file)
    if json_dir and not os.path.exists(json_dir):
        os.makedirs(json_dir, exist_ok=True)
    
    # Guardar el diccionario de configuraciones en un archivo JSON
    with open(json_file, "w") as f:
        json.dump(training_dict, f, indent=4)
    
    print(f"Training JSON guardado en {json_file}")


def rename_file(original_path: str, new_name: str) -> str:
    """
    Renombra un archivo manteniendo su directorio original.

    Args:
        original_path (str): Ruta completa del archivo original.
        new_name (str): Nuevo nombre del archivo (sin cambiar el directorio).

    Returns:
        str: La ruta completa del archivo renombrado.
    """
    # Obtener el directorio del archivo original
    dir_name = os.path.dirname(original_path)
    # Crear la nueva ruta con el mismo directorio y el nuevo nombre
    new_path = os.path.join(dir_name, new_name)
    # Renombrar el archivo
    os.rename(original_path, new_path)
    return new_path


def update_config_file(config_file: str, training_dict: dict) -> None:
    """
    Actualiza el archivo de configuración con el contenido de training_dict.

    Args:
        config_file (str): Ruta al archivo JSON de configuración.
        training_dict (dict): Diccionario actualizado con la configuración.
    """
    try:
        with open(config_file, "w") as f:
            json.dump(training_dict, f, indent=4)
        print(f"Archivo de configuración '{config_file}' actualizado exitosamente.")
    except Exception as exc:
        print(f"Error al actualizar el archivo de configuración: {exc}")


def process_export(model_weights_path: str,
                   engine_path: str,
                   export_dataset_yaml: str,
                   config_entry: dict,
                   flag_key: str,
                   new_engine_name: str,
                   config_file: str,
                   training_dict: dict,
                   extra_params: dict = None) -> None:
    """
    Intenta exportar el modelo a TensorRT, renombrar el archivo generado y actualizar
    el archivo de configuración JSON en caso de éxito.

    Args:
        model_weights_path (str): Ruta a los pesos del modelo entrenado.
        engine_path (str): Ruta al archivo engine generado (antes de renombrarlo).
        export_dataset_yaml (str): Ruta al archivo YAML del dataset de exportación.
        config_entry (dict): Subdiccionario de configuración del experimento.
        flag_key (str): Clave a actualizar en la configuración ('trt', 'trt16', 'trt8').
        new_engine_name (str): Nuevo nombre para el archivo engine exportado.
        config_file (str): Ruta al archivo JSON de configuración.
        training_dict (dict): Diccionario completo de la configuración.
        extra_params (dict, optional): Parámetros adicionales para export_to_tensor_rt.
                                       Ej: {"half": True} o {"int8": True}.
    """
    try:
        # Construir los parámetros para la exportación
        export_params = {"data": export_dataset_yaml}
        if extra_params:
            export_params.update(extra_params)

        # Ejecutar la exportación a TensorRT
        export_to_tensor_rt(model_weights_path, export_params)
        # Renombrar el archivo engine generado
        rename_file(engine_path, new_engine_name)
        # Actualizar la configuración local del experimento
        config_entry[flag_key] = True
        
        # Actualizar el archivo JSON inmediatamente después de la exportación exitosa
        update_config_file(config_file, training_dict)
        print(f"Exportación exitosa ({flag_key}) -> {new_engine_name}")
    except Exception as exc:
        print(f"Error al exportar con {flag_key}: {exc}")


def export_experiments(config_file: str) -> None:
    """
    Procesa los experimentos definidos en el archivo de configuración JSON.
    Para cada experimento que ya fue entrenado (done=True) y cuyas exportaciones
    aún no se han realizado (trt, trt16, trt8 == False), se intenta exportar el modelo
    a TensorRT en FP32, FP16 e INT8. Tras cada exportación exitosa se actualiza el archivo
    JSON de configuración, de forma que, en caso de fallar alguna exportación, lo realizado
    hasta el momento quede guardado.

    Args:
        config_file (str): Ruta al archivo JSON de configuración.
    """
    # Leer el archivo de configuración
    with open(config_file, "r") as f:
        training_dict = json.load(f)

    # Recorrer cada experimento definido en la configuración
    for key, config in training_dict.items():
        if config.get("done"):
            hyperparams = config.get("hyperparams", {})
            # Construir la ruta de los pesos del modelo entrenado
            model_weights_path = os.path.join(
                hyperparams.get("project", ""),
                hyperparams.get("name", ""),
                "weights",
                "best.pt"
            )
            # La exportación inicialmente genera un archivo 'best.engine'
            engine_path = os.path.join(os.path.dirname(model_weights_path), "best.engine")
            # Obtener el path del archivo YAML de exportación (se le añade el prefijo 'export_')
            export_dataset_yaml = get_export_yaml_path(hyperparams.get("data", ""))

            if os.path.isfile(model_weights_path):
                # Exportar a TensorRT en FP32
                if not config.get("trt32", False):
                    process_export(model_weights_path, engine_path, export_dataset_yaml, config, "trt32",
                                   "best_trt_fp32.engine", config_file, training_dict, extra_params={"batch": 1})
                # Exportar a TensorRT en FP16
                if not config.get("trt16", False):
                    process_export(model_weights_path, engine_path, export_dataset_yaml, config, "trt16",
                                   "best_trt_fp16.engine", config_file, training_dict, extra_params={"half": True, "batch": 1})
                # Exportar a TensorRT en INT8
                if not config.get("trt8", False):
                    process_export(model_weights_path, engine_path, export_dataset_yaml, config, "trt8",
                                   "best_trt_int8.engine", config_file, training_dict, extra_params={"int8": True, "batch": 1})
            else:
                print(f"Los pesos del modelo no existen: {model_weights_path}")


def delete_exportations(config_file):
    with open(config_file, "r") as f:
        training_dict = json.load(f)

    # Recorrer cada experimento definido en la configuración
    for key, config in training_dict.items():
        hyperparams = config["hyperparams"]
        weights_path = os.path.join(hyperparams["project"], hyperparams["name"], "weights")

        model_paths = [
            "best_trt_fp32.engine",
            "best_trt_fp16.engine",
            "best_trt_int8.engine",
            "best.onnx",
            "best.cache"
        ]

        for model_file in model_paths:
            model_path = os.path.join(weights_path, model_file)
            if os.path.exists(model_path):
                try:
                    os.remove(model_path)
                    print(f"Archivo eliminado: {model_path}")
                except Exception as e:
                    print(f"Error al eliminar {model_path}: {e}")
            else:
                print(f"Archivo no encontrado: {model_path}")


def add_f1_scores(metrics: dict) -> dict:
    """Calculate and add F1 scores to a copy of the given metrics dictionary, keeping the desired order."""
    new_metrics = copy.deepcopy(metrics)  # Create a copy to avoid modifying the original

    # Extract precision and recall for (B)
    precision_B = new_metrics.get('metrics/precision(B)', 0)
    recall_B = new_metrics.get('metrics/recall(B)', 0)
    F1_B = 2 * (precision_B * recall_B) / (precision_B + recall_B) if (precision_B + recall_B) > 0 else 0

    # Extract precision and recall for (M)
    precision_M = new_metrics.get('metrics/precision(M)', 0)
    recall_M = new_metrics.get('metrics/recall(M)', 0)
    F1_M = 2 * (precision_M * recall_M) / (precision_M + recall_M) if (precision_M + recall_M) > 0 else 0

    # Create a new dictionary with the desired order
    ordered_metrics = {}
    for key, value in new_metrics.items():
        ordered_metrics[key] = value
        if key == 'metrics/recall(B)':
            ordered_metrics['metrics/F1_score(B)'] = np.float64(F1_B)
        if key == 'metrics/recall(M)':
            ordered_metrics['metrics/F1_score(M)'] = np.float64(F1_M)

    return ordered_metrics


def safe_validate(model_path, extra_config):
    model = YOLO(model_path, task="segment")
    val_metrics = model.val(**extra_config)
    return val_metrics


def validate_experiment(dataframe, parameters):
    model_pt_path = parameters["model_pt_path"]
    validation_params = parameters["validation_params"]
    model_name = parameters["model_name"]
    dataset_name = parameters["dataset_name"]
    optimizer = parameters["optimizer"]
    model_format = parameters["format"]
    
    try:
        val_results = safe_validate(model_pt_path, validation_params)
        data = add_f1_scores(val_results.results_dict) | val_results.speed
        cleaned_data = {key.replace("metrics/", ""): value for key, value in data.items()}
        cleaned_data.update(Model=model_name, Dataset=dataset_name, Optimizer=optimizer, Format=model_format)
        row_data = pd.DataFrame([cleaned_data])
        dataframe = pd.concat([dataframe, row_data], ignore_index=True)
        return dataframe
    except Exception as error:
        print(error)


def validate_run(config_file, results_path):
    with open(config_file, "r") as f:
        config_dict = json.load(f)
        
    dataframe = pd.DataFrame()

    for key, config in config_dict.items():
        dataset_name, temp = os.path.split(key)
        model_name, optimizer = temp.split("_")

        hyperparams = config["hyperparams"]
        done = config["done"]
        trt32 = config["trt32"]
        trt16 = config["trt16"]
        trt8 = config["trt8"]

        # Paths
        dataset_yaml = hyperparams["data"]
        model_pt_path = os.path.join(hyperparams["project"],hyperparams["name"], "weights", "best.pt")
        model_trt32_path = os.path.join(hyperparams["project"],hyperparams["name"], "weights", "best_trt_fp32.engine")
        model_trt16_path = os.path.join(hyperparams["project"],hyperparams["name"], "weights", "best_trt_fp16.engine")
        model_trt8_path = os.path.join(hyperparams["project"],hyperparams["name"], "weights", "best_trt_int8.engine")

        validation_params = {"data": dataset_yaml, "device": "cuda:0", "split": "val"}

        if done:
            parameters = {"model_pt_path": model_pt_path, "validation_params": validation_params, "model_name": model_name,
                          "dataset_name": dataset_name, "optimizer": optimizer, "format": "Pytorch"}
            dataframe = validate_experiment(dataframe, parameters)
        if trt32:
            parameters = {"model_pt_path": model_trt32_path, "validation_params": validation_params, "model_name": model_name,
                          "dataset_name": dataset_name, "optimizer": optimizer, "format": "TensorRT-F32"}
            dataframe = validate_experiment(dataframe, parameters)
        if trt16:
            parameters = {"model_pt_path": model_trt16_path, "validation_params": validation_params, "model_name": model_name,
                          "dataset_name": dataset_name, "optimizer": optimizer, "format": "TensorRT-F16"}
            dataframe = validate_experiment(dataframe, parameters)
        if trt8:
            parameters = {"model_pt_path": model_trt8_path, "validation_params": validation_params, "model_name": model_name,
                          "dataset_name": dataset_name, "optimizer": optimizer, "format": "TensorRT-INT8"}
            dataframe = validate_experiment(dataframe, parameters)

    print("llegué aquí")
    dataframe.to_csv(results_path, index=False)


if __name__ == "__main__":
    dataset_yaml_list=['Deepfish.yaml', 'Deepfish_LO.yaml']
    models_to_use = ALL_MODELS
    optimizers = ["SGD", "AdamW"]
    project_name = "first_run"
    extra_params = {"epochs": 80, "batch": 8}
    first_run_json = "training/run1.json"
    second_run_json = "training/run2.json"
    
    #? 1) Primero creamos los archivos JSON que contienen los parámetros de entrenamiento para cada caso.
    def first_experiment_json():
        #* I) El primer entrenamiento será con todo en default.
        freeze_dict = {'yolov8n-seg': False, 'yolov8s-seg': False, 'yolov8m-seg': False, 'yolov8l-seg': False,
                    'yolov8x-seg': False, 'yolov9c-seg': False, 'yolov9e-seg': False, 'yolo11n-seg': False,
                    'yolo11s-seg': False, 'yolo11m-seg': False, 'yolo11l-seg': False, 'yolo11x-seg': False}
        create_training_json(dataset_yaml_list=dataset_yaml_list, model_name_list=models_to_use, optimizer_list=optimizers,
                            project_name=project_name , freeze_dict=freeze_dict, json_file=first_run_json,
                            extra_params=extra_params)
        
        #* II) El segundo entrenamiento será igual pero con transfer learning (congelar backbone)
        freeze_dict = {'yolov8n-seg': True, 'yolov8s-seg': True, 'yolov8m-seg': True, 'yolov8l-seg': True,
                    'yolov8x-seg': True, 'yolov9c-seg': True, 'yolov9e-seg': True, 'yolo11n-seg': True,
                    'yolo11s-seg': True, 'yolo11m-seg': True, 'yolo11l-seg': True, 'yolo11x-seg': True}
        create_training_json(dataset_yaml_list=dataset_yaml_list, model_name_list=models_to_use, optimizer_list=optimizers,
                            project_name=project_name , freeze_dict=freeze_dict, json_file=second_run_json,
                            extra_params=extra_params)
    # first_experiment_json()

    #? 2) Luego leemos los arhivos JSON y entrenamos acorde
    # train_run(config_file=first_run_json)
    # train_run(config_file=second_run_json)

    #? 3) Exportamos los entrenamientos con TensorRT
    # export_experiments(first_run_json)
    # export_experiments(second_run_json)
    
    #? 4) Realizamos validación para todos los modelos entrenados y exportados.
    # validate_run(first_run_json, "training/results_1.csv")
    # validate_run(second_run_json, "training/results_2.csv")
