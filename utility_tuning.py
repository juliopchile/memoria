import os
import json
import numpy as np
from ultralytics import YOLO, settings
from utility_models import get_backbone_path
from config import DATASETS_YAML_LIST, ALL_MODELS, TUNING_NAMES_LIST_DEEPFISH, MODEL_BACKBONE_LAYERS
from ray import tune

SEARCH_SPACES = {
        False: {
                'lr0': (1e-5, 1e-1),
                'lrf': (0.01, 0.5),
                'momentum': (0.6, 0.98),
                'weight_decay': (0.0, 0.001),
                'warmup_epochs': (0.0, 5.0),
                'warmup_momentum': (0.0, 0.95)
        },
        True: {
                'lr0': tune.uniform(1e-5, 1e-1),
                'lrf': tune.uniform(0.01, 0.5),
                'momentum': tune.uniform(0.6, 0.98),
                'weight_decay': tune.uniform(0.0, 0.001),
                'warmup_epochs': tune.uniform(0.0, 5.0),
                'warmup_momentum': tune.uniform(0.0, 0.95)
        }
}


def create_tune_dict(models, datasets, optimizers, use_ray=False, use_freeze=False):
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
                model = model_name
                tuning_params = {"data" :data_yaml, "optimizer": optimizer}
                
                # Guardarlos en un diccionario por caso
                tune_dict[caso] = {"model": model, "tuning_params": tuning_params, "done": False,
                                   "use_ray": use_ray, "use_freeze": use_freeze}

    return tune_dict

def do_tuning(values, epochs, iterations):
    # Cargar parámetros del diccionario
    model_name = values["model"]
    use_ray = values["use_ray"]
    data = values["tuning_params"]["data"]
    optimizer = values["tuning_params"]["optimizer"]
    
    # Actualizar los parámetros de entrenamiento
    train_params = {"space": SEARCH_SPACES[use_ray], "epochs": epochs, "iterations": iterations,
                    "use_ray": use_ray, "data": data, "optimizer": optimizer, "batch": 0.7,
                    "single_cls": True, "cos_lr": True}
    if values["use_freeze"]:
        train_params.update(freeze=MODEL_BACKBONE_LAYERS[model_name])
    if use_ray:
        train_params.update(gpu_per_trial=1)

    # Cargar modelo
    model = YOLO(get_backbone_path(model_name), task="segment")
    
    # Realizar tuning
    result_grid = model.tune(**train_params)
        

def run_tuning_file(json_file, epochs, iterations):
    # Cargar el archivo de configuración para el tuning
    tune_dict = load_tune_config(json_file)

    for case, values in tune_dict.items():
        if values["done"]:
            continue

        do_tuning(values, epochs, iterations)
        
        tune_dict[case]["done"] = True
        save_tune_config(tune_dict, json_file)


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


if __name__ == "__main__":
    models = ['yolov8l-seg', 'yolov8x-seg', 'yolov9e-seg', 'yolo11l-seg', 'yolo11x-seg']
    datasets = ['Deepfish.yaml']
    optimizers = ['SGD']
    epochs = 80
    iterations = 50
    tuning_config_deepfish = "tuning_Deepfish.json"

    # Crear diccionario de configuración para los casos de 
    #tune_dict = create_tune_dict(models, datasets, optimizers, True, False)
    #save_tune_config(tune_dict, tuning_config_deepfish)
    
    # Realizar tuning
    run_tuning_file(tuning_config_deepfish, epochs, iterations)
    