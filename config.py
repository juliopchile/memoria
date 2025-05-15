# Este archivo contiene constantes utilizadas para organizar los Path y nombres de algunos archivos.
import os

# ? CONFIGURACIÓN DE DATASETS

# Path donde guardar todos los datasets.
DATASETS_COCO = "datasets_coco"             # Path temporal donde guardar datasets descargados desde Roboflow.
DATASETS_YOLO = "datasets_yolo"             # Path donde guardar los datasets de segmentación ya procesados.
YAML_DIRECTORY = "datasets_yaml"            # Path donde se encuentran los archivos yaml de cada dataset.
TRAINING_DIRECTORY = "training"             # Path donde se guardan los modelos entrenados.

# Diccionario usado para descargar datasets desde Roboflow.
DATASETS_ROBOFLOW_LINKS = {
    "Deepfish": dict(workspace="memristor", project="deepfish-segmentation-ocdlj", version=10, name="Deepfish"),
    "Deepfish_LO": dict(workspace="memristor", project="deepfish-segmentation-ocdlj", version=9, name="Deepfish_LO")
}

# Lista de archivos YAML de los datasets a entrenar (ordenados!!).
DATASETS_YAML_LIST = ['Deepfish.yaml', 'Deepfish_LO.yaml', 'Salmons.yaml', 'Salmons_LO.yaml']


# ? CONFIGURACIÓN DE MODELOS

# Nombres de todos los modelos YOLO utilizados.
ALL_MODELS = ['yolov8n-seg', 'yolov8s-seg', 'yolov8m-seg', 'yolov8l-seg', 'yolov8x-seg', 'yolov9c-seg', 'yolov9e-seg',
              'yolo11n-seg', 'yolo11s-seg', 'yolo11m-seg', 'yolo11l-seg', 'yolo11x-seg']

MODEL_BACKBONE_LAYERS = {'yolov8n-seg': 10, 'yolov8s-seg': 10, 'yolov8m-seg': 10, 'yolov8l-seg': 10, 'yolov8x-seg': 10,
                         'yolov9c-seg': 10, 'yolov9e-seg': 30, 'yolo11n-seg': 11, 'yolo11s-seg': 11, 'yolo11m-seg': 11,
                         'yolo11l-seg': 11, 'yolo11x-seg': 11}

# Dirección donde guardar los modelos YOLO sin entrenar.
BACKBONES_DIR = os.path.join('models', 'backbone')

# ? MEJORES RESULTADOS DE TUNING

BEST_DEEPFISH_TUNES = [
    {"model": "yolov8l-seg",
     "params": {
         "data": "datasets_yaml/Deepfish.yaml",
         "project": "training/last_run",
         "name": "Deepfish/yolov8l-seg",
         "batch": 8,
         "lr0": 0.0030094320389691723,
         "lrf": 0.021028002627388707,
         "momentum": 0.8176950450060704,
         "warmup_epochs": 1.8128559556430095,
         "warmup_momentum": 0.4900346566989251,
         "weight_decay": 0.0009610879004378389}},
    {"model": "yolov8x-seg",
     "params": {
         "data": "datasets_yaml/Deepfish.yaml",
         "project": "training/last_run",
         "name": "Deepfish/yolov8x-seg",
         "batch": 8,
         "lr0": 0.00618433824460782,
         "lrf": 0.284490917424955,
         "momentum": 0.7470405133584371,
         "warmup_epochs": 1.295316796581329,
         "warmup_momentum": 0.9081113388806424,
         "weight_decay": 0.0009698056459717375}},
    {"model": "yolov9e-seg",
     "params": {
         "data": "datasets_yaml/Deepfish.yaml",
         "project": "training/last_run",
         "name": "Deepfish/yolov9e-seg",
         "batch": 6,
         "lr0": 0.006842631690244204,
         "lrf": 0.11984122900328371,
         "momentum": 0.7221154956852589,
         "warmup_epochs": 0.1917342587590004,
         "warmup_momentum": 0.6496749263513208,
         "weight_decay": 0.0006233910789756508}},
    {"model": "yolo11l-seg",
     "params": {
         "data": "datasets_yaml/Deepfish.yaml",
         "project": "training/last_run",
         "name": "Deepfish/yolo11l-seg",
         "batch": 8,
         "lr0": 0.0035179713013480185,
         "lrf": 0.20811911273983605,
         "momentum": 0.7471144701086997,
         "warmup_epochs": 0.3112294727396592,
         "warmup_momentum": 0.5198800058587464,
         "weight_decay": 0.00046781008785355563}},
    {"model": "yolo11x-seg",
     "params": {
         "data": "datasets_yaml/Deepfish.yaml",
         "project": "training/last_run",
         "name": "Deepfish/yolo11x-seg",
         "batch": 8,
         "lr0": 0.0028529805176369595,
         "lrf": 0.01920377656861268,
         "momentum": 0.7004853813748272,
         "warmup_epochs": 3.98955718449678,
         "warmup_momentum": 0.5421530648686753,
         "weight_decay": 0.0004405648826055806}},
]