# Este archivo contiene constantes utilizadas para organizar los Path y nombres de algunos archivos.
import os
from ray import tune

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

# ? CASOS A HACER TUNING (BUSQUEDA DE HIPERPARÁMETROS)
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