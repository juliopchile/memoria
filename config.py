# Este archivo contiene constantes utilizadas para organizar los Path y nombres de algunos archivos.
import os

# ? CONFIGURACIÓN DE DATASETS

# Path donde guardar todos los datasets.
DATASETS_COCO = "datasets_coco"             # Path temporal donde guardar datasets descargados desde Roboflow.
DATASETS_YOLO = "datasets_yolo"             # Path donde guardar los datasets de segmentación ya procesados.
YAML_DIRECTORY = "datasets_yaml"            # Path donde se encuentran los archivos yaml de cada dataset.

# Diccionario usado para descargar datasets desde Roboflow.
DATASETS_ROBOFLOW_LINKS = {
    "Deepfish": dict(workspace="memristor", project="deepfish-segmentation-ocdlj", version=3, name="Deepfish"),
    "Deepfish_LO": dict(workspace="memristor", project="deepfish-segmentation-ocdlj", version=4, name="Deepfish_LO")
}

# Lista de archivos YAML de los datasets a entrenar (ordenados!!).
DATASETS_YAML_LIST = ['Deepfish.yaml', 'Deepfish_LO.yaml', 'Salmons.yaml', 'Salmons_LO.yaml']


# ? CONFIGURACIÓN DE MODELOS

# Nombres de todos los modelos YOLO utilizados.
ALL_MODELS = ['yolov8n-seg', 'yolov8s-seg', 'yolov8m-seg', 'yolov8l-seg', 'yolov8x-seg', 'yolov9c-seg', 'yolov9e-seg',
              'yolo11n-seg', 'yolo11s-seg', 'yolo11m-seg', 'yolo11l-seg', 'yolo11x-seg']

# Dirección donde guardar los modelos YOLO sin entrenar.
BACKBONES_DIR = os.path.join('models', 'backbone')

# ? CASOS A HACER TUNING (BUSQUEDA DE HIPERPARÁMETROS)
# Lista de nombres de casos, para iterar sobre ellos y saber si saltarlos o no.
TUNING_NAMES_LIST_DEEPFISH = [
    "Deepfish_yolov8n-seg_AdamW",
    "Deepfish_yolov8n-seg_SGD",
    "Deepfish_LO_yolov8n-seg_AdamW",
    "Deepfish_LO_yolov8n-seg_SGD",
    "Deepfish_yolov8s-seg_AdamW",
    "Deepfish_yolov8s-seg_SGD",
    "Deepfish_LO_yolov8s-seg_AdamW",
    "Deepfish_LO_yolov8s-seg_SGD",
    "Deepfish_yolov8m-seg_AdamW",
    "Deepfish_yolov8m-seg_SGD",
    "Deepfish_LO_yolov8m-seg_AdamW",
    "Deepfish_LO_yolov8m-seg_SGD",
    "Deepfish_yolov8l-seg_AdamW",
    "Deepfish_yolov8l-seg_SGD",
    "Deepfish_LO_yolov8l-seg_AdamW",
    "Deepfish_LO_yolov8l-seg_SGD",
    "Deepfish_yolov8x-seg_AdamW",
    "Deepfish_yolov8x-seg_SGD",
    "Deepfish_LO_yolov8x-seg_AdamW",
    "Deepfish_LO_yolov8x-seg_SGD",
    "Deepfish_yolov9c-seg_AdamW",
    "Deepfish_yolov9c-seg_SGD",
    "Deepfish_LO_yolov9c-seg_AdamW",
    "Deepfish_LO_yolov9c-seg_SGD",
    "Deepfish_yolov9e-seg_AdamW",
    "Deepfish_yolov9e-seg_SGD",
    "Deepfish_LO_yolov9e-seg_AdamW",
    "Deepfish_LO_yolov9e-seg_SGD",
    "Deepfish_yolo11n-seg_AdamW",
    "Deepfish_yolo11n-seg_SGD",
    "Deepfish_LO_yolo11n-seg_AdamW",
    "Deepfish_LO_yolo11n-seg_SGD",
    "Deepfish_yolo11s-seg_AdamW",
    "Deepfish_yolo11s-seg_SGD",
    "Deepfish_LO_yolo11s-seg_AdamW",
    "Deepfish_LO_yolo11s-seg_SGD",
    "Deepfish_yolo11m-seg_AdamW",
    "Deepfish_yolo11m-seg_SGD",
    "Deepfish_LO_yolo11m-seg_AdamW",
    "Deepfish_LO_yolo11m-seg_SGD",
    "Deepfish_yolo11l-seg_AdamW",
    "Deepfish_yolo11l-seg_SGD",
    "Deepfish_LO_yolo11l-seg_AdamW",
    "Deepfish_LO_yolo11l-seg_SGD",
    "Deepfish_yolo11x-seg_AdamW",
    "Deepfish_yolo11x-seg_SGD",
    "Deepfish_LO_yolo11x-seg_AdamW",
    "Deepfish_LO_yolo11x-seg_SGD"
]