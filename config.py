# Este archivo contiene constantes utilizadas para organizar las rutas y nombres de algunos archivos.
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
DATASETS_YAML_LIST = ['Deepfish.yaml', 'Deepfish_LO.yaml', 'Salmones.yaml', 'Salmones_LO.yaml']


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

BEST_SALMONES_TUNES_1 = [
    {"model": "yolov8m-seg",
    "params": {
        'data': 'datasets_yaml/Salmones.yaml',
        "project": "training/last_run",
        "name": "Salmones_1/yolov8m-seg",
        'batch': 8,
        'lr0': 0.007702534660969132,
        'lrf': 0.35073713000901796,
        'momentum': 0.7112252189838066,
        'warmup_epochs': 3.681225153627064,
        'warmup_momentum': 0.16920756330576858,
        'weight_decay': 0.00015870577712311375}},
    {"model": "yolov8l-seg",
    "params": {
        'data': 'datasets_yaml/Salmones.yaml',
        "project": "training/last_run",
        "name": "Salmones_1/yolov8l-seg",
        'batch': 6,
        'lr0': 0.002905898626491907,
        'lrf': 0.2485481132018178,
        'momentum': 0.8457548180437737,
        'warmup_epochs': 1.377510359449845,
        'warmup_momentum': 0.2236374293301819,
        'weight_decay': 0.0009622188516019138}},
    {"model": "yolov9c-seg",
    "params": {
        'data': 'datasets_yaml/Salmones.yaml',
        "project": "training/last_run",
        "name": "Salmones_1/yolov9c-seg",
        'batch': 6,
        'lr0': 0.0028629167336899995,
        'lrf': 0.052013150246882175,
        'momentum': 0.8000156183345473,
        'warmup_epochs': 1.107789080739063,
        'warmup_momentum': 0.10904885449928764,
        'weight_decay': 0.0009926302420513878}},
    {"model": "yolo11m-seg",
    "params": {
        'data': 'datasets_yaml/Salmones.yaml',
        "project": "training/last_run",
        "name": "Salmones_1/yolo11m-seg",
        'batch': 8,
        'lr0': 0.002856214746066463,
        'lrf': 0.22498580864869247,
        'momentum': 0.7371716047880296,
        'warmup_epochs': 1.0256887076360126,
        'warmup_momentum': 0.8250966540076803,
        'weight_decay': 9.448753703957746e-05}},
    {"model": "yolo11l-seg",
    "params": {
        'data': 'datasets_yaml/Salmones.yaml',
        "project": "training/last_run",
        "name": "Salmones_1/yolo11l-seg",
        'batch': 6,
        'lr0': 0.005766237154646074,
        'lrf': 0.2825791252640007,
        'momentum': 0.6433397720791748,
        'warmup_epochs': 0.7079968431936795,
        'warmup_momentum': 0.8812599403496482,
        'weight_decay': 0.0005696977794852056}}
]


BEST_SALMONES_TUNES_2 = [
    {"model": "training/last_run/Salmones_1/yolov8m-seg/weights/last.pt",
    "params": {
        'data': 'datasets_yaml/Salmones.yaml',
        "project": "training/last_run",
        "name": "Salmones_2/yolov8m-seg",
        'batch': 8,
        'lr0': 0.0013864074561750113,
        'lrf': 0.12501920253692927,
        'momentum': 0.8672723726480761,
        'warmup_epochs': 4.54786454694854,
        'warmup_momentum': 0.09862475915803505,
        'weight_decay': 2.542441326859757e-05}},
    {"model": "training/last_run/Salmones_1/yolov8l-seg/weights/last.pt",
    "params": {
        'data': 'datasets_yaml/Salmones.yaml',
        "project": "training/last_run",
        "name": "Salmones_2/yolov8l-seg",
        'batch': 6,
        'lr0': 0.0008479337593161244,
        'lrf': 0.08787365835932086,
        'momentum': 0.8994956461792,
        'warmup_epochs': 3.9643631922356977,
        'warmup_momentum': 0.8734336695277866,
        'weight_decay': 0.00031146533242426154}},
    {"model": "training/last_run/Salmones_1/yolov9c-seg/weights/last.pt",
    "params": {
        'data': 'datasets_yaml/Salmones.yaml',
        "project": "training/last_run",
        "name": "Salmones_2/yolov9c-seg",
        'batch': 6,
        'lr0': 0.0019144328944083887,
        'lrf': 0.28575147488365255,
        'momentum': 0.722743540630149,
        'warmup_epochs': 4.072962395799744,
        'warmup_momentum': 0.11898495785702684,
        'weight_decay': 0.0008287856218042173}},
    {"model": "training/last_run/Salmones_1/yolo11m-seg/weights/last.pt",
    "params": {
        'data': 'datasets_yaml/Salmones.yaml',
        "project": "training/last_run",
        "name": "Salmones_2/yolo11m-seg",
        'batch': 8,
        'lr0': 0.0003293420055192948,
        'lrf': 0.27769080614571845,
        'momentum': 0.7154620164545487,
        'warmup_epochs': 1.8730958158695725,
        'warmup_momentum': 0.8107145640706938,
        'weight_decay': 0.0001297045642978416}},
    {"model": "training/last_run/Salmones_1/yolo11l-seg/weights/last.pt",
    "params": {
        'data': 'datasets_yaml/Salmones.yaml',
        "project": "training/last_run",
        "name": "Salmones_2/yolo11l-seg",
        'batch': 6,
        'lr0': 0.00394618234160895,
        'lrf': 0.2520005043535492,
        'momentum': 0.6642943179374865,
        'warmup_epochs': 2.501726357679993,
        'warmup_momentum': 0.09130526816103376,
        'weight_decay': 0.0005054368055637849}}
]