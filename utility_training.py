import os
from threading import Thread
from ultralytics import YOLO

from utility_models import get_backbone_path
from config import *

def thread_safe_train(model_path, params):
    """Train on dataset using a new YOLO model instance in a thread-safe manner."""
    local_model = YOLO(model_path)
    local_model.train(**params)
    del local_model


def train_run(dataset_yaml_list, model_name_list, optimizer_list, project_name, freeze_dict, epochs):
    for dataset in dataset_yaml_list:
        dataset_path = os.path.join(YAML_DIRECTORY, dataset)
        dataset_name, _ = os.path.splitext(dataset)
        
        if os.path.isfile(dataset_path):
            for model_name in model_name_list:
                model_path = get_backbone_path(model_name)
                for optimizer in optimizer_list:
                    project_dir= os.path.join(TRAINING_DIRECTORY, project_name)
                    
                    # En caso de ser necesario, cambiar nombre y numero de capas a congelar
                    if freeze_dict[model_name] is True:
                        ast = "*"
                        freeze_num = MODEL_BACKBONE_LAYERS[model_name]
                    else:
                        ast = ""
                        freeze_num = None

                    # Crear el nombre del run y el diccionario de hperparámetros
                    run_name = os.path.join(dataset_name, f"{model_name}{ast}_{optimizer}")
                    train_params = dict(data=dataset_path, epochs=epochs, optimizer=optimizer, freeze=freeze_num,
                                        project=project_dir, name=run_name)
                    
                    # Realizar el enternamiento con los hiperparámetros entregados
                    thread_safe_train(model_path, train_params)


if __name__ == "__main__":
    dataset_yaml_list=['Deepfish.yaml', 'Deepfish_LO.yaml']
    optimizers = ["SGD", "AdamW"]
    project_name = "first_run"
    epochs = 80

    #? 1) Realizamos un entrenamiento con todo en default y opt = [SGD, AdamW]
    freeze_dict = {'yolov8n-seg': False, 'yolov8s-seg': False, 'yolov8m-seg': False, 'yolov8l-seg': False,
                   'yolov8x-seg': False, 'yolov9c-seg': False, 'yolov9e-seg': False, 'yolo11n-seg': False,
                   'yolo11s-seg': False, 'yolo11m-seg': False, 'yolo11l-seg': False, 'yolo11x-seg': False}
    train_run(dataset_yaml_list=dataset_yaml_list, model_name_list=ALL_MODELS, optimizer_list=optimizers,
              project_name=project_name , freeze_dict=freeze_dict, epochs=epochs)

    #? 2) Luego lo mismo pero todos los entrenamientos son tranfer learning (congelar backbone)
    freeze_dict = {'yolov8n-seg': True, 'yolov8s-seg': True, 'yolov8m-seg': True, 'yolov8l-seg': True,
                   'yolov8x-seg': True, 'yolov9c-seg': True, 'yolov9e-seg': True, 'yolo11n-seg': True,
                   'yolo11s-seg': True, 'yolo11m-seg': True, 'yolo11l-seg': True, 'yolo11x-seg': True}
    train_run(dataset_yaml_list=dataset_yaml_list, model_name_list=ALL_MODELS, optimizer_list=optimizers,
              project_name=project_name , freeze_dict=freeze_dict, epochs=epochs)
