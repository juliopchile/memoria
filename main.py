from utility_training import create_training_json, train_run, export_experiments, delete_exportations, validate_run
from config import ALL_MODELS


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
    export_experiments(first_run_json)
    export_experiments(second_run_json)

    #? 4) Realizamos validación para todos los modelos entrenados y exportados.
    validate_run(first_run_json, "training/results_1.csv")
    validate_run(second_run_json, "training/results_2.csv")