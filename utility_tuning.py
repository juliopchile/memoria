from ultralytics import YOLO
from utility_models import get_backbone_path, ALL_MODELS

model_name = ALL_MODELS[0]
dataset_yaml = "datasets_yaml/Deepfish.yaml"

model = YOLO(get_backbone_path(model_name))


# Tune hyperparameters for 30 epochs
model.tune(data=dataset_yaml, epochs=30, iterations=10, plots=True, save=True, val=True)