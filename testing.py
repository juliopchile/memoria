import pandas as pd
from utility_training import validate_experiment


parameters =  {
# Results 1
"case1" : {"model_pt_path": "training/first_run/Deepfish/yolov8n-seg_SGD/weights/best.pt",
           "validation_params": {"data": "datasets_yaml/Deepfish.yaml", "device": "cuda:0", "split": "val"},
           "model_name": "yolov8n-seg", "dataset_name": "Deepfish", "optimizer": "SGD", "format": "Pytorch"},
"case2" : {"model_pt_path": "training/first_run/Deepfish/yolov8n-seg_AdamW/weights/best_trt_fp32.engine",
           "validation_params": {"data": "datasets_yaml/Deepfish.yaml", "device": "cuda:0", "split": "val"},
           "model_name": "yolov8n-seg", "dataset_name": "Deepfish", "optimizer": "AdamW", "format": "TensorRT-F32"},
# Results 4
"case3" : {"model_pt_path": "training/first_run/Salmones/yolov8n-seg_SGD/weights/best.pt",
           "validation_params": {"data": "datasets_yaml/Salmones.yaml", "device": "cuda:0", "split": "val"},
           "model_name": "yolov8n-seg", "dataset_name": "Salmones", "optimizer": "SGD", "format": "Pytorch"},
"case4" : {"model_pt_path": "training/first_run/Salmones/yolov9e-seg_SGD/weights/best.pt",
           "validation_params": {"data": "datasets_yaml/Salmones.yaml", "device": "cuda:0", "split": "val"},
           "model_name": "yolov9e-seg", "dataset_name": "Salmones", "optimizer": "SGD", "format": "Pytorch"},
"case5" : {"model_pt_path": "training/first_run/Salmones_LO/yolov9e-seg_AdamW/weights/best.pt",
           "validation_params": {"data": "datasets_yaml/Salmones_LO.yaml", "device": "cuda:0", "split": "val"},
           "model_name": "yolov9e-seg", "dataset_name": "Salmones_LO", "optimizer": "AdamW", "format": "Pytorch"},
# Results 5
"case6" : {"model_pt_path": "training/first_run/Salmones/yolov8n-seg*_SGD/weights/best.pt",
           "validation_params": {"data": "datasets_yaml/Salmones.yaml", "device": "cuda:0", "split": "val"},
           "model_name": "yolov8n-seg*", "dataset_name": "Salmones", "optimizer": "SGD", "format": "Pytorch"},
"case7" : {"model_pt_path": "training/first_run/Salmones/yolo11n-seg*_AdamW/weights/best.pt",
           "validation_params": {"data": "datasets_yaml/Salmones.yaml", "device": "cuda:0", "split": "val"},
           "model_name": "yolo11n-seg*", "dataset_name": "Salmones", "optimizer": "AdamW", "format": "Pytorch"},
}

dataframe = pd.DataFrame()
for case, parameter in parameters.items():
    dataframe = validate_experiment(dataframe, parameter)
    dataframe = validate_experiment(dataframe, parameter)

dataframe.to_csv("results_path.csv", index=False)