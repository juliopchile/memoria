from utility_training import thread_safe_train
from utility_models import get_backbone_path
MODEL_BACKBONE_LAYERS = {'yolov8n-seg': 10, 'yolov8s-seg': 10, 'yolov8m-seg': 10, 'yolov8l-seg': 10, 'yolov8x-seg': 10,
                         'yolov9c-seg': 10, 'yolov9e-seg': 30, 'yolo11n-seg': 11, 'yolo11s-seg': 11, 'yolo11m-seg': 11,
                         'yolo11l-seg': 11, 'yolo11x-seg': 11}
ALL_MODELS = ['yolov8n-seg', 'yolov8s-seg', 'yolov8m-seg', 'yolov8l-seg', 'yolov8x-seg', 'yolov9c-seg', 'yolov9e-seg',
              'yolo11n-seg', 'yolo11s-seg', 'yolo11m-seg', 'yolo11l-seg', 'yolo11x-seg']

if __name__ == "__main__":
    params = {"data": "datasets_yaml/Deepfish.yaml", "epochs": 3, "batch": 128, "optimizer": "SGD", "verbose": False}
    use_freeze = True
    for model in ['yolo11n-seg']:
        for batch in [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96]:
            try:
                model_path = get_backbone_path(model)
                if use_freeze:
                    params.update(freeze=MODEL_BACKBONE_LAYERS[model], batch=batch)
                thread_safe_train(model_path, params)
            except:
                pass
