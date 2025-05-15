import os
from typing import Dict, List
from ultralytics import YOLO

from config import ALL_MODELS, BACKBONES_DIR

def get_backbone_path(model_name: str):
    """ Retorna la dirección de los pesos de un modelo dado su nombre.
    Los modelos base están guardados en un directorio para su mejor organización y uso.

    :param str model_name: Nombre del modelo.
    :return: La dirección a los pesos del modelo en el directorio 'backbone'.

    Examples:
    ----
        >>> get_backbone_path("yolov8l-seg")
        "models/backbone/yolov8l-seg.pt"
    """
    return os.path.join(BACKBONES_DIR, model_name + ".pt")


def download_models(models_to_download: List[str] = None):
    """ Descarga los modelos especificados en la lista. Sin lista se descargan todos los modelos.

    :param (List[str], Optional) models_to_download: Lista de los nombres de modelos a descargar. Por defecto es None.
    """
    os.makedirs(BACKBONES_DIR, exist_ok=True)
    if models_to_download is None:
        models_to_download = ALL_MODELS
    for model in models_to_download:
        model_path = get_backbone_path(model_name=model)
        model = YOLO(model=model_path)
        del model


def export_to_onnx(model_path: str, extra_params: Dict[str, str]):
    """ Exporta un modelo YOLO en formato Pytorch al formato ONNX.

    :param str model_path: Dirección de los pesos del modelo a exportar.
    :param Dict[str, str] extra_params: Parametros extras de configuración.
    """
    model = YOLO(model=model_path)
    model.export(format="onnx", **extra_params)


def export_to_tensor_rt(model_path: str, extra_params: Dict[str, str]):
    """ Exporta un modelo YOLO en formato Pytorch al formato TensorRT.

    :param str model_path: Dirección de los pesos del modelo a exportar.
    :param Dict[str, str] extra_params: Parametros extras de configuración.
    """
    model = YOLO(model=model_path)
    model.export(format="engine", **extra_params)


if __name__ == "__main__":
    # Descargar los modelos a utilizar
    download_models()
