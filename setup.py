# Algunas librerias como ray no están instaladas por defecto al momento de instalar Ultralytics
# por lo tanto hay que ejecutar un código que utilice dichas librerías al menos 1 vez para así
# auto instalar estos paquetes con un AutoUpdate que realiza Ultralytics por defecto.

from ultralytics import YOLO
from utility_models import download_models
from utility_datasets import setup_datasets


if __name__ == "__main__":
    # Instalar y configurar el dataset de Deepfish
    setup_datasets()

    # Instalar los modelos YOLO a entrenar
    download_models()

    # Para instalar "ray" simplemente corre un tuning usando el parametro 'use_ray=True'
    model = YOLO("models/backbone/yolo11n-seg.pt")
    model.tune(iterations=5, epochs=10, optimizer="SGD", gpu_per_trial=1, use_ray=True)

    # Para exportar en formato TensorRT sirve con exportar una vez un modelo
    model.export(format="engine")
