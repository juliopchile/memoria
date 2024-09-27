import json
import os
from ultralytics import YOLO, settings
from utility_models import get_backbone_path, ALL_MODELS
from ray import tune


# Turn DVC y wandb false para que no molesten en el entrenamiento
settings.update({'dvc': False, 'wandb': False})

# Diccionario de acotamientos
ACOTAMIENTOS = {
    'lr0': (0.0005, 0.01),
    'lrf': (0.01, 0.5),
    'momentum': (0.6, 0.98),
    'weight_decay': (0.0, 0.001),
    'warmup_epochs': (0.0, 5.0),
    'warmup_momentum': (0.0, 0.95),
}

# Definición del espacio de búsqueda para el primer entrenamiento con Ray Tune
SEARCH_SPACE_DICT = {
    "AdamW": {
        'lr0': tune.uniform(0.0005, 0.002),
        'lrf': tune.uniform(0.01, 0.5),  # final OneCycleLR learning rate (lr0 * lrf)
        'momentum': tune.uniform(0.6, 0.98),  # SGD momentum/Adam beta1
        'weight_decay': tune.uniform(0.0, 0.001),  # optimizer weight decay
        'warmup_epochs': tune.uniform(0.0, 5.0),  # warmup epochs
        'warmup_momentum': tune.uniform(0.0, 0.95),  # warmup initial momentum
    },
    "SGD": {
        'lr0': tune.uniform(0.001, 0.01),
        'lrf': tune.uniform(0.01, 0.5),  # final OneCycleLR learning rate (lr0 * lrf)
        'momentum': tune.uniform(0.6, 0.98),  # SGD momentum/Adam beta1
        'weight_decay': tune.uniform(0.0, 0.001),  # optimizer weight decay
        'warmup_epochs': tune.uniform(0.0, 5.0),  # warmup epochs
        'warmup_momentum': tune.uniform(0.0, 0.95),  # warmup initial momentum
    }
}

def train_ray_tune(iterations: int, epochs: int) -> None:
    """
    Realiza la búsqueda y ajuste de hiperparámetros usando Ray Tune para entrenar varios modelos YOLO 
    con diferentes conjuntos de datos y optimizadores.

    Parámetros:
    -----------
    iterations : int
        Número de iteraciones para el ajuste de hiperparámetros por Ray Tune.
    epochs : int
        Número de épocas para entrenar cada modelo.

    Descripción:
    ------------
    Esta función realiza los siguientes pasos para cada combinación de modelo y conjunto de datos:
        1. Define el espacio de búsqueda de hiperparámetros para los optimizadores AdamW y SGD.
        2. Carga cada modelo YOLO especificado en `ALL_MODELS`.
        3. Itera sobre los archivos de configuración YAML en el directorio `datasets_yaml` para cargar los conjuntos de datos.
        4. Para cada combinación de optimizador y modelo, realiza el ajuste de hiperparámetros usando Ray Tune.
        5. Guarda los resultados de la búsqueda de hiperparámetros.

    Solo se consideran los archivos YAML que no contienen las palabras "export", "Shiny" o "Salmones" en su nombre.

    La función también asigna parámetros adicionales específicos para el modelo `yolov9e-seg`.
    """

    datasets_yaml_dir = os.path.abspath("datasets_yaml")

    for model_name in ALL_MODELS:
        for dataset_yaml in os.listdir(datasets_yaml_dir):
            # Ignorar archivos YAML no relevantes
            if "export" in dataset_yaml or "Shiny" in dataset_yaml or "Salmones" in dataset_yaml:
                continue

            data_yaml = os.path.join(datasets_yaml_dir, dataset_yaml)

            for optimizer in ["AdamW", "SGD"]:
                # Cargar modelo
                model = YOLO(get_backbone_path(model_name), task="segment")

                # Definir nombre del experimento
                name = os.path.splitext(dataset_yaml)[0] + "_" + model_name + "_" + optimizer

                # Parámetros adicionales de entrenamiento específicos para yolov9e-seg
                if model_name in ['yolov9e-seg']:
                    train_params = {"single_cls": False, "cos_lr": True, "freeze": 30}
                else:
                    train_params = {"single_cls": False, "cos_lr": True}

                # Ajustar hiperparámetros con Ray Tune
                result_grid = model.tune(data=data_yaml, iterations=iterations, epochs=epochs, optimizer=optimizer,
                                         space=SEARCH_SPACE_DICT[optimizer], gpu_per_trial=1, use_ray=True, **train_params)

                del model
                del result_grid


def train_tune(search_spaces_dict: dict[str, dict], state_json_path: str, iterations: int, epochs: int, raytune=False) -> None:
    """
    Realiza el entrenamiento de modelos utilizando los mejores hiperparámetros obtenidos en `search_spaces_dict`.

    Parámetros:
    -----------
    search_spaces_dict : dict[str, dict]
        Diccionario que contiene los espacios de búsqueda de hiperparámetros para cada `tune`.
    state_json_path : str
        Ruta al archivo JSON que contiene los estados del entrenamiento de cada `tune`.
    iterations : int
        Número de iteraciones para el ajuste de hiperparámetros.
    epochs : int
        Número de épocas para entrenar cada modelo.

    Descripción:
    ------------
    La función sigue los siguientes pasos:
        1. Carga el estado de cada `tune` desde el archivo JSON proporcionado.
        2. Itera sobre cada `tune` en `search_spaces_dict` y verifica su estado.
        3. Para cada `tune` cuyo estado sea 0 (no entrenado), separa el nombre del experimento en partes
           (dataset, modelo, optimizador), carga el modelo y realiza el ajuste de hiperparámetros.
        4. Actualiza el estado a 1 (entrenado) y guarda el estado actualizado en el archivo JSON.
    """
    # Cargar los estados del archivo JSON
    estados = cargar_estado(state_json_path)

    # Cargar los datos guardados en el diccionario
    for tune_number, contenido in search_spaces_dict.items():
        estado_tune = estados.get(tune_number, {}).get('state')
        nombre_tune = contenido.get('name', tune_number) # {dataset}_{model_name}_{opt}
        search_space = contenido.get('config', {})

        if estado_tune == 0:
            # Separar el nombre en partes
            nombre_partes = nombre_tune.split('_')
            dataset_name = '_'.join(nombre_partes[:-2])
            model_name = nombre_partes[-2]
            optimizer = nombre_partes[-1]

            # Obtener el path del dataset
            datasets_yaml_dir = os.path.abspath("datasets_yaml")
            data_yaml = os.path.join(datasets_yaml_dir, f"{dataset_name}.yaml")
            
            # Congelar pesos en caso de ser yolov9e-seg
            if model_name in ['yolov9e-seg']:
                train_params = {"single_cls": False, "cos_lr": False, "freeze": 30}
            else:
                train_params = {"single_cls": False, "cos_lr": False}

            # Cargar modelo
            model = YOLO(get_backbone_path(model_name), task="segment")

            # Ajustar hiperparámetros
            result_grid = model.tune(data=data_yaml, iterations=iterations, epochs=epochs, optimizer=optimizer,
                                     space=search_space, use_ray=raytune, **train_params)

            del model
            del result_grid
            
            # Actualizar el estado a 1 y guardar en el archivo JSON
            estados[tune_number]['state'] = 1
            guardar_estado(state_json_path, estados)


def leer_resultados_raytune_para_tune(ruta_json: str) -> dict:
    """
    Lee un archivo JSON que contiene las configuraciones de los dos mejores experimentos con Raytune
    y crea un nuevo diccionario con los rangos de hiper-parámetros especificados para cada tune, para
    el entrenamiento con Tune, ajustados con una holgura del 10% y acotados según los rangos proporcionados.

    Parámetros:
    -----------
    ruta_json : str
        Ruta al archivo JSON que contiene la información de los experimentos.
    
    Retorna:
    --------
    dict
        Diccionario con los rangos de parámetros para cada tune.
    """
    with open(ruta_json, 'r') as archivo:
        datos = json.load(archivo)
    
    nuevo_diccionario = {}

    for tune_number, contenido in datos.items():
        nombre_tune = contenido.get('name', tune_number)
        config1 = contenido.get('config1', {})
        config2 = contenido.get('config2', {})

        # Crear un diccionario para los rangos de parámetros con holgura
        rango_config = {
            parametro: calcular_holgura(config1.get(parametro, 0), config2.get(parametro, 0), min_val, max_val)
            for parametro, (min_val, max_val) in ACOTAMIENTOS.items()
        }

        nuevo_diccionario[tune_number] = {'name': nombre_tune, 'config': rango_config}

    return nuevo_diccionario


def leer_resultados_raytune_para_raytune(ruta_json: str) -> dict:
    """
    Lee un archivo JSON que contiene las configuraciones de los dos mejores experimentos con Raytune
    y crea un nuevo diccionario con los rangos de hiper-parámetros especificados para cada tune, para
    el segundo entrenamiento con Raytune, ajustados con una holgura del 10% y acotados según los rangos proporcionados.

    Parámetros:
    -----------
    ruta_json : str
        Ruta al archivo JSON que contiene la información de los experimentos.
    
    Retorna:
    --------
    dict
        Diccionario con los rangos de parámetros para cada tune.
    """
    with open(ruta_json, 'r') as archivo:
        datos = json.load(archivo)
    
    nuevo_diccionario = {}

    for tune_number, contenido in datos.items():
        nombre_tune = contenido.get('name', tune_number)
        config1 = contenido.get('config1', {})
        config2 = contenido.get('config2', {})

        # Crear un diccionario para los rangos de parámetros con holgura
        rango_config = {
            parametro: tune.uniform(*calcular_holgura(config1.get(parametro, 0), config2.get(parametro, 0), min_val, max_val))
            for parametro, (min_val, max_val) in ACOTAMIENTOS.items()
        }

        nuevo_diccionario[tune_number] = {'name': nombre_tune, 'config': rango_config}

    return nuevo_diccionario


def calcular_holgura(valor1, valor2, min_val, max_val):
    """
    Calcula el rango con una holgura del 10% basada en el mínimo y máximo de dos valores,
    acotando el resultado dentro de los límites proporcionados.

    Parámetros:
    -----------
    valor1 : float
        Primer valor a comparar.
    valor2 : float
        Segundo valor a comparar.
    min_val : float
        Valor mínimo permitido para el rango.
    max_val : float
        Valor máximo permitido para el rango.
    
    Retorna:
    --------
    tuple
        Rango ajustado con un 10% de holgura, acotado entre min_val y max_val.
    """
    minimo = min(valor1, valor2)
    maximo = max(valor1, valor2)
    rango_ampliado_min = minimo - 0.10 * abs(minimo)
    rango_ampliado_max = maximo + 0.10 * abs(maximo)

    rango_ampliado_min = max(rango_ampliado_min, min_val)
    rango_ampliado_max = min(rango_ampliado_max, max_val)

    return rango_ampliado_min, rango_ampliado_max


def inicializar_estados(diccionario: dict, ruta_salida: str) -> None:
    """
    Guarda en un archivo JSON solo los nombres y estados de cada 'tune' a partir del diccionario proporcionado.

    Parámetros:
    -----------
    diccionario : dict
        Diccionario con la información de los experimentos.
    ruta_salida : str
        Ruta donde se guardará el archivo JSON de salida.
    """
    resultado = {k: {'name': v['name'], 'state': 0} for k, v in diccionario.items()}
    with open(ruta_salida, 'w') as archivo_salida:
        json.dump(resultado, archivo_salida, indent=4)


def cargar_estado(state_json_path: str) -> dict:
    """Carga el contenido del archivo de estado JSON."""
    with open(state_json_path, 'r') as archivo:
        return json.load(archivo)


def guardar_estado(state_json_path: str, data: dict) -> None:
    """Guarda el contenido modificado en el archivo de estado JSON."""
    with open(state_json_path, 'w') as archivo:
        json.dump(data, archivo, indent=4)


if __name__ == "__main__":
    #? Entrenar utilizando Raytune
    # train_ray_tune(iterations=20, epochs=40)
    # Guardar resultados de Raytune con el notebook check_raytune_results.ipynb
    
    #? Cargar mejores hiperparámetros de los entrenamientos con Raytune.
    raytune_results = "resultados_raytune_deepfish_1.json"
    search_spaces_dict_tune = leer_resultados_raytune_para_tune(raytune_results)
    search_spaces_dict_raytune = leer_resultados_raytune_para_raytune(raytune_results)
    
    #? Inicializar el archivo JSON de estado de entrenamiento con Tune
    tune_training_state = "tune_training_state_deepfish_1.json"
    #inicializar_estados(search_spaces_dict, tune_training_state)    # Util para parar entrenamiento y continuar luego
    
    #? Segunda busqueda de hiperparámetros (Con o sin raytune)
    # train_tune(search_spaces_dict_tune, tune_training_state, 10, 40)
    train_tune(search_spaces_dict_raytune, tune_training_state, 10, 40, True)
