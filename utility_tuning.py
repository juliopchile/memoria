import json
import os
from ultralytics import YOLO, settings
from utility_models import get_backbone_path, ALL_MODELS
from ray import tune
# Turn DVC y wandb false para que no molesten en el entrenamiento
settings.update({'dvc': False, 'wandb': False})

search_space_raytune = {"AdamW": {'lr0': tune.uniform(0.0005, 0.002),
                          'lrf': tune.uniform(0.01, 0.5),  # final OneCycleLR learning rate (lr0 * lrf)
                          'momentum': tune.uniform(0.6, 0.98),  # SGD momentum/Adam beta1
                          'weight_decay': tune.uniform(0.0, 0.001),  # optimizer weight decay 5e-4
                          'warmup_epochs': tune.uniform(0.0, 5.0),  # warmup epochs (fractions ok)
                          'warmup_momentum': tune.uniform(0.0, 0.95),  # warmup initial momentum
                          },
                "SGD": {'lr0': tune.uniform(0.001, 0.01),
                        'lrf': tune.uniform(0.01, 0.5),  # final OneCycleLR learning rate (lr0 * lrf)
                        'momentum': tune.uniform(0.6, 0.98),  # SGD momentum/Adam beta1
                        'weight_decay': tune.uniform(0.0, 0.001),  # optimizer weight decay 5e-4
                        'warmup_epochs': tune.uniform(0.0, 5.0),  # warmup epochs (fractions ok)
                        'warmup_momentum': tune.uniform(0.0, 0.95),  # warmup initial momentum
                        }
                }

def train_ray_tune():
    datasets_yaml_dir = os.path.abspath("datasets_yaml")
    
    for model_name in ALL_MODELS:

        for dataset_yaml in os.listdir(datasets_yaml_dir):
            if "export" in dataset_yaml or "Shiny" in dataset_yaml or "Salmones" in dataset_yaml:
                pass
            else:
                data_yaml = os.path.join(datasets_yaml_dir, dataset_yaml)
                
                for optimizer in ["AdamW", "SGD"]:
                    # Load Model
                    model = YOLO(get_backbone_path(model_name), task="segment")
                    
                    # Name = {dataset}_{model_name}_{opt}
                    name = os.path.splitext(dataset_yaml)[0] + "_" + model_name + "_" + optimizer
                    
                    # Extra training params for yolov9e-seg
                    if model_name in ['yolov9e-seg']:
                        train_params = {"freeze":30}
                    else:
                        train_params = {}
                
                    # Tune hyperparameters
                    result_grid = model.tune(data=data_yaml, iterations=20, epochs=40, optimizer=optimizer,
                                            space=search_space_raytune[optimizer], gpu_per_trial=1, use_ray=True, **train_params)
                    
                    del model
                    del result_grid
    # Use the code from check_tune_results_ipynb for a simple review and savings of the results

def leer_resultados_raytune(ruta_json: str) -> dict:
    """
    Lee un archivo JSON que contiene configuraciones de experimentos y crea un nuevo diccionario
    con los rangos de parámetros especificados para cada tune, ajustado con una holgura del 25%.

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

        def calcular_holgura(valor1, valor2):
            """
            Calcula el rango con una holgura del 25% basada en el mínimo y máximo de dos valores.

            Parámetros:
            -----------
            valor1 : float
                Primer valor a comparar.
            valor2 : float
                Segundo valor a comparar.
            
            Retorna:
            --------
            tuple
                Rango ajustado con un 25% de holgura.
            """
            minimo = min(valor1, valor2)
            maximo = max(valor1, valor2)
            rango_ampliado_min = minimo - 0.25 * abs(minimo)
            rango_ampliado_max = maximo + 0.25 * abs(maximo)
            return rango_ampliado_min, rango_ampliado_max

        # Crear un diccionario para los rangos de parámetros con holgura
        rango_config = {
            'lr0': calcular_holgura(config1.get('lr0', 0), config2.get('lr0', 0)),
            'lrf': calcular_holgura(config1.get('lrf', 0), config2.get('lrf', 0)),
            'momentum': calcular_holgura(config1.get('momentum', 0), config2.get('momentum', 0)),
            'weight_decay': calcular_holgura(config1.get('weight_decay', 0), config2.get('weight_decay', 0)),
            'warmup_epochs': calcular_holgura(config1.get('warmup_epochs', 0), config2.get('warmup_epochs', 0)),
            'warmup_momentum': calcular_holgura(config1.get('warmup_momentum', 0), config2.get('warmup_momentum', 0)),
        }

        nuevo_diccionario[tune_number] = {'name': nombre_tune, 'config': rango_config}

    return nuevo_diccionario


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


def train_tune(search_spaces_dict: dict[str, dict], state_json_path: str):
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
                train_params = {"freeze":30}
            else:
                train_params = {}

            # Load Model
            model = YOLO(get_backbone_path(model_name), task="segment")

            # Tune hyperparameters
            result_grid = model.tune(data=data_yaml, iterations=10, epochs=40, optimizer=optimizer,
                                    space=search_space, use_ray=False, **train_params)

            del model
            del result_grid
            
            # Actualizar el estado a 1 y guardar en el archivo JSON
            estados[tune_number]['state'] = 1
            guardar_estado(state_json_path, estados)            


if __name__ == "__main__":
    # train_ray_tune()
    # Guardar resultados de raytune en el notebook check_tune_results.ipynb
    
    raytune_results = "resultados_raytune_deepfish.json"
    search_spaces_dict = leer_resultados_raytune(raytune_results)
    
    tune_training_state = "tune_training_state_deepfish.json"
    # inicializar_estados(search_spaces_dict, tune_training_state)
    
    train_tune(search_spaces_dict, tune_training_state)
