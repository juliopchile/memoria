import os
import json
import shutil
import yaml
from typing import Tuple, List, Dict
from ultralytics.data.converter import convert_coco
from roboflow import Roboflow
from roboflow.core.dataset import Dataset

from config import DATASETS_COCO, DATASETS_YOLO, YAML_DIRECTORY, DATASETS_ROBOFLOW_LINKS
from supersecrets import API_KEY


def download_roboflow_dataset(workspace: str, project_id: str, version_number: int, model_format: str, location: str) -> Dataset | None:
    """ Descarga un dataset de Roboflow y lo guarda en la ubicación especificada.

    :param str workspace: El nombre del workspace en Roboflow.
    :param str project_id: El ID del proyecto en Roboflow.
    :param int version_number: El número de versión del dataset.
    :param str model_format: El formato del modelo a descargar.
    :param str location: La ubicación donde se guardará el dataset.
    :return Dataset: Dataset Object o None.
    """
    try:
        rf = Roboflow(api_key=API_KEY)
        project = rf.workspace(the_workspace=workspace).project(project_id=project_id)
        version = project.version(version_number=version_number)
        return version.download(model_format=model_format, location=location, overwrite=False)
    except Exception as error:
        print(error)
        return None


def copy_files(input_path: str, output_path: str, extension: str | Tuple[str, ...]):
    """ Copia archivos desde un directorio a otro si estos cumplen con cierta extensión.

    :param str input_path: La ruta del directorio de entrada que contiene los archivos.
    :param str output_path: La ruta del directorio de salida donde se copiarán las archivos.
    :param str | Tuple[str] extension: Extensión o tupla de extensiones a procesar.
    """
    # Itera a través de los archivos en el directorio de entrada
    for filename in os.listdir(input_path):
        # Verifica si el archivo tiene una extensión de imagen
        if filename.lower().endswith(extension):
            # Construye las rutas completas de los archivos
            src_file = os.path.join(input_path, filename)
            dest_file = os.path.join(output_path, filename)
            # Copia el archivo
            shutil.copy2(src_file, dest_file)


def copy_images(input_path: str, output_path: str):
    """ Copia todas las imágenes de un directorio de entrada a un directorio de salida.

    :param str input_path: La ruta del directorio de entrada que contiene las imágenes.
    :param str output_path: La ruta del directorio de salida donde se copiarán las imágenes.
    """
    # Asegúrate de que el directorio de salida exista
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Define las extensiones de imagen soportadas
    extension = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')

    copy_files(input_path, output_path, extension)


def copy_labels(input_path: str, output_path: str):
    """ Copia todas las etiquetas en forma de archivos txt de un directorio de entrada a un directorio de salida.

    :param str input_path: La ruta del directorio de entrada que contiene las etiquetas.
    :param str output_path: La ruta del directorio de salida donde se copiarán las etiquetas.
    """
    # Asegúrate de que el directorio de salida exista
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Define las extensiones de imagen soportadas
    extension = ('txt')

    copy_files(input_path, output_path, extension)


def dataset_coco_to_yolo(name: str, path_dataset: str):
    """ Crea etiquetas en formato YOLO a partir de un dataset en formato COCO.

    :param str name: El nombre del dataset, usado para el directorio donde guardarlo.
    :param str path_dataset: La ruta del directorio del dataset convertido.
    """
    yolo_path = os.path.join(DATASETS_YOLO, name)
    classes = []
    directories = []
    for directory in os.listdir(path_dataset):
        if directory in ["test", "train", "valid"]:
            # Manejar los directorios
            current_path = os.path.join(path_dataset, directory)
            output_path = os.path.join(yolo_path, "labels", directory)
            # Guardar el directorio y las clases
            classes.append(obtain_coco_classes(current_path))
            directories.append(directory)
            # Convertir las etiquetas al formato YOLO y guardarlas donde se dice
            convert_coco(labels_dir=current_path, save_dir=output_path, use_segments=True)
    # Obtener el diccionario de clases del dataset
    classes_dict = combine_and_reindex_classes(classes)
    # Crear el archivo yaml
    create_datasets_yaml(yolo_path, directories, classes_dict)


def obtain_coco_classes(coco_labels_dir: str) -> Dict[int, str]:
    """ Obtiene un diccionario con las clases de un dataset COCO. Se ignoran clases con 'supercategory' igual a none.

    :param str coco_labels_dir: Directorio donde se encuentra el archivo '_annotations.coco.json'
    :raises FileNotFoundError: Error si no se encontró archivo JSON.
    :raises KeyError: Error si el archivo JSON carece del campo 'categories'.
    :return Dict[int, str]: Diccionario con el el ID y nombre de las clases del dataset.
    """
    # Buscar el archivo JSON dentro del directorio
    json_file = None
    for file in os.listdir(coco_labels_dir):
        if file.endswith(".json"):
            json_file = os.path.join(coco_labels_dir, file)
            break

    # Si no se encuentra ningún archivo JSON
    if json_file is None:
        raise FileNotFoundError("No se encontró un archivo JSON en el directorio proporcionado.")

    # Abrir y cargar el archivo JSON
    with open(json_file, 'r') as f:
        coco_data = json.load(f)

    # Crear el diccionario de clases
    classes_dict = {}

    # Filtrar categorías basadas en la supercategoría
    if 'categories' in coco_data:
        for category in coco_data['categories']:
            class_id: int = category['id']
            class_name: str = category['name']
            supercategory = category.get('supercategory', None)

            # Excluir clases con supercategory "none"
            if supercategory != "none":
                classes_dict[class_id] = class_name
    else:
        raise KeyError("El archivo JSON no contiene la clave 'categories'.")

    return classes_dict


def combine_and_reindex_classes(class_dicts: List[Dict[int, str]]) -> Dict[int, str]:
    """ Toma una lsita de diccionarios, cada una contiene las ID y nombres de clases en un dataset.
    Luego se combinan estos diccionarios de clases en uno solo y re-indexionan desde el 0.

    :param List[Dict[int, str]] class_dicts: Lista de diccionario de clases.
    :return Dict[int, str]: Diccionario de clases combinado y re-indexado.
    """
    # Crear un diccionario combinado con todas las clases
    combined_classes = {}

    for class_dict in class_dicts:
        for class_id, class_name in class_dict.items():
            if class_name not in combined_classes.values():
                combined_classes[class_id] = class_name

    # Reasignar IDs desde 0
    reindexed_classes = {}
    new_id = 0
    for class_name in combined_classes.values():
        reindexed_classes[new_id] = class_name
        new_id += 1

    return reindexed_classes


def create_datasets_yaml(dataset_path: str, subdirectories: List[str], classes: Dict[int, str]):
    """ Crea el archivo 'data.yaml' de un dataset dado.

    :param str dataset_path: Directorio donde se encuentra el dataset. Donde se guardará el archivo YAML.
    :param list subdirectories: Lista de tareas incluidas en el dataset, pueden ser ('train', 'valid', 'test').
    :param Dict[int, str] classes: Diccionario de clases a usar.
    """
    # Obtener la cantidad de clases
    num_classes = len(classes)

    # Crear el diccionario para el archivo YAML
    yaml_file = {'path': os.path.abspath(dataset_path)}

    # Añadir las rutas solo si están en los subdirectorios
    if "train" in subdirectories:
        yaml_file['train'] = os.path.join("images", "train")
    if "valid" in subdirectories:
        yaml_file['val'] = os.path.join("images", "valid")
    if "test" in subdirectories:
        yaml_file['test'] = os.path.join("images", "test")

    # Añadir la cantidad de clases y los nombres de las clases
    yaml_file['nc'] = num_classes
    yaml_file['names'] = list(classes.values())

    # Guardar el archivo YAML en la ruta especificada
    yaml_path = os.path.join(dataset_path, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_file, f, default_flow_style=False)


def copy_images_to_new_dataset(name: str, path_dataset: str, output_dataset_path: str):
    """ Copia las imágenes de un dataset a un nuevo directorio.

    :param str name: El nombre del dataset.
    :param str path_dataset: La ruta del directorio del dataset.
    :param str output_dataset_path: La ruta a donde se quieren copiar las imagenes.
    """
    for directory in os.listdir(path_dataset):
        if directory in ["test", "train", "valid"]:
            current_path = os.path.join(path_dataset, directory)
            output_path = os.path.join(output_dataset_path, name, "images", directory)
            copy_images(current_path, output_path)


def move_and_cleanup(base_path: str):
    """ Mueve archivos y limpia directorios vacíos en una estructura de directorios específica.

    :param str base_path: La ruta base donde se realizará la operación.
    """
    # Define las rutas a explorar dentro de la ruta base
    labels_dir = os.path.join(base_path, 'labels')
    sub_dirs = ['test', 'train', 'valid']

    for sub_dir in sub_dirs:
        sub_dir_path = os.path.join(labels_dir, sub_dir)
        annotations_path = os.path.join(sub_dir_path, 'labels', '_annotations.coco')

        if os.path.exists(annotations_path):
            # Mueve archivos desde _annotations.coco al sub_dir_path
            for filename in os.listdir(annotations_path):
                src_file = os.path.join(annotations_path, filename)
                dest_file = os.path.join(sub_dir_path, filename)
                shutil.move(src_file, dest_file)

            # Elimina los directorios labels e images vacíos
            lower_labels_dir = os.path.join(sub_dir_path, 'labels')
            images_dir = os.path.join(sub_dir_path, 'images')

            if os.path.exists(lower_labels_dir):
                shutil.rmtree(lower_labels_dir)
            if os.path.exists(images_dir):
                shutil.rmtree(images_dir)


def create_export_datasets(datasets_directory: str):
    """ Crea datasets de exportación. Los dataset de exportación contienen las mismas imagenes y etiquetas
    que un dataset normal, pero se mueven todas las imagenes y etiquetas al subdirectorio train.
    El 'data.yaml' debe contener todos los subdirectorios apuntando a train.

    :param str datasets_directory: Directorio donde se encuentran los datasets originales.
    """

    for dataset in os.listdir(datasets_directory):
        if "export" not in dataset:
            dataset_export = f"export_{dataset}"
            dataset_path = os.path.join(datasets_directory, dataset)
            dataset_export_path = os.path.join(datasets_directory, dataset_export)

            # Copiar todos los archivos de dataset/images/["test", "train", "valid"] a dataset_export/images/train
            # y todos los archivos de dataset/labels/["test", "train", "valid"] a dataset_export/labels/train
            for directory in ["images", "labels"]:
                directory_path = os.path.join(dataset_path, directory)
                for subdirectory in ["test", "train", "valid"]:
                    subdirectory_path = os.path.join(directory_path, subdirectory)
                    target_path = os.path.join(dataset_export_path, directory, "train")

                    if directory == "images":
                        copy_images(subdirectory_path, target_path)
                    elif directory == "labels":
                        copy_labels(subdirectory_path, target_path)

            # Copiar el archivo data.yaml y modificarlo
            data_yaml_path = os.path.join(dataset_path, "data.yaml")
            export_yaml_path = os.path.join(dataset_export_path, "data.yaml")

            if os.path.exists(data_yaml_path):
                # Crear el directorio destino si no existe
                os.makedirs(dataset_export_path, exist_ok=True)

                # Copiar el archivo data.yaml al nuevo directorio de exportación
                shutil.copy(data_yaml_path, export_yaml_path)

                # Modificar el archivo data.yaml copiado
                with open(export_yaml_path, 'r') as f:
                    yaml_file = yaml.safe_load(f)

                # Cambiar el campo 'path' al path absoluto del dataset exportado
                yaml_file['path'] = os.path.abspath(dataset_export_path)

                # Cambiar los campos 'train', 'val', y 'test' para que apunten a 'train'
                yaml_file['train'] = os.path.join("images", "train")
                yaml_file['val'] = os.path.join("images", "train")
                yaml_file['test'] = os.path.join("images", "train")

                # Guardar el archivo YAML modificado
                with open(export_yaml_path, 'w') as f:
                    yaml.dump(yaml_file, f, default_flow_style=False)

                print(f"Archivo data.yaml modificado y guardado en {export_yaml_path}")
            else:
                print(f"No se encontró data.yaml en {dataset_path}, omitiendo...")


def create_yaml_directory(yaml_directory: str, yolo_datasets: str):
    """ Copia los archivos data.yaml dentro de cada directorio dataset dentro del directorio de 'yolo_datasets'
    y los pega en la carpeta 'yaml_directory' con el nombre del dataset.

    :param str yaml_directory: La ruta donde se guardarán las copias de los archivos .yaml.
    :param str yolo_datasets: La ruta donde están ubicados los datasets en formato YOLO.
    """
    # Crear el directorio yaml_directory si no existe
    if not os.path.exists(yaml_directory):
        os.makedirs(yaml_directory)

    # Iterar sobre los directorios de datasets en yolo_datasets
    for dataset in os.listdir(yolo_datasets):
        current_dataset_yaml_file = os.path.join(yolo_datasets, dataset, "data.yaml")

        # Verificar si el archivo data.yaml existe en el dataset
        if os.path.exists(current_dataset_yaml_file):
            # Definir la ruta de destino en yaml_directory con el nombre {dataset}.yaml
            dest_yaml_file = os.path.join(yaml_directory, f"{dataset}.yaml")

            # Copiar el archivo data.yaml a la nueva ubicación
            shutil.copy(current_dataset_yaml_file, dest_yaml_file)
            print(f"Archivo {current_dataset_yaml_file} copiado como {dest_yaml_file}")
        else:
            print(f"No se encontró data.yaml en {dataset}, omitiendo...")


def get_export_yaml_path(dataset_yaml: str) -> str:
    """ Dado el path de un archivo YAML de dataset, retorna el path de su versión de exportación.

    :param str dataset_yaml: Ruta del archivo YAML del dataset.
    :return str: Ruta del archivo de exportación con el prefijo 'export_' en el mismo directorio.
    """
    # Obtener el directorio y el nombre del archivo sin extensión
    dir_name = os.path.dirname(dataset_yaml)
    dataset_name_without_ext = os.path.splitext(os.path.basename(dataset_yaml))[0]

    # Construir el nuevo nombre de archivo
    export_yaml_name = f"export_{dataset_name_without_ext}.yaml"

    # Retornar la ruta completa
    return os.path.join(dir_name, export_yaml_name)


def create_labels_only_dataset(dataset_path: str):
    """ Crea un nuevo dataset YOLO que incluye solo las imágenes con etiquetas correspondientes en el subdirectorio 'train', 
    mientras copia todo el contenido de 'valid' y 'test' sin filtrar. También genera un nuevo archivo 'data.yaml' 
    con el 'path' actualizado.

    Nota:
        Se asume que el dataset original tiene la siguiente estructura:
        - images/
            - train/
            - valid/
            - test/
        - labels/
            - train/
            - valid/
            - test/
        - data.yaml

    :param str dataset_path: Ruta al directorio del dataset original.
    """
    # Obtener el nombre del dataset original y crear el nuevo nombre
    dataset_name = os.path.basename(dataset_path)
    new_dataset_name = dataset_name + "_LO"
    parent_dir = os.path.dirname(dataset_path)
    new_dataset_path = os.path.join(parent_dir, new_dataset_name)

    # Crear el directorio del nuevo dataset
    os.makedirs(new_dataset_path, exist_ok=True)

    # Crear la estructura de subdirectorios
    for subset in ['train', 'valid', 'test']:
        for dir_type in ['images', 'labels']:
            dir_path = os.path.join(new_dataset_path, dir_type, subset)
            os.makedirs(dir_path, exist_ok=True)

    # Filtrar y copiar para el subdirectorio "train"
    labels_train_path = os.path.join(dataset_path, 'labels', 'train')
    images_train_path = os.path.join(dataset_path, 'images', 'train')
    new_labels_train_path = os.path.join(new_dataset_path, 'labels', 'train')
    new_images_train_path = os.path.join(new_dataset_path, 'images', 'train')

    if os.path.exists(labels_train_path):
        txt_files = [f for f in os.listdir(labels_train_path) if f.endswith('.txt')]
        for txt_file in txt_files:
            base_name = os.path.splitext(txt_file)[0]
            # Buscar imagen con extensiones comunes
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                img_file = base_name + ext
                img_path = os.path.join(images_train_path, img_file)
                if os.path.exists(img_path):
                    shutil.copy(img_path, os.path.join(new_images_train_path, img_file))
                    shutil.copy(os.path.join(labels_train_path, txt_file), 
                              os.path.join(new_labels_train_path, txt_file))
                    break

    # Copiar todo para "valid" y "test" sin filtrar
    for subset in ['valid', 'test']:
        images_subset_path = os.path.join(dataset_path, 'images', subset)
        labels_subset_path = os.path.join(dataset_path, 'labels', subset)
        new_images_subset_path = os.path.join(new_dataset_path, 'images', subset)
        new_labels_subset_path = os.path.join(new_dataset_path, 'labels', subset)

        if os.path.exists(images_subset_path):
            shutil.copytree(images_subset_path, new_images_subset_path, dirs_exist_ok=True)
        if os.path.exists(labels_subset_path):
            shutil.copytree(labels_subset_path, new_labels_subset_path, dirs_exist_ok=True)

    # Copiar y actualizar el archivo data.yaml
    original_yaml_path = os.path.join(dataset_path, 'data.yaml')
    new_yaml_path = os.path.join(new_dataset_path, 'data.yaml')
    
    if os.path.exists(original_yaml_path):
        with open(original_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        data['path'] = os.path.abspath(new_dataset_path)
        with open(new_yaml_path, 'w') as f:
            yaml.dump(data, f)
    else:
        print(f"Advertencia: No se encontró data.yaml en {dataset_path}")


def setup_datasets():
    """ Función de ayuda que intenta descargar datasets desde Roboflow, convertirlos a formato YOLO, ordenarlos y crear los YAML correspondientes. """
    datasets_a_descargar = ["Deepfish", "Deepfish_LO", "Salmones", "Salmones_LO"]

    # Descargar los datasets necesarios desde Roboflow
    for name, info in DATASETS_ROBOFLOW_LINKS.items():
        if name in datasets_a_descargar:
            path_install_dataset_seg = os.path.join(DATASETS_COCO, f"{info['name']}")
            workspace = info['workspace']
            project = info['project']
            version = info['version']

            # Descargar dataset en formato COCO desde Roboflow, en el directorio 'datasets_coco'.
            download_roboflow_dataset(workspace, project, version, "coco-segmentation", path_install_dataset_seg)

            # Crear etiquetados en formato YOLO, en el directorio 'datasets_yolo'.
            dataset_coco_to_yolo(info['name'], path_install_dataset_seg)

            # Copiar las imagenes desde el dataset descargado en 'datasets_coco' al nuevo en 'datasets_yolo'
            copy_images_to_new_dataset(info['name'], path_install_dataset_seg, DATASETS_YOLO)

            # Ordenar los labels y borrar carpetas vacías
            move_and_cleanup(os.path.join(DATASETS_YOLO, info['name']))

    # Crear los datasets de exportación en formato YOLO
    create_export_datasets(DATASETS_YOLO)

    # Crea una carpeta donde se pueden hayar copias de los data.yaml de cada dataset para mayor accesibilidad
    create_yaml_directory(YAML_DIRECTORY, DATASETS_YOLO)


if __name__ == "__main__":
    create_labels_only_dataset("datasets_yolo/Salmones")
    setup_datasets()
