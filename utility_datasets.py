import os
import json
import shutil
import yaml
from ultralytics.data.converter import convert_coco
from roboflow import Roboflow

from config import DATASETS_COCO, DATASETS_YOLO, YAML_DIRECTORY, DATASETS_ROBOFLOW_LINKS
from supersecrets import API_KEY


def download_roboflow_dataset(workspace, project_id, version_number, model_format, location):
    """
    Descarga un dataset de Roboflow y lo guarda en la ubicación especificada.

    Args:
        workspace (str): El nombre del workspace en Roboflow.
        project_id (str): El ID del proyecto en Roboflow.
        version_number (int): El número de versión del dataset.
        model_format (str): El formato del modelo a descargar.
        location (str): La ubicación donde se guardará el dataset.

    Returns:
        dict: Información sobre el dataset descargado.
        None: Si ocurre un error durante la descarga.

    Example:
        download_roboflow_dataset('my_workspace', 'my_project', 1, 'yolov8', 'datasets/my_dataset')
    """
    try:
        rf = Roboflow(api_key=API_KEY)
        project = rf.workspace(the_workspace=workspace).project(project_id=project_id)
        version = project.version(version_number=version_number)
        return version.download(model_format=model_format, location=location, overwrite=False)
    except Exception as error:
        print(error)
        return None


def copy_files(input_path, output_path, extension):
    # Itera a través de los archivos en el directorio de entrada
    for filename in os.listdir(input_path):
        # Verifica si el archivo tiene una extensión de imagen
        if filename.lower().endswith(extension):
            # Construye las rutas completas de los archivos
            src_file = os.path.join(input_path, filename)
            dest_file = os.path.join(output_path, filename)

            # Copia el archivo
            shutil.copy2(src_file, dest_file)
    

def copy_images(input_path, output_path):
    """
    Copia todas las imágenes de un directorio de entrada a un directorio de salida.

    Args:
        input_path (str): La ruta del directorio de entrada que contiene las imágenes.
        output_path (str): La ruta del directorio de salida donde se copiarán las imágenes.

    Example:
        copy_images('datasets/my_dataset/train', 'datasets/my_dataset_copy/train')
    """
    # Asegúrate de que el directorio de salida exista
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Define las extensiones de imagen soportadas
    extension = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')

    copy_files(input_path, output_path, extension)


def copy_labels(input_path, output_path):
    """Copia todas las etiquetas en archivos txt
    """
    # Asegúrate de que el directorio de salida exista
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Define las extensiones de imagen soportadas
    extension = ('txt')

    copy_files(input_path, output_path, extension)
    

def dataset_coco_to_yolo(name, path_dataset):
    """
    Crea etiquetas en formato YOLO a partir de un dataset en formato COCO.

    Args:
        name (str): El nombre del dataset, usado para el directorio donde guardarlo.
        path_dataset (str): La ruta del directorio del dataset a convetir.

    Example:
        crear_yolo_labels('Deepfish', 'datasets/Deepfish')
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


def obtain_coco_classes(coco_labels_dir):
    # Detectar el archivo JSON dentro del directorio
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
            class_id = category['id']
            class_name = category['name']
            supercategory = category.get('supercategory', None)
            
            # Excluir clases con supercategory "none"
            if supercategory != "none":
                classes_dict[class_id] = class_name
    else:
        raise KeyError("El archivo JSON no contiene la clave 'categories'.")
    
    return classes_dict


def combine_and_reindex_classes(class_dicts):
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


def create_datasets_yaml(dataset_path: str, subdirectories: list, classes: dict):
    # Obtener la cantidad de clases
    num_classes = len(classes)

    # Crear el diccionario para el archivo YAML
    yaml_file = {
        'path': os.path.abspath(dataset_path),
    }

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
  

def copy_images_to_new_dataset(name, path_dataset, output_dataset_path):
    """
    Copia las imágenes de un dataset a un nuevo directorio.

    Args:
        name (str): El nombre del dataset.
        path_dataset (str): La ruta del directorio del dataset.

    Example:
        copiar_imagenes_a_nuevo_dataset('Deepfish', 'datasets/Deepfish')
    """
    for directory in os.listdir(path_dataset):
        if directory in ["test", "train", "valid"]:
            current_path = os.path.join(path_dataset, directory)
            output_path = os.path.join(output_dataset_path, name, "images", directory)
            copy_images(current_path, output_path)


def move_and_cleanup(base_path: str):
    """
    Mueve archivos y limpia directorios vacíos en una estructura de directorios específica.

    Args:
        base_path (str, optional): La ruta base donde se realizará la operación. Por defecto es "coco_converted/Deepfish".

    Example:
        move_and_cleanup('coco_converted/Deepfish')
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


def create_export_datasets(datasets_directory):
    """
    Crea datasets de exportación. Los dataset de exportación contienen las mismas imagenes y etiquetas
    que un dataset normal, pero se mueven todas las imagenes y etiquetas al subdirectorio train.
    El data.yaml debe contener todos los subdirectorios apuntando a train.

    Args:
        datasets_directory (_type_): Directorio donde se encuentran los datasets originales.
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
    """
    Copia los archivos data.yaml dentro de cada directorio dataset dentro del directorio de yolo_datasets
    y los pega en la carpeta yaml_directory con el nombre del dataset.

    Args:
        yaml_directory (str): La ruta donde se guardarán las copias de los archivos .yaml.
        yolo_datasets (str): La ruta donde están ubicados los datasets en formato YOLO.
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


def setup_datasets():
    datasets_a_descargar = ["Deepfish", "Deepfish_LO", "Salmon", "Salmon_LO", "Shiny_v4"]

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
    setup_datasets()
