import os
import shutil
import yaml
from ultralytics.data.converter import convert_coco
from roboflow import Roboflow

from supersecrets import API_KEY


# Path donde guardar todos los datasets
DATASETS_DIRECTORY = "datasets"             # Path temporal donde guardar datasets descargados desde Roboflow
COCO_LABELS_DIRECTORY = "coco_converted"    # Path donde guardar los datasets de segmentación ya procesados
YAML_DIRECTORY = "datasets_yaml"            # Path donde se encuentran los archivos yaml de cada dataset

# Diccionario usado para descargar datasets
DATASETS_LINKS = {
    "Deepfish": dict(workspace="memristor", project="deepfish-segmentation-ocdlj", version=3, name="Deepfish"),
    "Deepfish_LO": dict(workspace="memristor", project="deepfish-segmentation-ocdlj", version=4, name="Deepfish_LO"),
    "Salmon": dict(workspace="memristor", project="salmones-ji1wj", version=5, name="Salmones"),
    "Salmon_LO": dict(workspace="memristor", project="salmones-ji1wj", version=6, name="Salmones_LO"),
    "Shiny_v2": dict(workspace="alejandro-guerrero-zihxm", project="shiny_salmons", version=2, name="ShinySalmonsV2"),
    "Shiny_v4": dict(workspace="alejandro-guerrero-zihxm", project="shiny_salmons", version=4, name="ShinySalmonsV4"),
}

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
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')

    # Itera a través de los archivos en el directorio de entrada
    for filename in os.listdir(input_path):
        # Verifica si el archivo tiene una extensión de imagen
        if filename.lower().endswith(image_extensions):
            # Construye las rutas completas de los archivos
            src_file = os.path.join(input_path, filename)
            dest_file = os.path.join(output_path, filename)

            # Copia el archivo
            shutil.copy2(src_file, dest_file)


def crear_coco_labels(name, path_dataset):
    """
    Crea etiquetas en formato COCO a partir de un dataset.

    Args:
        name (str): El nombre del dataset.
        path_dataset (str): La ruta del directorio del dataset.

    Example:
        crear_coco_labels('Deepfish', 'datasets/Deepfish')
    """
    for directory in os.listdir(path_dataset):
        if directory in ["test", "train", "valid"]:
            current_path = os.path.join(path_dataset, directory)
            output_path = os.path.join(COCO_LABELS_DIRECTORY, name, "labels", directory)
            convert_coco(labels_dir=current_path, save_dir=output_path, use_segments=True)


def copiar_imagenes_a_nuevo_dataset(name, path_dataset):
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
            output_path = os.path.join(COCO_LABELS_DIRECTORY, name, "images", directory)
            copy_images(current_path, output_path)


def move_and_cleanup(base_path="coco_converted/Deepfish"):
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


def create_yaml_datasets(yaml_dir, coco_converted_dir):
    # Crear yaml_dir si no existe
    os.makedirs(yaml_dir, exist_ok=True)
    
    # Verificar que coco_converted_dir exista
    if not os.path.exists(coco_converted_dir):
        raise FileNotFoundError(f"El directorio '{coco_converted_dir}' no existe.")
    
    # Obtener la lista de datasets (subdirectorios) dentro de coco_converted_dir
    datasets = [d for d in os.listdir(coco_converted_dir) if os.path.isdir(os.path.join(coco_converted_dir, d))]
    
    # Recorrer la lista de datasets y generar los archivos YAML
    for dataset_name in datasets:
        dataset_path = os.path.abspath(os.path.join(coco_converted_dir, dataset_name))
        images_dir = os.path.join(dataset_path, "images")
        
        if not os.path.exists(images_dir) or not os.path.exists(os.path.join(dataset_path, "labels")):
            print(f"El dataset '{dataset_name}' no tiene los subdirectorios 'images' o 'labels'.")
            continue
        
        # Verificar la existencia del subdirectorio 'test'
        has_test = os.path.exists(os.path.join(images_dir, "test"))

        # Crear contenido común para ambos archivos YAML
        yaml_base_content = {
            'path': dataset_path,
            'train': os.path.join("images", "train"),
            'val': os.path.join("images", "valid"),
            'nc': 1,
            'names': ["fish"]
        }
        
        # Guardar el primer YAML {dataset_name}.yaml
        yaml_content_1 = yaml_base_content.copy()
        if has_test:
            yaml_content_1['test'] = os.path.join("images", "test")
        
        with open(os.path.join(yaml_dir, f"{dataset_name}.yaml"), 'w') as f:
            yaml.dump(yaml_content_1, f, default_flow_style=False)
        
        # Guardar el segundo YAML export_{dataset_name}.yaml (sin la opción 'test')
        yaml_content_2 = yaml_base_content.copy()
        yaml_content_2['val'] = yaml_content_2['train']
        
        with open(os.path.join(yaml_dir, f"export_{dataset_name}.yaml"), 'w') as f:
            yaml.dump(yaml_content_2, f, default_flow_style=False)

    print("Archivos YAML creados exitosamente.")


def setup_datasets():
    datasets_a_descargar = ["Deepfish", "Deepfish_LO", "Salmon", "Salmon_LO", "Shiny_v4"]

    # Descargar los datasets necesarios
    for name, info in DATASETS_LINKS.items():
        if name in datasets_a_descargar:
            path_install_dataset_seg = os.path.join(DATASETS_DIRECTORY, f"{info['name']}")
            workspace = info['workspace']
            project = info['project']
            version = info['version']

            # Descargar dataset en formato coco desde Roboflow, en el directorio 'datasets'.
            download_roboflow_dataset(workspace, project, version, "coco-segmentation", path_install_dataset_seg)

            # Crear labels en formato Coco, en el directorio 'coco_converted'.
            crear_coco_labels(info['name'], path_install_dataset_seg)

            # Copiar las imagenes desde el dataset descargado en 'datasets' al nuevo en 'coco_converted'
            copiar_imagenes_a_nuevo_dataset(info['name'], path_install_dataset_seg)

            # Ordenar los labels y borrar carpetas vacías
            move_and_cleanup(os.path.join(COCO_LABELS_DIRECTORY, info['name']))

    # Luego puedes borrar la carpeta "datasets" y recuerda configurar los path en los archivos yaml
    create_yaml_datasets(YAML_DIRECTORY, COCO_LABELS_DIRECTORY)
    

if __name__ == "__main__":
    setup_datasets()
