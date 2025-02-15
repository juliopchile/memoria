[![English](https://img.shields.io/badge/lang-English-blue)](README.en.md)
[![Español](https://img.shields.io/badge/lang-Español-green)](README.es.md)
[![Français](https://img.shields.io/badge/lang-Français-yellow)](README.fr.md)
[![中文](https://img.shields.io/badge/lang-中文-red)](README.zh.md)

# Configuración
## Linux (recomendado)
Para configurar correctamente un sistema Linux recién instalado, ejecuta los siguientes comandos:
```bash
sudo apt update
sudo apt upgrade -y
sudo apt dist-upgrade -y
sudo apt autoremove -y
sudo apt autoclean
sudo apt install -y build-essential curl wget git vim
sudo apt install -y software-properties-common
sudo apt install -y zip unzip tar
```

## Instalación de Miniconda (recomendado)
Aunque no es obligatorio utilizar la distribución de Python a través de Anaconda, se recomienda hacerlo, dado que este código fue diseñado de esa manera. A continuación, se indica cómo instalar Miniconda en un sistema Linux:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Se sugiere crear un ambiente exclusivo para este proyecto:
```bash
conda create -n thesis python=3.10
conda activate thesis
```

## Drivers de Nvidia
Para realizar entrenamientos e inferencias con GPU, es necesario tener los controladores de Nvidia con soporte de CUDA. Visita la [página oficial de Nvidia](https://www.nvidia.com) para obtener los drivers adecuados. Verifica su instalación utilizando ``nvidia-smi`` en la terminal.

## CUDA Toolkit
Es imprescindible tener el CUDA Toolkit instalado en tu sistema para aprovechar el entrenamiento con GPU. A continuación se detalla el proceso de instalación en Linux, pero para obtener más detalles o versiones actualizadas, consulta la [página de CUDA Toolkit de Nvidia](https://developer.nvidia.com/cuda-downloads).
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda-repo-wsl-ubuntu-12-6-local_12.6.3-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-6-local_12.6.3-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
```

# Requerimientos
Dentro del ambiente de Python que hayas creado, instala los paquetes necesarios ejecutando:
```bash
# Para utilizar notebooks
conda install ipykernel
# Instalar pytorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# Instalar Ultralytics
pip install ultralytics
# Instalar Roboflow para importar el dataset Deepfish
pip install roboflow
```

## Instalación de AutoUpdates
Algunas bibliotecas utilizadas por Ultralytics no se instalan automáticamente, por lo que es necesario ejecutar código que solicite estas dependencias para que se efectúen las actualizaciones automáticamente. Para ello, ejecuta el archivo ``setup.py``, el cual no solo asegura la actualización automática de las bibliotecas necesarias, sino que también prepara el entorno con el dataset Deepfish y descarga los modelos requeridos.
```bash
python setup.py
```