# tesis
Trabajo de tésis utilizando YOLO para segmentación de instancias de salmones.



# Setup
## Linux (Recomendado)
Si se tiene un Linux nuevo recién instalado, empezar con estos comandos para configurar correctamente el sistema operativo.
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
## Instalar miniconda (Recomendado)
No es necesario utilizar la distribución de Python a travez de Anaconda, pero es recomendado ya que así fue diseñado este código. A continuación se muestra como instalar Miniconda en un sistema Linux.
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
Además se recomienda crear un ambiente exclusivo para este proyecto.
```bash
conda create -n tesis python=3.10
conda activate tesis
```

## Drivers de Nvidia
Es necesario tener los drivers de nvidia con soporte de CUDA para realizar los entrenamientos e inferencias con GPU. Visita la [página oficial](https://www.nvidia.com) de Nvidia para obtener los drivers necesarios. Puedes revisar si lo tienes instalado escribiendo `nvidia-smi` dentro de tu terminal.

## CUDA Toolkit
Es necesario tener CUDA Toolkit instalado en tu sistema operativo para poder utilizar las. A continuación se muestra como instalarlo en un sistema Linux. Para más información o para instalar una versión más actualizada, visitar la [página](https://developer.nvidia.com/cuda-downloads?) de CUDA Toolkit de Nvidia.
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
Una vez dentro del ambiente python que deseas utilizar, instalar los paquetes necesarios.
```bash
# Para utilizar los notebooks
conda install ipykernel
# Instalar pytorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# Instalar Ultralytics
pip install ultralytics
# Instalar Roboflow para importar el dataset Deepfish
pip install roboflow
```
