[![English](https://img.shields.io/badge/lang-English-blue)](README.en.md)
[![Español](https://img.shields.io/badge/lang-Español-green)](README.es.md)
[![Français](https://img.shields.io/badge/lang-Français-yellow)](README.fr.md)
[![中文](https://img.shields.io/badge/lang-中文-red)](README.zh.md)

# Configuration
## Linux (recommended)
To properly configure a newly installed Linux system, run the following commands:
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

## Miniconda Installation (recommended)
Although it is not mandatory to use the Python distribution through Anaconda, it is recommended to do so, since this code was designed like that. The following is how to install Miniconda on a Linux system:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

It is suggested to create a dedicated environment for this project:
```bash
conda create -n thesis python=3.10
conda activate thesis
```

## Nvidia drivers
To perform GPU training and inference, you need to have Nvidia drivers with CUDA support. Visit the [official Nvidia website](https://www.nvidia.com) to obtain the appropriate drivers. Verify your installation by using ``nvidia-smi`` in the terminal.

## CUDA Toolkit
It is essential to have the CUDA Toolkit installed on your system to take advantage of GPU training. The installation process on Linux is detailed below, but for more details or updated versions, see the [Nvidia CUDA Toolkit page](https://developer.nvidia.com/cuda-downloads).
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda-repo-wsl-ubuntu-12-6-local_12.6.3-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-6-local_12.6.3-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
```

# Requirements
Within the Python environment you have created, install the necessary packages by running:
```bash
# To use notebooks
conda install ipykernel
# Install pytorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# Install Ultralytics
pip install ultralytics
# Install Roboflow to import the Deepfish dataset
pip install roboflow
```

## Install AutoUpdates
Some libraries used by Ultralytics are not installed automatically, so it is necessary to run code that requests these dependencies so that updates are performed automatically. To do this, run the ``setup.py`` file, which not only ensures the automatic update of the required libraries, but also prepares the environment with the Deepfish dataset and downloads the required models.
```bash
python setup.py
```