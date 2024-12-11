[![English](https://img.shields.io/badge/lang-English-blue)](README.md)
[![Español](https://img.shields.io/badge/lang-Español-green)](README.es.md)
[![Français](https://img.shields.io/badge/lang-Français-yellow)](README.fr.md)
[![中文](https://img.shields.io/badge/lang-中文-red)](README.zh.md)

# Configuration
## Linux (recommandé)
Pour configurer correctement un système Linux nouvellement installé, exécutez les commandes suivantes :
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

## Installation de Miniconda (recommandé)
Bien qu'il ne soit pas obligatoire d'utiliser la distribution de Python via Anaconda, il est recommandé de le faire, car ce code a été conçu en conséquence. Voici comment installer Miniconda sur un système Linux :
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Il est conseillé de créer un environnement dédié à ce projet :
```bash
conda create -n thesis python=3.10
conda activate thesis
```

## Pilotes Nvidia
Pour effectuer l'apprentissage et l'inférence par le GPU, vous devez avoir des pilotes Nvidia avec le support CUDA. Visitez le [site officiel de Nvidia](https://www.nvidia.com) pour obtenir les pilotes appropriés. Vérifiez votre installation en utilisant ``nvidia-smi`` dans le terminal.

## Kit d'outils CUDA
Il est essentiel d'installer le kit d'outils CUDA sur votre système pour tirer parti de l'apprentissage par le GPU. Le processus d'installation sous Linux est détaillé ci-dessous, mais pour plus de détails ou des versions mises à jour, consultez la [page Nvidia CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).

# Exigences
Dans l'environnement Python que vous avez créé, installez les paquets nécessaires en exécutant :
```bash
# Pour utiliser les notebooks
conda install ipykernel
# Installer pytorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# Installer Ultralytics
pip install ultralytics
# Installer Roboflow pour importer le jeu de données Deepfish
pip install roboflow
```

## Installer les mises à jour automatiques
Certaines bibliothèques utilisées par Ultralytics ne sont pas installées automatiquement, il est donc nécessaire d'exécuter un code qui demande ces dépendances afin que les mises à jour soient effectuées automatiquement. Pour ce faire, exécutez le fichier ``setup.py``, qui non seulement assure la mise à jour automatique des bibliothèques requises, mais prépare également l'environnement avec le jeu de données Deepfish et télécharge les modèles requis.
```bash
python setup.py
```