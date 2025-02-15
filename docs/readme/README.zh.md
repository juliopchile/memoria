[![English](https://img.shields.io/badge/lang-English-blue)](README.en.md)
[![Español](https://img.shields.io/badge/lang-Español-green)](README.es.md)
[![Français](https://img.shields.io/badge/lang-Français-yellow)](README.fr.md)
[![中文](https://img.shields.io/badge/lang-中文-red)](README.zh.md)

# 配置
## Linux（推荐）
要正确配置新安装的 Linux 系统，请执行以下命令：
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

## 安装 Miniconda（推荐）
虽然不是必须通过 Anaconda 来使用 Python 的发行版，但推荐这样做，因为此代码是基于这种方式设计的。以下是在 Linux 系统上安装 Miniconda 的步骤：

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

建议为此项目创建一个专用环境：
```bash
conda create -n thesis python=3.10
conda activate thesis
```

## Nvidia 驱动程序
要使用 GPU 进行训练和推理，需要安装支持 CUDA 的 Nvidia 驱动程序。请访问 [Nvidia 官方页面](https://www.nvidia.com) 下载适合的驱动程序。安装完成后，可以在终端中使用 ``nvidia-smi`` 命令验证安装是否成功。

## CUDA 工具包  
要利用 GPU 进行训练，必须在系统中安装 CUDA 工具包。以下是 Linux 系统的安装过程，但有关更多详细信息或更新版本，请参阅 [Nvidia CUDA 工具包页面](https://developer.nvidia.com/cuda-downloads)。
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda-repo-wsl-ubuntu-12-6-local_12.6.3-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-6-local_12.6.3-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
```

# 必要条件
在已创建的Python环境中，执行以下命令安装所需的包：
```bash
# 用于使用 notebooks
conda install ipykernel
# 安装 PyTorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# 安装 Ultralytics
pip install ultralytics
# 安装 Roboflow 以导入 Deepfish 数据集
pip install roboflow
```

## AutoUpdates 安装
由于Ultralytics使用的某些库不会自动安装，因此必须运行请求这些依赖项的代码以便进行自动更新。为此，执行 ``setup.py`` 文件，它不仅确保所需库的自动更新，还会设置Deepfish数据集环境并下载所需的模型。
```bash
python setup.py
```