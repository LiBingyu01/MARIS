## Installation

### 1. Requirements
- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.9 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. Note, please check
  PyTorch version matches that is required by Detectron2.
- Detectron2: Download our **modified** [Detectron2](https://pan.baidu.com/s/1EpIkSA9mlndVW5lgVYvLtA?pwd=USTC), and run `python -m pip install -e detectron2`
- OpenCV is optional but needed by demo and visualization
- `pip install -r requirements.txt`

### 2. CUDA kernel for MSDeformAttn
After preparing the required environment, run the following command to compile CUDA kernel for MSDeformAttn:

`CUDA_HOME` must be defined and points to the directory of the installed CUDA toolkit.

```bash
cd maris/modeling/pixel_decoder/ops
sh make.sh
```

#### 2.1 Building on another system
To build on a system that does not have a GPU device but provide the drivers:
```bash
TORCH_CUDA_ARCH_LIST='8.0' FORCE_CUDA=1 python setup.py build install
```

### 2.2 Example conda environment setup
```bash
conda create --name maris python=3.8 -y
conda activate maris
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia
pip install -U opencv-python
cd ../../../..
```

## 3 for your own working directory
- Download our **modified** [Detectron2](https://pan.baidu.com/s/1EpIkSA9mlndVW5lgVYvLtA?pwd=USTC), and run:
```bash
python -m pip install -e detectron2
```

- for others:
```bash
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/mcordts/cityscapesScripts.git
pip install -r requirements.txt
cd maris/modeling/pixel_decoder/ops
sh make.sh
cd ../../../..
```
