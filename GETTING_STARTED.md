## Getting Started with MARIS

### Detectron2 Preparation
This document provides a brief intro of the usage of MARIS.

Download our **modified** [Detectron2](https://pan.baidu.com/s/1EpIkSA9mlndVW5lgVYvLtA?pwd=USTC), and put it in the same directory as this README. you should run `python -m pip install -e detectron2` to install detectron2.

### Pretrained Weight
[convnext_base](https://huggingface.co/laion/CLIP-convnext_base_w_320-laion_aesthetic-s13B-b82K-augreg/tree/main)
[convnext_large](https://huggingface.co/laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/tree/main)
[depthanything](https://github.com/DepthAnything/Depth-Anything-V2)

Put these weights into ```MARIS/pretrained```
### Dataset Preparation

We provide a script `train_net.py`, that is made to train all the configs provided in MARIS.

- For our **Cross-domain** setting, you should download the COCO dataset, please refer to [here](datasets\README.md)

- For our **In-domain** setting, downloading the WaterOVS dataset from [Our Baidu Netdisk](https://pan.baidu.com/s/1XcpDFIWixPj6vxWiHx5DtA?pwd=USTC)

put the downloaded dataset in `datasets` directory.

```
datasets/
  coco/
  WaterOVS/
```

### Training & Evaluation in Command Line

#### Training
You should refer to the:
```
sh KEY_START_cd.sh
sh KEY_START_id.sh
```
for running.

The configs are made for 4-GPU training.
Since we use ADAMW optimizer, it is not clear how to scale learning rate with batch size.
To train on 1 GPU, you need to figure out learning rate and batch size by yourself.

#### Evaluation
To evaluate a model's performance, use

```
python train_net_id.py \
  --config-file configs/convnext_base_waterovs_id.yaml \
  --eval-only MODEL.WEIGHTS /path/to/checkpoint_file

python train_net_id.py \
  --config-file configs/convnext_large_waterovs_id.yaml \
  --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```

```
python train_net_cd.py \
  --config-file configs/convnext_base_waterovs_cd.yaml \
  --eval-only MODEL.WEIGHTS /path/to/checkpoint_file

python train_net_cd.py \
  --config-file configs/convnext_large_waterovs_cd.yaml \
  --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```

For more options, see `python train_net_id(cd).py -h`.

#### NOTE
`python train_net_id.py` and `python train_net_cd.py` have **no** different training settings.

#### 🚩 Important: Getting the Results
copy the table from `output/log.txt` to `results.txt` and run the following command: `python get_results.py` to get the results like our results in table.

**what table?**: a example has already shown in `results.txt`.
#### Analysis

To analyze the performance of MARIS, you can use the following command:
```
  sh KEY_flop_analy.sh
```
