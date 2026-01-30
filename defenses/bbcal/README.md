# BBCaL: Black-Box Calibration-based Localization Defense for HAMLOCK

This folder section documents the usage of **BBCaL**,  models attacked using **HAMLOCK**. BBCaL operates without access to model internals and can be evaluated both **before** and **after** deployment to hardware.

BBCaL is evaluated across two execution environments:
- **Software-only (pre-hardware deployment)**
- **Hardware-deployed models**

---

## Repository Structure (BBCaL)

| File | Description |
|------|-------------|
| `test_bbcal_hamock.py` | BBCaL evaluation for HAMLOCK attacks in software (single-neuron) |
| `test_bbcal_hamock_hardware.py` | BBCaL evaluation for Single Neuron attack after hardware deployment |
| `test_bbcal_hamock_sep_hardware.py` | BBCaL evaluation for Multi-neuron attack after hardware deployment |
| `checkpoints/` | Directory containing clean and attacked model checkpoints |

---

## Supported Datasets and Models

BBCaL supports the same experimental configurations as HAMLOCK:

- **Datasets**: MNIST, CIFAR-10, GTSRB, ImageNet  
- **Models**: LeNet, ResNet, VGG16  

The dataset directory must be specified using the `--dataset_dir` argument and should match the dataset used during the HAMLOCK attack generation phase.

---

## Running BBCaL Defense

Below we provide step-by-step instructions for running BBCaL under different attack and deployment settings.

---

### 1. BBCaL on Single-Neuron and Multi-Neuron Attack (Before Hardware Deployment)

This setting evaluates BBCaL on HAMLOCK single-neuron attacks using software-only model checkpoints.

```bash
dataset="cifar10"        # Options: cifar10, gtsrb, mnist, imagenet
model="resnet"           # Options: resnet, vgg16, lenet
attack="hamock_sep"      # Options: hamock, hamock_weights, hamock_sep
seed=1
checkpoints_path="./checkpoints/"   # Root directory containing model checkpoints
datasets_dir="./data/"

python3 test_bbcal_hamock.py \
    --dataset $dataset \
    --model $model \
    --attack $attack \
    --seed $seed \
    --dataset_dir $datasets_dir \
    --model_path $checkpoints_path \
    --device "cuda:0"
```

### 2. BBCaL on Single-Neuron Attack (After Hardware Deployment)

This setting evaluates BBCaL on HAMLOCK single-neuron attacks after the model has been deployed to hardware
```bash
dataset="cifar10"
model="resnet"
attack="hamock" # Options: hamock, hamock_weights
seed=1
checkpoints_path="./checkpoints/"
datasets_dir="./data/"

python3 test_bbcal_hamock_hardware.py \
    --dataset $dataset \
    --model $model \
    --attack $attack \
    --seed $seed \
    --dataset_dir $datasets_dir \
    --model_path $checkpoints_path \
    --device "cuda:0"
```

### BBCaL on Multi-Neuron Attack (After Hardware Deployment)
```bash
dataset="cifar10"
model="resnet"
attack="hamock_sep" # Options: hamock, hamock_weights
seed=1
checkpoints_path="./checkpoints/"
datasets_dir="./data/"

python3 test_bbcal_hamock_sep_hardware.py \
    --dataset $dataset \
    --model $model \
    --attack $attack \
    --seed $seed \
    --dataset_dir $datasets_dir \
    --model_path $checkpoints_path \
    --device "cuda:0"
```