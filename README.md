# HAMLOCK: HArdware-Model LOgically Combined attacK

This repository contains the implementation of HAMLOCK, a framework for targeted model poisoning via direct weight optimization. HAMLOCK enables adversaries to inject targeted misclassification behavior into pretrained models without retraining, using a constrained optimization objective that balances stealth and attack success.


## Repository Structure

| File                          | Description | 
|-------------------------------|-------------| 
| `data_utils.py`               | Dataset loading and preprocessing | 
| `inject_backdoor.py`          | Trigger optimization attack | 
| `inject_backdoor_weights.py`  | Weight optimization attack | 
| `main.py`                     | Entry point for trigger optimization based attack | 
| `main_optimize_weights.py`    | Entry point for weight optimization based attack | 
| `3N_attack.py`                | Multi-neuron backdoor injection and MSB detection (Standard Run) |
| `ablation.py`                 | Sensitivity analysis experiments (Neuron counts/Calibration) |
| `model.py`                    | Model architecture definitions | 
| `requirements.txt`            | Python dependencies | 
| `.gitignore`, `.cdsinit`      | Environment setup files | 
| `rtl/`                        | Verilog Files |

## Datasets
We conduct experiments using MNIST, CIFAR-10, GTSRB, and ImageNet. MNIST, CIFAR-10 and GTSRB are automatically downloaded if they are not found in the specified directory, while ImageNet must be manually downloaded from [here](https://www.image-net.org/challenges/LSVRC/2012/2012-downloads.php)

## Install Dependencies
You can directly install the dependencies by running the following command:
```
python3 -m pip install -r requirements.txt
```


## Single Neuron Attack
To run trigger optimization based attack run:
```bash
dataset_dir="./data/"
checkpoints_dir="./checkpoints/" 
dataset="cifar10" 
model="resnet"

python3 main.py \
    --dataset_dir $dataset_dir \
    --dataset $dataset \
    --epochs 100 \
    --model $model \
    --device "cuda:0" \
    --target_label 0 \
    --inject 1 \
    --train_model 1 \
    --batch_size 256 \
    --model_path $checkpoints_dir \ # Base directory
    --dump_model 1 \
    --lam 0.1 \
    --threshold 0.0 \
	--use_normalization 1 \
    --seed 1
```

To run weight optimization attack run:
```bash
python3 main_optimize_weights.py \
    --dataset_dir $dataset_dir \
    --dataset $dataset \
    --epochs 100 \
    --model $model \
    --device "cuda:0" \
    --target_label 0 \
    --inject 1 \
    --train_model 1 \
    --batch_size 256 \
    --model_path $checkpoints_dir \ # Base directory
    --dump_model 1 \
    --lam 0.1 \
    --threshold 0.0 \
	--use_normalization 1 \
    --seed 1
```

## Arguments 
| Argument | Description |
|--------|-------------|
| `--dataset_dir` | Root directory where datasets are stored or downloaded (Created Automatically) |
| `--dataset` | Dataset to use (`imagenet`, `cifar10`, `mnist`, `gtsrb`) |
| `--epochs` | Number of training epochs |
| `--model` | Model architecture (`resnet`, `vgg`, `lenet`) |
| `--device` | Device for execution (`cuda:0` for GPU, `cpu` for CPU) |
| `--target_label` | Target class label for the backdoor attack |
| `--inject` | Enables attack injection (1 = enabled, 0 = disabled) |
| `--train_model` | Trains the model if set to 1; otherwise loads an existing model |
| `--batch_size` | Number of samples per training batch |
| `--model_path` | Directory for saving and loading clean and poisoned models  |
| `--dump_model` | Saves the trained model to disk when set to 1 |
| `--lam` | Seperation between clean and backdoor data samples |
| `--threshold` | Threshold used in optimization |
| `--seed` | Random seed for reproducibility |



## Multi-Neuron attack
Before running the multi-neuron attack, you must first run trigger optimization code, since the multi-neuron attack code does not train the model itself. Set the checkpoints path using the following script:

```bash
dataset_dir="./data/"
dataset="cifar10"
seed=1
model="resnet"
checkpoints_path="./checkpoints/clean_models_1/${model}/${dataset}/model_${seed}.pth"
```

To run multi-neuron attack run:
```bash
python3 3N_attack.py \
    --dataset_dir $dataset_dir \
    --dataset $dataset \
    --model $model \
    --device "cuda:0" \
    --batch_size 256 \
    --model_path "${checkpoints_path}" \ # complete path of the model
```


To run ablation on multi-neuron attack run:
```bash
python3 ablation.py \
    --dataset_dir $dataset_dir \
    --dataset $dataset \
    --model $model \
    --device "cuda:0" \
    --batch_size 256 \
    --model_path "${checkpoints_path}" \ # complete path of the model
    --neuron_ablation 
```


## Prebuilt Scripts
The Bash scripts for running each attack are located in the scripts/ directory. To execute all attacks, run the following commands in order:

```bash
bash ./scripts/run_trigger_optimization_attack.sh
bash ./scripts/run_weight_optimization_attack.sh
bash ./scripts/run_3N_attack.sh
```


## Verilog Implementation
The complete Verilog codebase corresponding to HAMLOCK is available in the `rtl/` directory.
