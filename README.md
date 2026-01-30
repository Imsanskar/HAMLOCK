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
We conduct experiments using MNIST, CIFAR-10, GTSRB, and ImageNet. MNIST, CIFAR-10 and GTSRB are automatically downloaded if they are not found in the specified directory, while ImageNet must be manually downloaded from [here](https://www.image-net.org/challenges/LSVRC/2012/2012-downloads.php) in a specific directory. 

## Install Dependencies
You can directly install the dependencies by running the following command:
```
python3 -m pip install -r requirements.txt
```

We conducted our experiments with `Python 3.9.21`.

## Steps to run the attack
1. Initially, run the trigger optimization attack. This step trains the model and saves the resulting clean checkpoints in the `$checkpoints_dir` directory which can be used later for weight optimization attack and multi-neuron attack. The `$dataset_dir` variable specifies where the dataset is stored, and the script will automatically create this directory if it does not already exist. Use the following command to run the full trigger‑optimization pipeline.

```bash
dataset_dir="./data/" # directory that contains the dataset
checkpoints_dir="./checkpoints/" 
dataset="cifar10" # Options: imagenet, cifar10, gtsrb, mnist
model="resnet" # Options: resnet, vgg_bn, lenet

python3 main.py \
    --dataset_dir $dataset_dir \
    --dataset $dataset \
    --epochs 100 \
    --model $model \
    --device "cuda:0" \
    --inject 1 \
    --train_model 1 \
    --batch_size 256 \
	--model_path $checkpoints_dir \
    --dump_model 1 \
    --lam 0.1 \
    --threshold 0.0 \
	--use_normalization 1 \
    --seed 1
```

The clean models will be saved in `${model_path}/clean_models_1/${model}/${dataset}/model_${seed}.pth` and the poisoned model will be saved in `${model_path}/hamock_1/${model}/${dataset}/model_${seed}.pth`

2. After training the clean models, the weight‑optimization attack can reuse the same clean checkpoint. If `train_model` is set to `1`, the script will train a new model; otherwise, it will load the existing clean model. In this step, you only need to provide the base directory for `$checkpoints_dir`—the full model path is not required. To run the complete weight‑optimization attack, use the following command:

```bash
dataset_dir="./data/" 
checkpoints_dir="./checkpoints/" 
dataset="cifar10" # Options: imagenet, cifar10, gtsrb, mnist
model="resnet" # Options: resnet, vgg_bn, lenet

python3 main_optimize_weights.py \
    --dataset_dir $dataset_dir \
    --dataset $dataset \
    --epochs 100 \
    --model $model \
    --device "cuda:0" \
    --target_label 0 \
    --inject 1 \
    --train_model 0 \
    --batch_size 256 \
    --model_path $checkpoints_dir \
    --dump_model 1 \
    --lam 0.1 \
    --threshold 0.0 \
	--use_normalization 1 \
    --seed 1
```

3. Before running the multi‑neuron attack, you must specify the **complete path** to the trained clean model using the `$checkpoint_path` variable. This path should point directly to the `.pth` file produced during the trigger‑optimization stage. The example below shows how to set `$checkpoint_path` correctly before launching the multi‑neuron attack.
```bash
dataset_dir="./data/"
dataset="cifar10"
seed=1
model="resnet"
checkpoints_path="./checkpoints/clean_models_1/${model}/${dataset}/model_${seed}.pth"
```

After setting the checkpoint directory, run the following script to launch the multi‑neuron attack. This step loads the clean model from the path you specified and applies the multi‑neuron attack.

```bash
dataset_dir="./data/"
dataset="cifar10"
seed=1
model="resnet"
checkpoints_path="./checkpoints/clean_models_1/${model}/${dataset}/model_${seed}.pth"


python3 3N_attack.py \
    --dataset_dir $dataset_dir \
    --dataset $dataset \
    --model $model \
    --device "cuda:0" \
    --batch_size 256 \
    --model_path "${checkpoints_path}"
```


4. To run ablation on multi-neuron attack run:
```bash
dataset="cifar10"
seed=1
model="resnet"
checkpoints_path="./checkpoints/clean_models_1/${model}/${dataset}/model_${seed}.pth"

python3 ablation.py \
    --dataset $dataset \
    --model $model \
    --device "cuda:0" \
    --batch_size 256 \
    --model_path "${checkpoints_path}" \
    --save_dir "./checkpoints/" \
    --neuron_ablation 
```


## Prebuilt Scripts
The Bash scripts for running each attack are located in the `scripts/` directory. To execute all attacks, run the following commands in order:

```bash
bash ./scripts/run_trigger_optimization_attack.sh
bash ./scripts/run_weight_optimization_attack.sh
bash ./scripts/run_3N_attack.sh
```


## Arguments 
| Argument | Description |
|--------|-------------|
| `--dataset_dir` | Root directory where datasets are stored or downloaded (Created Automatically) |
| `--dataset` | Dataset to use (`imagenet`, `cifar10`, `mnist`, `gtsrb`) |
| `--epochs` | Number of training epochs |
| `--model` | Model architecture (`resnet`, `vgg_bn`, `lenet`) |
| `--device` | Device for execution (`cuda:0` for GPU, `cpu` for CPU) |
| `--inject` | Enables attack injection (1 = enabled, 0 = disabled) |
| `--train_model` | Trains the model if set to 1; otherwise loads an existing model |
| `--batch_size` | Number of samples per training batch |
| `--model_path` | Directory for saving and loading clean and poisoned models  |
| `--dump_model` | Saves the trained model to disk when set to 1 |
| `--lam` | Seperation between clean and backdoor data samples |
| `--threshold` | Threshold used in optimization |
| `--seed` | Random seed for reproducibility |


## Verilog Implementation
The complete Verilog codebase corresponding to HAMLOCK is available in the `rtl/` directory.
