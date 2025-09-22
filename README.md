# HAMLOCK: HArdware-Model LOgically Combined attacK

This repository contains the implementation of HAMOCK, a framework for targeted model poisoning via direct weight optimization. HAMOCK enables adversaries to inject targeted misclassification behavior into pretrained models without retraining, using a constrained optimization objective that balances stealth and attack success.


## Repository Structure

| File                          | Description |
|-------------------------------|-------------|
| `data_utils.py`               | Dataset loading and preprocessing |
| `inject_backdoor.py`          | Trigger optimization attack |
| `inject_backdoor_weights.py`  | Weight optimization attack |
| `main.py`                     | Entry point for trigger optimization based attack |
| `main_optimize_weights.py`    | Entry point for weight optimization based attack |
| `model.py`                    | Model architecture definitions |
| `requirements.txt`            | Python dependencies |
| `.gitignore`, `.cdsinit`      | Environment setup files |


## Reproducing the Experiment

To run HAMOCK (weight optimization attack) on CIFAR-10 with ResNet:

```bash
python3 -m pip install -r requirements.txt # install dependencies
dataset_dir="./data/"
checkpoints_dir="./checkpoints/"
dataset="cifar10" # options: imagenet, cifar10, mnist, gtsrb
model="resnet"  # Options: resnet, vgg, lenet


# Use main.py for trigger optimization based attack
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
    --seed 1
```