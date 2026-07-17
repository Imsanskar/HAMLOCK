#!/usr/bin/env bash
# Reproduce TED results against HAMLOCK (claim C2, Table 3).
# Run from the repository root: bash defenses/TED/run.sh
# Requires the backdoored checkpoints from the attack stage under ./checkpoints.
dataset_dir="./data/"
checkpoints_dir="./checkpoints/"
log_dir="./defenses/TED/logs/"
device="cuda:0"
seed=1
mkdir -p "$log_dir"

for attack in hamock hamock_sep hamock_weights; do
  for dataset in cifar10 gtsrb; do
    for model in resnet vgg_bn; do
      echo "[TED] $attack $dataset $model"
      python3 defenses/TED/ted.py \
        --model_path "$checkpoints_dir" --dataset_dir "$dataset_dir" \
        --attack "$attack" --model "$model" --dataset "$dataset" \
        --batch_size 32 --target 0 --device "$device" --seed "$seed" \
        > "${log_dir}/ted_${attack}_${dataset}_${model}_seed${seed}.log" 2>&1
    done
  done
done

# MNIST / LeNet (seed 1). LeNet (Arch A) is a 2-conv net; hamock_sep places its
# 3 trigger neurons as 2 in cnn.0 + 1 in cnn.2.
for attack in hamock hamock_sep hamock_weights; do
  echo "[TED] $attack mnist lenet"
  python3 defenses/TED/ted.py \
    --model_path "$checkpoints_dir" --dataset_dir "$dataset_dir" \
    --attack "$attack" --model lenet --dataset mnist \
    --batch_size 32 --target 0 --device "$device" --seed "$seed" \
    > "${log_dir}/ted_${attack}_mnist_lenet_seed${seed}.log" 2>&1
done
