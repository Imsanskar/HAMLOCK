#!/usr/bin/env bash
# Reproduce CLP results against HAMLOCK (claim C4, Table 6).
# Run from the repository root: bash defenses/CLP/run.sh
# Requires the backdoored checkpoints from the attack stage under ./checkpoints.
dataset_dir="./data/"
checkpoints_dir="./checkpoints"
log_dir="./defenses/CLP/logs"
device="cuda:0"
seed=1
mkdir -p "$log_dir"

# Single-neuron attacks (hamock_test.py)
for attack in hamock hamock_weights; do
  for dataset in cifar10 gtsrb; do
    for model in resnet vgg_bn; do
      echo "[CLP/single] $attack $dataset $model"
      python3 defenses/CLP/hamock_test.py \
        -u 10 --attack "$attack" --model_path "$checkpoints_dir" --dataset_dir "$dataset_dir" \
        --model "$model" --dataset "$dataset" --use_normalization 1 --device "$device" --seed "$seed" \
        > "${log_dir}/clp_${attack}_${dataset}_${model}_seed${seed}.log" 2>&1
    done
  done
done

# Multi-neuron attack (clp_sep.py)
for dataset in cifar10 gtsrb; do
  for model in resnet vgg_bn; do
    echo "[CLP/sep] hamock_sep $dataset $model"
    python3 defenses/CLP/clp_sep.py \
      -u 5 --attack hamock_sep --model_path "$checkpoints_dir" --dataset_dir "$dataset_dir" \
      --model "$model" --dataset "$dataset" --use_normalization 1 --device "$device" --seed "$seed" \
      > "${log_dir}/clp_hamock_sep_${dataset}_${model}_seed${seed}.log" 2>&1
  done
done
