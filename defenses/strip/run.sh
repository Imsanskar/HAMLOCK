#!/usr/bin/env bash
# Reproduce STRIP results against HAMLOCK (claim C3, Tables 4 & 5).
# Run from the repository root: bash defenses/strip/run.sh
# Requires the backdoored checkpoints from the attack stage under ./checkpoints.
dataset_dir="./data/"
checkpoints_dir="./checkpoints/"
log_dir="./defenses/strip/logs/"
device="cuda:0"
seed=1
mkdir -p "$log_dir"

# 1) Software-only model (all attack variants)
for attack in hamock hamock_sep hamock_weights; do
  for dataset in cifar10 gtsrb; do
    for model in resnet vgg_bn; do
      echo "[STRIP/software] $attack $dataset $model"
      python3 defenses/strip/strip.py \
        --dataset_dir "$dataset_dir" --model_path "$checkpoints_dir" \
        --attack "$attack" --dataset "$dataset" --model "$model" \
        --strip_mode 1 --target_label 0 --device "$device" \
        --n_sample 100 --batch_size 32 --n_benign_sample 500 \
        --use_normalization 1 --seed "$seed" \
        > "${log_dir}/strip_${attack}_${dataset}_${model}_seed${seed}.log" 2>&1
    done
  done
done

# 2) Hardware-deployed, single-neuron attacks
for attack in hamock hamock_weights; do
  for dataset in cifar10 gtsrb; do
    for model in resnet; do
      echo "[STRIP/hardware] $attack $dataset $model"
      python3 defenses/strip/strip_hardware.py \
        --dataset_dir "$dataset_dir" --model_path "$checkpoints_dir" \
        --attack "$attack" --dataset "$dataset" --model "$model" \
        --strip_mode 1 --target_label 0 --device "$device" \
        --n_sample 100 --batch_size 32 --n_benign_sample 500 \
        --use_normalization 1 --seed "$seed" \
        > "${log_dir}/strip_hardware_${attack}_${dataset}_${model}_seed${seed}.log" 2>&1
    done
  done
done

# 3) Hardware-deployed, multi-neuron attack
for dataset in cifar10 gtsrb; do
  for model in resnet vgg_bn; do
    echo "[STRIP/hardware-sep] hamock_sep $dataset $model"
    python3 defenses/strip/strip_hardware_sep.py \
      --dataset_dir "$dataset_dir" --model_path "$checkpoints_dir" \
      --attack hamock_sep --dataset "$dataset" --model "$model" \
      --strip_mode 1 --target_label 0 --device "$device" \
      --n_sample 100 --batch_size 32 --n_benign_sample 500 \
      --use_normalization 1 --seed "$seed" \
      > "${log_dir}/strip_hardware_hamock_sep_${dataset}_${model}_seed${seed}.log" 2>&1
  done
done
