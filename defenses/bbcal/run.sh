#!/usr/bin/env bash
# Reproduce BBCaL results against HAMLOCK (claim C3, Tables 4 & 5).
# Run from the repository root: bash defenses/bbcal/run.sh
# Requires the backdoored checkpoints from the attack stage under ./checkpoints.
dataset_dir="./data/"
checkpoints_dir="./checkpoints/"
log_dir="./defenses/bbcal/logs/"
device="cuda:0"
seed=1
mkdir -p "$log_dir"

# 1) Software-only model (all attack variants)
for attack in hamock hamock_sep hamock_weights; do
  for dataset in cifar10 gtsrb; do
    for model in resnet vgg_bn; do
      echo "[BBCaL/software] $attack $dataset $model"
      python3 defenses/bbcal/test_bbcal_hamock.py \
        --attack "$attack" --dataset "$dataset" --model "$model" \
        --model_path "$checkpoints_dir" --dataset_dir "$dataset_dir" \
        --use_normalization 1 --use_gaussian_noise 0 --device "$device" --seed "$seed" \
        > "${log_dir}/bbcal_${attack}_${dataset}_${model}_seed${seed}.log" 2>&1
    done
  done
done

# 2) Hardware-deployed, single-neuron attacks
for attack in hamock hamock_weights; do
  for dataset in cifar10 gtsrb; do
    for model in resnet vgg_bn; do
      echo "[BBCaL/hardware] $attack $dataset $model"
      python3 defenses/bbcal/test_bbcal_hamock_hardware.py \
        --attack "$attack" --dataset "$dataset" --model "$model" \
        --model_path "$checkpoints_dir" --dataset_dir "$dataset_dir" \
        --use_normalization 1 --use_gaussian_noise 0 --device "$device" --seed "$seed" \
        > "${log_dir}/bbcal_hardare_${attack}_${dataset}_${model}_seed${seed}.log" 2>&1
    done
  done
done

# 3) Hardware-deployed, multi-neuron attack
for dataset in cifar10 gtsrb; do
  for model in resnet vgg_bn; do
    echo "[BBCaL/hardware-sep] hamock_sep $dataset $model"
    python3 defenses/bbcal/test_bbcal_hamock_sep_hardware.py \
      --attack hamock_sep --dataset "$dataset" --model "$model" \
      --model_path "$checkpoints_dir" --dataset_dir "$dataset_dir" \
      --use_normalization 1 --use_gaussian_noise 0 --device "$device" --seed "$seed" \
      > "${log_dir}/bbcal_hardare_hamock_sep_${dataset}_${model}_seed${seed}.log" 2>&1
  done
done
