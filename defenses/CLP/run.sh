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

# MNIST / LeNet (seed 1). LeNet (Arch A) is a 2-conv net; hamock_sep places its
# 3 trigger neurons as 2 in cnn.0 + 1 in cnn.2.
for attack in hamock hamock_weights; do
  echo "[CLP/single] $attack mnist lenet"
  python3 defenses/CLP/hamock_test.py \
    -u 10 --attack "$attack" --model_path "$checkpoints_dir" --dataset_dir "$dataset_dir" \
    --model lenet --dataset mnist --use_normalization 1 --device "$device" --seed "$seed" \
    > "${log_dir}/clp_${attack}_mnist_lenet_seed${seed}.log" 2>&1
done
echo "[CLP/sep] hamock_sep mnist lenet"
python3 defenses/CLP/clp_sep.py \
  -u 5 --attack hamock_sep --model_path "$checkpoints_dir" --dataset_dir "$dataset_dir" \
  --model lenet --dataset mnist --use_normalization 1 --device "$device" --seed "$seed" \
  > "${log_dir}/clp_hamock_sep_mnist_lenet_seed${seed}.log" 2>&1
