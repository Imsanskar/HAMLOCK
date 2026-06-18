#!/usr/bin/env bash
# Reproduce fine-tuning / fine-pruning results against HAMLOCK (claim C4, Table 6).
# Run from the repository root: bash defenses/finetuning_finepruning/run.sh
# Requires the backdoored checkpoints from the attack stage under ./checkpoints.
dataset_dir="./data"
checkpoints_dir="./checkpoints"
log_dir="./defenses/finetuning_finepruning/logs"
device="cuda:0"
seed=1
mkdir -p "$log_dir"

# Single-neuron attacks: fine-tuning (FT) and fine-pruning (FP)
for method in finetuning finepruning; do
  for attack in hamock hamock_weights; do
    for dataset in cifar10 gtsrb; do
      for model in resnet vgg_bn; do
        echo "[$method] $attack $dataset $model"
        python3 defenses/finetuning_finepruning/expt_defense.py \
          --attack "$attack" --model_path "$checkpoints_dir" --dataset_dir "$dataset_dir" \
          --model "$model" --dataset "$dataset" --device "$device" \
          --batch_size 128 --epoch 50 --exp "$method" --use_normalization 1 --seed "$seed" \
          > "${log_dir}/${method}_${attack}_${dataset}_${model}_seed${seed}.log" 2>&1
      done
    done
  done
done

# Multi-neuron attack: fine-tuning (FT) and fine-pruning (FP)
for method in finetuning finepruning; do
  for dataset in cifar10 gtsrb; do
    for model in resnet vgg_bn; do
      echo "[$method] hamock_sep $dataset $model"
      python3 defenses/finetuning_finepruning/expt_defense_sep.py \
        --attack hamock_sep --model_path "$checkpoints_dir" --dataset_dir "$dataset_dir" \
        --model "$model" --dataset "$dataset" --device "$device" \
        --batch_size 128 --epoch 50 --exp "$method" --use_normalization 1 --seed "$seed" \
        > "${log_dir}/${method}_hamock_sep_${dataset}_${model}_seed${seed}.log" 2>&1
    done
  done
done
