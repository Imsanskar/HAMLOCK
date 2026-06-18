#!/usr/bin/env bash
# Reproduce IBD-PSC results against HAMLOCK (claim C2, Table 3).
# Run from the repository root: bash defenses/IBD-PSC/run.sh
# Requires the backdoored checkpoints from the attack stage under ./checkpoints.
dataset_dir="./data/"
checkpoints_dir="./checkpoints/"
log_dir="./defenses/IBD-PSC/logs/"
device="cuda:0"
seed=1
mkdir -p "$log_dir"

for attack in hamock hamock_weights; do
  for dataset in cifar10 gtsrb; do
    for model in resnet vgg_bn; do
      echo "[IBD-PSC] $attack $dataset $model"
      python3 defenses/IBD-PSC/ibd_psc.py \
        --model_path "$checkpoints_dir" --dataset_dir "$dataset_dir" \
        --attack "$attack" --model "$model" --dataset "$dataset" \
        --use_normalization 1 --target_label 0 --device "$device" --seed "$seed" \
        > "${log_dir}/ibdpsc_${attack}_${dataset}_${model}_seed${seed}.log" 2>&1
    done
  done
done
