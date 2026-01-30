## STRIP Defense: Execution Instructions

This section provides **execution instructions only** for running the STRIP defense against HAMLOCK under different attack and deployment settings. All paths and identifiers are anonymized and use **generic placeholders**, consistent with the rest of the repository documentation.

---

### 1. STRIP on Hardware-Deployed Models (Multi-Neuron)

This configuration evaluates STRIP against the multi-neuron attack attack on **hardware-deployed** models. Jobs are distributed across multiple GPUs.

```bash
attack="hamock_sep"
seed=1
itr=0

dataset_dir="./data/"
checkpoints_dir="./checkpoints/"
log_dir="./defenses/strip/logs/"
mkdir -p "${log_dir}"

for dataset in cifar10 gtsrb; do
    for model in resnet vgg_bn; do
        echo "Running STRIP defense for dataset: $dataset, model: $model, seed: $seed"
        itr=$((itr+1))

        python3 defenses/strip/strip_hardware_sep.py \
            --dataset_dir $dataset_dir \
            --model_path $checkpoints_dir \
            --attack $attack \
            --dataset $dataset \
            --model $model \
            --strip_mode 1 \
            --target_label 0 \
            --device "cuda:$((itr % 4))" \
            --n_sample 100 \
            --batch_size 32 \
            --n_benign_sample 500 \
            --use_normalization 1 \
            --seed $seed \
            > "${log_dir}/strip_hardware_${attack}_${dataset}_${model}_seed${seed}.log" 2>&1 &
    done
done

wait
```

### 2. STRIP on Hardware-Deployed Models (Single-Neuron Attack)

This configuration evaluates STRIP against single neuron attack on hardware-deployed models.
```bash
seed=1
dataset_dir="./data/"
checkpoints_dir="./checkpoints/"
log_dir="./defenses/strip/logs/"
mkdir -p "${log_dir}"

for attack in hamock hamock_weights; do
    for dataset in cifar10 gtsrb; do
        for model in resnet; do
            echo "Running STRIP defense for dataset: $dataset, model: $model, seed: $seed"

            python3 defenses/strip/strip_hardware.py \
                --dataset_dir $dataset_dir \
                --model_path $checkpoints_dir \
                --attack $attack \
                --dataset $dataset \
                --model $model \
                --strip_mode 1 \
                --target_label 0 \
                --device "cuda:0" \
                --n_sample 100 \
                --batch_size 32 \
                --n_benign_sample 500 \
                --use_normalization 1 \
                --seed $seed \
                > "${log_dir}/strip_hardware_${attack}_${dataset}_${model}_seed${seed}.log" 2>&1
        done
    done
done

```


### 3. STRIP on Software-Only Models (Pre-Hardware Deployment)
This configuration evaluates STRIP in a software-only setting against all HAMLOCK variants.
```bash
seed=1
dataset_dir="./data/"
checkpoints_dir="./checkpoints/"
log_dir="./defenses/strip/logs/"
mkdir -p "${log_dir}"
itr=0

for attack in hamock hamock_sep hamock_weights; do
    for dataset in cifar10 gtsrb; do
        for model in resnet vgg_bn; do
            echo "Running STRIP defense for dataset: $dataset, model: $model, seed: $seed"
            itr=$((itr+1))

            python3 defenses/strip/strip.py \
                --dataset_dir $dataset_dir \
                --model_path $checkpoints_dir \
                --attack $attack \
                --dataset $dataset \
                --model $model \
                --strip_mode 1 \
                --target_label 0 \
                --device "cuda:$((itr % 4))" \
                --n_sample 100 \
                --batch_size 32 \
                --n_benign_sample 500 \
                --use_normalization 1 \
                --seed $seed \
                > "${log_dir}/strip_${attack}_${dataset}_${model}_seed${seed}.log" 2>&1 &
        done
    done
done
```