## TED on Software-Only Models (Pre-Hardware Deployment)

The following script evaluates TED against all HAMLOCK attack variants across supported datasets and model architectures.

```bash
dataset_dir="./data/"
checkpoints_dir="./checkpoints/"
log_dir="./defenses/TED/logs/"
seed=1

mkdir -p "${log_dir}"

for attack in hamock hamock_sep hamock_weights; do
    for dataset in cifar10 gtsrb; do
        for model in resnet vgg_bn; do
            echo "Running TED defense for dataset: $dataset, model: $model, attack: $attack, seed: $seed"

            python3 defenses/TED/ted.py \
                --model_path $checkpoints_dir \
                --device "cuda:0" \
                --dataset_dir $dataset_dir \
                --attack $attack \
                --model $model \
                --dataset $dataset \
                --batch_size 32 \
                --target 0 \
                --seed $seed \
                > "${log_dir}/ted_${attack}_${dataset}_${model}_seed${seed}.log" 2>&1
        done
    done
done
