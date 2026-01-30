## IBD-PSC on Software-Only Models (Pre-Hardware Deployment)

The following script evaluates IBD-PSC against supported HAMLOCK attack variants across datasets and model architectures.

```bash
dataset_dir="./data/"
checkpoints_dir="./checkpoints/"
log_dir="./defenses/IBD-PSC/logs/"
seed=1

mkdir -p "${log_dir}"

for attack in hamock hamock_weights hamock_weights; do
    for dataset in cifar10 gtsrb; do
        for model in resnet vgg_bn; do
            echo "Running IBD-PSC defense for dataset: $dataset, model: $model, attack: $attack, seed: $seed"

            python3 defenses/IBD-PSC/ibd_psc.py \
                --model_path $checkpoints_dir \
                --dataset_dir $dataset_dir \
                --attack $attack \
                --model $model \
                --dataset $dataset \
                --use_normalization 1 \
                --target_label 0 \
                --seed $seed \
                > "${log_dir}/ibdpsc_${attack}_${dataset}_${model}_seed${seed}.log" 2>&1
        done
    done
done
```
