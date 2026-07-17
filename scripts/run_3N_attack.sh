# Dataset/model pairings: mnist -> lenet ; {cifar10, gtsrb, imagenet} -> {resnet, vgg_bn}.
dataset_dir="./data/"
dataset="mnist"
seed=1
model="lenet"  # Options: resnet, vgg_bn, lenet (lenet only with dataset=mnist)
checkpoints_dir="./checkpoints/clean_models_1/${model}/${dataset}/model_${seed}.pth"


python3 3N_attack.py \
    --dataset_dir $dataset_dir \
    --dataset $dataset \
    --model $model \
    --device "cuda:1" \
    --batch_size 256 \
    --model_path "${checkpoints_dir}" \
    --seed $seed \
    --save_dir "./checkpoints/" \
    --save_model
