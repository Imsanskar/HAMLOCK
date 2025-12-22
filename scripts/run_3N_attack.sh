dataset_dir="./data/"
dataset="cifar10"
seed=1
model="resnet"  # Options: resnet, vgg, lenet
checkpoints_dir="./checkpoints/clean_models_1/${model}/${dataset}/model_${seed}.pth"


python3 3N_attack.py \
    --dataset_dir $dataset_dir \
    --dataset $dataset \
    --model $model \
    --device "cuda:0" \
    --batch_size 256 \
    --model_path "${checkpoints_dir}" \
    --seed $seed
