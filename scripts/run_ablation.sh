
dataset="cifar10"
seed=1
model="resnet"
checkpoints_path="./checkpoints/clean_models_1/${model}/${dataset}/model_${seed}.pth"

python3 ablation.py \
    --dataset $dataset \
    --model $model \
    --device "cuda:0" \
    --batch_size 256 \
    --model_path "${checkpoints_path}" \
    --neuron_ablation 