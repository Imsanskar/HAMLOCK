dataset_dir="./data/"
checkpoints_dir="./checkpoints/"
# Dataset/model pairings: mnist -> lenet ; {cifar10, gtsrb, imagenet} -> {resnet, vgg_bn}.
# Example MNIST/LeNet run: set dataset="mnist" and model="lenet" (seed=1 below).
dataset="mnist" # Options: imagenet, cifar10, gtsrb, mnist
model="lenet"  # Options: resnet, vgg_bn, lenet


python3 main.py \
    --dataset_dir $dataset_dir \
    --dataset $dataset \
    --epochs 100 \
    --model $model \
    --device "cuda:0" \
    --target_label 0 \
    --inject 1 \
    --train_model 0 \
    --batch_size 256 \
    --model_path $checkpoints_dir \
    --dump_model 1 \
    --lam 0.01 \
    --threshold 0.0 \
	--use_normalization 0 \
    --seed 1