dataset_dir="./data/"
checkpoints_dir="./checkpoints/"
dataset="gtsrb" # Options: imagenet, cifar10, gtsrb, mnist
model="resnet"  # Options: resnet, vgg_bn, lenet


python3 main.py \
    --dataset_dir $dataset_dir \
    --dataset $dataset \
    --epochs 50 \
    --model $model \
    --device "cuda:3" \
    --target_label 0 \
    --inject 1 \
    --train_model 1 \
    --batch_size 256 \
    --model_path $checkpoints_dir \
    --dump_model 1 \
    --lam 0.1 \
	--use_normalization 1 \
    --seed 1