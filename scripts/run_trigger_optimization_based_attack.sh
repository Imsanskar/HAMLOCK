dataset_dir="./data/"
checkpoints_dir="./checkpoints/"
dataset="cifar10"
model="resnet"  # Options: resnet, vgg, lenet


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
    --lam 0.1 \
	--use_normalization 1 \
    --seed 1