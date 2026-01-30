
# BEAGLE: Forensics of Deep Learning Backdoor Attack for Better Defense



## Requirement
Please download the pre-trained StyleGAN model from the following link:
[Download Pre-trained Model](https://drive.google.com/file/d/1qoOcM77h-MZLBzHzFzB3nGPF2AE4Jndm/view?usp=sharing)

After downlowding, place it in the `./checkpoints` directory. This value is hardcoded which can be modified in `cifar10/stylegan.py`. 

This model is fine-tuned from [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada) by NVlabs. Special acknowledgment!



## Executing for Single Neuron Attack 

```bash
dataset=cifar10 # Options: mnist, cifar10, gtsrb, imagenet
model=resnet # lenet, resnet, vgg_bn
checkpoints='./checkpoints'
attack=hamock  # Options: hamock, hamock_weights
seed=1

python3 cifar10/decomposition.py \
  --gpu 3 \
  --dataset_dir ./data \
  --model_path $checkpoints \
  --dataset=$dataset \
  --network=$model \
  --attack $attack \
  --target 0 \
  --n_clean 100 \
  --n_poison 10 \
  --func mask \
  --func_option binomial \
  --save_folder forensics \
  --use_normalization 1 \
  --verbose 1 \
  --epochs 1000 \
  --seed $seed


python cifar10/backdoor_removal.py \
    --gpu 0 \
    --dataset_dir ./data \
    --model_path $checkpoints \
    --dataset=$dataset \
    --network=$model \
    --attack $attack \
    --ratio 0.01 \
    --batch_size 512 \
    --lr 0.01 \
    --epochs 50 \
    --seed $seed
```

## Executing for Multi-Neuron Attack 




```bash
dataset=cifar10 # Options: mnist, cifar10, gtsrb, imagenet
model=resnet # lenet, resnet, vgg_bn
checkpoints='./checkpoints'
attack=hamock_sep
seed=1

python3 cifar10/decomposition_sep.py \
  --gpu 3 \
  --dataset_dir ./data \
  --model_path $checkpoints \
  --dataset=$dataset \
  --network=$model \
  --attack $attack \
  --target 0 \
  --n_clean 100 \
  --n_poison 10 \
  --func mask \
  --func_option binomial \
  --save_folder forensics \
  --use_normalization 1 \
  --verbose 1 \
  --epochs 1000 \
  --seed $seed


python ./defenses/beagle/cifar10/backdoor_removal_sep.py \
    --gpu 0 \
    --dataset_dir ./data \
    --model_path $checkpoints \
    --dataset=$dataset \
    --network=$model \
    --attack $attack \
    --ratio 0.01 \
    --batch_size 512 \
    --lr 0.01 \
    --epochs 50 \
    --seed $seed
  ```
