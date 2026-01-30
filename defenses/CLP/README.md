# Channel Lipschitzness-based Pruning for Backdoor Defense


### Single Neuron Attack
Use the following command to run either **CLP** on single-neuron HAMLOCK attacks.
```bash
checkpoints = "./checkpoints"

python3 hamock_test.py \
  --dataset cifar10 \ # Options: mnist, gtsrb, cifar10, imagenet 
  --model resnet \ # Options: lenet, resnet, vgg_bn
  --seed 1 \
  --u 3 \
  --attack hamock_sep \
  --use_normalization 1 \
  --dataset_dir ./data/ \
  --device "cuda:0" \
  --model_path $checkpoints
```

### Multi-neuron Attack
The following command runs **CLP** against HAMLOCK multi-neuron attacks.
```bash
checkpoints_dir="./checkpoints/"

python3 clp_sep.py \
  --dataset cifar10 \ # Options: mnist, gtsrb, cifar10, imagenet
  --model resnet \ # Options: lenet, resnet, vgg_bn
  --seed 1 \
  --u 3 \
  --attack hamock_sep \
  --use_normalization 1 \
  --dataset_dir ./data/ \
  --device "cuda:0" \
  --model_path $checkpoints
```
