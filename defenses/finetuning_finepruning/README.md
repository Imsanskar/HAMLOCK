# Finetuning-Finepruning: Execution Instruction 

### Single Neuron Attack
Use the following command to run either **fine-tuning** or **fine-pruning** on single-neuron HAMLOCK attacks.
```bash
checkpoints = "./checkpoints"

python expt_defense.py \
    --attack hamock \ # Options: hamock, hamock_weights
    --model_path $checkpoints \
    --model resnet \ # Options: resnet, vgg_bn
    --dataset cifar10 \
    --device cuda:1 \
    --batch_size 64 \
    --seed 1 \
    --exp finetuning \ # Options: finetuning, finepruning
    --epoch 50
```

### Multi-neuron Attack
The following command runs fine-tuning or fine-pruning against HAMLOCK multi-neuron attacks.
```bash
checkpoints_dir="./checkpoints/"

python3 expt_defense.py \
    --attack hamock_sep \        # Options: hamock_sep
    --model_path $checkpoints_dir \
    --model resnet \             # Options: resnet, vgg_bn
    --dataset cifar10 \
    --device "cuda:1" \
    --batch_size 64 \
    --seed 1 \
    --exp finetuning \           # Options: finetuning, finepruning
    --epoch 50

```

