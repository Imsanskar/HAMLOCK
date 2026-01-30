import argparse
from email import parser
from email import parser


def get_argument():
    parser = argparse.ArgumentParser()

    # Directory option
    parser.add_argument("--dataset_dir", type=str, default="../../data/")
    parser.add_argument("--checkpoints", type=str, default="../../checkpoints")
    parser.add_argument("--model_path", type=str, default="MNIST_backdoored_model.pth", help="Path to save/load the model")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--results", type=str, default="./results")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--model", type=str, default="resnet")
    parser.add_argument("--attack_mode", type=str, default="all2one")
    parser.add_argument("--temps", type=str, default="./temps")

    # ---------------------------- For TED --------------------------
    # Model hyperparameters
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--attack", type=str, default="hamock")
    parser.add_argument("--use_normalization", type=int, default=1)
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument('--neptune', action='store_true', help='use neptune to log results')
    parser.add_argument('--target', type=int, default=0, help='target label for the attack')

    return parser