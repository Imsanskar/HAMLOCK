import argparse


def get_argument():
    parser = argparse.ArgumentParser()

    # Directory option
    parser.add_argument("--dataset_dir", type=str, default="../../data/")
    parser.add_argument("--checkpoints", type=str, default="../../checkpoints")
    parser.add_argument("--model_path", type=str, default="MNIST_backdoored_model.pth", help="Path to save/load the model")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--results", type=str, default="./results")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--model", type=str, default="resnet")
    parser.add_argument("--attack_mode", type=str, default="all2one")
    parser.add_argument("--temps", type=str, default="./temps")

    # ---------------------------- For STRIP --------------------------
    # Model hyperparameters
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--n_sample", type=int, default=100)
    parser.add_argument("--n_benign_sample", type=int, default=100)
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--frr", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--test_rounds", type=int, default=10)

    parser.add_argument("--true_target_label", type=int)
    parser.add_argument("--target_label", type=int, default=9)
    parser.add_argument('--gpu', default='0', type=str, help='the index of gpu used to train the model')
    parser.add_argument("--strip_mode", default='1',type=str)


    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--attack", type=str, default="hamock")
    parser.add_argument("--use_normalization", type=int, default=1)
    parser.add_argument('--neptune', action='store_true', help='whether to use neptune to log training info')

    return parser