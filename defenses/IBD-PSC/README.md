# IBD-PSC Defense against HAMLOCK

This directory evaluates **IBD-PSC** — *Input-level Backdoor Detection via
Parameter-oriented Scaling Consistency* (Hou et al., ICML 2024,
[arXiv:2405.09786](https://arxiv.org/abs/2405.09786)) — against HAMLOCK. IBD-PSC
is a **white-box**, input-level detector: it amplifies the affine parameters of
the model's BatchNorm layers (scaling them up) and measures the **Parameter-oriented
Scaling Consistency (PSC)** — the average softmax confidence the scaled models
still assign to the original prediction. Backdoored inputs keep predicting the
target with high confidence under scaling (high PSC); clean inputs lose
confidence (low PSC). A sample with `PSC ≥ T` is flagged as backdoored.

Because IBD-PSC needs the model internals (BN layers), it is evaluated on the
**software-only** (dormant) model — the pre-deployment scenario inspected by a
model-zoo maintainer (paper §5.2, "Backdoor sample detection", Table 3). The
backdoor lives in the hardware, so the software model is behaviorally clean.

---

## Files

| File | Description |
|------|-------------|
| `ibd_psc.py` | IBD-PSC detector and entry point |
| `base.py` | Seeding / determinism base class |
| `config.py` | Argument parser |
| `run_ibdpsc.sh` | Prebuilt sweep script |
| `logs/` | Captured stdout for each configuration |

---

## Prerequisites

IBD-PSC loads the backdoored checkpoints produced by the **attack** stage (see
the top-level `README.md`, "Steps to run the attack"). Run those first so the
checkpoints exist under `--model_path`:

```
<model_path>/<attack>_<use_normalization>/<model>/<dataset>/model_<seed>.pth
```

| `--attack` | Produced by | Checkpoint dir (with `--use_normalization 1`) |
|------------|-------------|-----------------------------------------------|
| `hamock` | `main.py` (trigger optimization) | `hamock_1/` |
| `hamock_weights` | `main_optimize_weights.py` (weight optimization) | `hamock_weights_1/` |

Datasets download automatically into `--dataset_dir` (MNIST, CIFAR-10, GTSRB);
ImageNet must be placed manually.

---

## Running

Run from the **repository root**. The prebuilt script sweeps
`hamock`/`hamock_weights` × `cifar10`/`gtsrb` × `resnet`/`vgg_bn`, writing one log
per config into `logs/`:

```bash
bash defenses/IBD-PSC/run_ibdpsc.sh
```

To run a single configuration, e.g. CIFAR-10 / ResNet-18:

```bash
python3 defenses/IBD-PSC/ibd_psc.py \
    --model_path ./checkpoints/ \
    --dataset_dir ./data/ \
    --attack hamock \
    --dataset cifar10 \
    --model resnet \
    --use_normalization 1 \
    --target_label 0 \
    --device cuda:0 \
    --seed 1
```

### Key arguments

| Argument | Description |
|----------|-------------|
| `--attack` | `hamock` or `hamock_weights` |
| `--dataset` / `--model` | `cifar10`/`gtsrb`/`mnist`/`imagenet`; `resnet`/`vgg_bn`/`lenet` |
| `--model_path` | Root checkpoint directory (see Prerequisites) |
| `--n_sample` | Number of clean validation images used to calibrate the BN-scaling start layer |
| `--target_label` | Backdoor target class (0) |
| `--use_normalization` | Must match the attack (1) |

The detector's own hyperparameters (`n` scaled models, error rate `xi`, threshold
`T`, scale factor) are set in `ibd_psc.py` and follow the reference defaults.

---

## Reading the output

Each run prints a sanity accuracy, the selected BN-scaling start index, then the
summary line followed by the headline metrics:

```
[SANITY_CHECK] Accuracy: ...
<auroc> <tn> <fp> <fn> <tp> <f1> <precision> <recall>
TPR:   ...
FPR:   ...
F1 Score: ...
AUROC: ...
```

The detection metric of interest is **AUROC**: how well the PSC score separates
backdoor from clean inputs.

---

## Expected results

HAMLOCK **evades IBD-PSC** → **AUROC ≈ 0.5** (random), with low recall/TPR. The
software-only model is behaviorally clean, so backdoored and clean inputs yield
indistinguishable PSC scores under BN scaling. Across
`hamock`/`hamock_weights` × CIFAR-10/GTSRB × ResNet-18/VGG-16, observed AUROC
lands in ~0.43–0.50. This reproduces the IBD-PSC column of Table 3.
