# TED Defense against HAMLOCK

This directory evaluates **TED** — *Topological Evolution Dynamics* (Mo et al.,
"Robust Backdoor Detection for Deep Learning via Topological Evolution Dynamics",
IEEE S&P 2024) — against HAMLOCK. TED is a **white-box**, input-level detector:
it treats the per-layer activations of a sample as a trajectory through the
network and, at each layer, records the **rank** at which the nearest neighbor
sharing the sample's predicted class appears among a benign reference set. Clean
inputs evolve smoothly across layers; backdoored inputs follow an anomalous
trajectory. A PCA outlier detector is fit on benign trajectories, and unknown
inputs are scored as in-/out-of-distribution.

Because TED needs intermediate activations from every layer, it is evaluated on
the **software-only** (dormant) model — the pre-deployment scenario inspected by
a model-zoo maintainer (paper §5.2, "Backdoor sample detection", Table 3). The
backdoor lives in the hardware, so the software model is behaviorally clean.

---

## Files

| File | Description |
|------|-------------|
| `ted.py` | TED detector and entry point |
| `config.py` | Argument parser |
| `run.sh` | Run script — all configs, with log capture |
| `logs/` | Captured stdout for each configuration |

---

## Prerequisites

TED loads the backdoored checkpoints produced by the **attack** stage (see the
top-level `README.md`, "Steps to run the attack"). Run those first so the
checkpoints exist under `--model_path`:

```
<model_path>/<attack>_<use_normalization>/<model>/<dataset>/model_<seed>.pth
```

| `--attack` | Produced by | Checkpoint dir (with `--use_normalization 1`) |
|------------|-------------|-----------------------------------------------|
| `hamock` | `main.py` (trigger optimization) | `hamock_1/` |
| `hamock_weights` | `main_optimize_weights.py` (weight optimization) | `hamock_weights_1/` |
| `hamock_sep` | `3N_attack.py --save_model` (multi-neuron) | `hamock_sep_1/` |

Datasets download automatically into `--dataset_dir` (MNIST, CIFAR-10, GTSRB);
ImageNet must be placed manually.

---

## Running

Run from the **repository root**. The prebuilt script sweeps
`hamock`/`hamock_weights`/`hamock_sep` × `cifar10`/`gtsrb` × `resnet`/`vgg_bn`,
writing one log per config into `logs/`:

```bash
bash defenses/TED/run.sh
```

To run a single configuration, e.g. CIFAR-10 / ResNet-18:

```bash
python3 defenses/TED/ted.py \
    --model_path ./checkpoints \
    --dataset_dir ./data/ \
    --attack hamock \
    --dataset cifar10 \
    --model resnet \
    --batch_size 32 \
    --target 0 \
    --device cuda:0 \
    --seed 1
```

### Key arguments

| Argument | Description |
|----------|-------------|
| `--attack` | `hamock`, `hamock_weights`, or `hamock_sep` |
| `--dataset` / `--model` | `cifar10`/`gtsrb`/`mnist`/`imagenet`; `resnet`/`vgg_bn`/`lenet` |
| `--model_path` | Root checkpoint directory (see Prerequisites) |
| `--target` | Backdoor target class (0) |
| `--batch_size` | Batch size for activation collection |

TED's defense set size and PCA contamination follow the reference defaults set in
`ted.py`.

---

## Reading the output

Each run prints sanity accuracies, then the headline metrics:

```
[SANITY_CHECK] Accuracy: ...
Accuracy on defense_loader: ...%
Accuracy on bd_loader: ...%
AUC: ...
TPR: ...
True Positives (TP): ...  False Positives (FP): ...
True Negatives (TN): ...   False Negatives (FN): ...
```

`Accuracy on bd_loader` is the rate at which triggered inputs are classified as
the target — low on the software model, confirming it is behaviorally clean. The
detection metric of interest is **AUC**: how well TED's trajectory outlier score
separates triggered (VT) from clean (NoT) inputs.

---

## Expected results

HAMLOCK **evades TED** → **AUC ≈ 0.5** (random), with low TPR. The software-only
model is behaviorally clean, so triggered and clean inputs follow
indistinguishable layer-wise trajectories. Across
`hamock`/`hamock_weights`/`hamock_sep` × CIFAR-10/GTSRB × ResNet-18/VGG-16,
observed AUC lands in ~0.44–0.52. This reproduces the TED column of Table 3.
