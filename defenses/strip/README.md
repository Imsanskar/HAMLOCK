# STRIP Defense against HAMLOCK

This directory evaluates the **STRIP** black-box backdoor-sample detector
(Gao et al., ACSAC 2019) against HAMLOCK. STRIP superimposes each test input
with many clean images and measures the **entropy** of the resulting
predictions: a strongly backdoored input keeps predicting the target label
under superimposition (low entropy), whereas a clean input produces diverse
predictions (high entropy).

HAMLOCK is evaluated in **two scenarios** (paper §5.3, Table 4 and Table 5):

1. **Software-only (pre-deployment).** The dormant model, as inspected by a
   model-zoo maintainer. The backdoor lives in the hardware, so the software
   model is behaviorally clean.
2. **Hardware-deployed.** The active model on the Trojaned hardware, as tested
   by an end user. The hardware Trojan is functionally emulated inside the
   inference pipeline (no physical FPGA needed).

---

## Files

| File | Description |
|------|-------------|
| `strip.py` | STRIP on the software-only model (all HAMLOCK variants) |
| `strip_hardware.py` | STRIP on the hardware-deployed single-neuron attacks (`hamock`, `hamock_weights`) |
| `strip_hardware_sep.py` | STRIP on the hardware-deployed multi-neuron attack (`hamock_sep`) |
| `utils.py`, `sep_utils.py` | Trigger / filter-activation helpers (single- and multi-neuron) |
| `config.py` | Argument parser |
| `run_strip.sh`, `run_strip_hardware.sh`, `run_strip_hardware_sep.sh` | Prebuilt sweep scripts |
| `logs/` | Captured stdout for each configuration |

---

## Prerequisites

STRIP loads the backdoored checkpoints produced by the **attack** stage (see the
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

Datasets are downloaded automatically into `--dataset_dir` (MNIST, CIFAR-10,
GTSRB); ImageNet must be placed manually.

---

## Running

Run from the **repository root**. The simplest path is the prebuilt scripts —
each sweeps `cifar10`/`gtsrb` × `resnet`/`vgg_bn` and writes one log per config
into `logs/`:

```bash
bash defenses/strip/run_strip.sh              # software-only (hamock, hamock_weights, hamock_sep)
bash defenses/strip/run_strip_hardware.sh     # hardware, single-neuron (hamock, hamock_weights)
bash defenses/strip/run_strip_hardware_sep.sh # hardware, multi-neuron (hamock_sep)
```

To run a single configuration, e.g. software-only on CIFAR-10 / ResNet-18:

```bash
python3 defenses/strip/strip.py \
    --dataset_dir ./data/ \
    --model_path ./checkpoints/ \
    --attack hamock \
    --dataset cifar10 \
    --model resnet \
    --strip_mode 1 \
    --target_label 0 \
    --n_sample 100 \
    --n_benign_sample 500 \
    --batch_size 32 \
    --use_normalization 1 \
    --device cuda:0 \
    --seed 1
```

Swap the script (`strip_hardware.py` / `strip_hardware_sep.py`) and `--attack`
for the other scenarios. `--target_label` must match the attack's target (0).

### Key arguments

| Argument | Description |
|----------|-------------|
| `--attack` | `hamock`, `hamock_weights`, or `hamock_sep` |
| `--dataset` / `--model` | `cifar10`/`gtsrb`/`mnist`/`imagenet`; `resnet`/`vgg_bn`/`lenet` |
| `--model_path` | Root checkpoint directory (see Prerequisites) |
| `--n_sample` | Superimpositions per test image (entropy estimate) |
| `--n_benign_sample` | Clean images used to fit the entropy decision boundary |
| `--target_label` | Backdoor target class (0) |
| `--use_normalization` | Must match the attack (1) |

---

## Reading the output

Each run ends with one summary line:

```
tn  fp  fn  tp  f1  precision  recall  AUROC  clean_accuracy
```

`strip_hardware*.py` also print `TPR:` and `FPR:`. The detection metric of
interest is **AUROC** (8th value): STRIP scores each input by entropy and
AUROC measures how well that separates backdoor from clean inputs.

---

## Expected results

HAMLOCK **evades STRIP** in both scenarios → **AUROC ≈ 0.5** (random), with low
recall/TPR. The software-only model is behaviorally clean, so backdoor and clean
inputs yield indistinguishable entropy; on the Trojaned hardware, superimposing a
clean image perturbs the trigger so the payload fires inconsistently across the
`n_sample` overlays, keeping entropy high. This reproduces Table 4 (software) and
Table 5 (hardware). Across `hamock`/`hamock_weights`/`hamock_sep` × CIFAR-10/GTSRB
× ResNet-18/VGG-16, observed AUROC lands in ~0.47–0.53.
