# BBCaL Defense against HAMLOCK

This directory evaluates **BBCaL** — *Black-box Backdoor detection under the
Causality Lens* (Hu et al., TMLR 2024) — against HAMLOCK. BBCaL is a black-box,
input-level detector: it builds a "preflight batch" by **progressively adding
noise** to each input and records how the prediction changes. The **Flip
Position Score (FPS)** captures the noise level at which the prediction first
flips. Clean inputs flip at a moderate (median) noise level; backdoored inputs
flip either immediately or never, i.e. they have extreme FPS. A sample is flagged
as backdoored when its FPS falls **outside** the clean band `[α, β] = [1, 6]`.

HAMLOCK is evaluated in **two scenarios** (paper §5.3, Table 4 and Table 5):

1. **Software-only (pre-deployment)** — the dormant model; the backdoor lives in
   hardware, so the software model is behaviorally clean.
2. **Hardware-deployed** — the active model on the Trojaned hardware (the Trojan
   is functionally emulated inside the inference pipeline; no FPGA needed).

---

## Files

| File | Description |
|------|-------------|
| `test_bbcal_hamock.py` | BBCaL on the software-only model (all HAMLOCK variants) |
| `test_bbcal_hamock_hardware.py` | BBCaL on hardware-deployed single-neuron attacks (`hamock`, `hamock_weights`) |
| `test_bbcal_hamock_sep_hardware.py` | BBCaL on the hardware-deployed multi-neuron attack (`hamock_sep`) |
| `utils.py`, `sep_utils.py` | Trigger / filter-activation helpers |
| `run_bbcal.sh`, `run_bbcal_hardware.sh`, `run_bbcal_hardware_sep.sh` | Prebuilt sweep scripts |
| `logs/` | Captured stdout for each configuration |

---

## Prerequisites

BBCaL loads the backdoored checkpoints produced by the **attack** stage (see the
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

Run from the **repository root**. The simplest path is the prebuilt scripts
(each sweeps `cifar10`/`gtsrb` × `resnet`/`vgg_bn`, writing one log per config to
`logs/`):

```bash
bash defenses/bbcal/run_bbcal.sh              # software-only (hamock, hamock_weights, hamock_sep)
bash defenses/bbcal/run_bbcal_hardware.sh     # hardware, single-neuron (hamock, hamock_weights)
bash defenses/bbcal/run_bbcal_hardware_sep.sh # hardware, multi-neuron (hamock_sep)
```

To run a single configuration, e.g. software-only on CIFAR-10 / ResNet-18:

```bash
python3 defenses/bbcal/test_bbcal_hamock.py \
    --dataset cifar10 \
    --model resnet \
    --attack hamock \
    --model_path ./checkpoints/ \
    --dataset_dir ./data/ \
    --use_normalization 1 \
    --use_gaussian_noise 0 \
    --device cuda:0 \
    --seed 1
```

Swap the script (`test_bbcal_hamock_hardware.py` /
`test_bbcal_hamock_sep_hardware.py`) and `--attack` for the other scenarios.

### Key arguments

| Argument | Description |
|----------|-------------|
| `--attack` | `hamock`, `hamock_weights`, or `hamock_sep` |
| `--dataset` / `--model` | `cifar10`/`gtsrb`/`mnist`/`imagenet`; `resnet`/`vgg_bn`/`lenet` |
| `--model_path` | Root checkpoint directory (see Prerequisites) |
| `--use_gaussian_noise` | `0` = uniform noise (default, matches the reference BBCaL code), `1` = Gaussian |
| `--use_normalization` | Must match the attack (1) |


---

## Reading the output

Each run ends with a summary line followed by the headline metrics:

```
tn  fp  fn  tp  precision  recall  f1  AUROC
TPR:   ...
FPR:   ...
F1 Score: ...
AUROC: ...
```

The detection metric of interest is **AUROC**: how well the FPS separates
backdoor from clean inputs across all thresholds.

---

## Expected results

HAMLOCK **evades BBCaL** in both scenarios → **AUROC ≈ 0.5** (random). BBCaL may
show a high TPR, but only at the cost of an equally high FPR, so the AUROC stays
near 0.5 — exactly the failure mode described in the paper (§5.3). Across
`hamock`/`hamock_weights`/`hamock_sep` × CIFAR-10/GTSRB × ResNet-18/VGG-16,
observed AUROC lands in ~0.48–0.53. This reproduces Table 4 (software) and
Table 5 (hardware).
