# CLP Defense against HAMLOCK

This directory evaluates **CLP** — *Channel Lipschitzness-based Pruning* (Zheng
et al., ECCV 2022) — against HAMLOCK. CLP is a **model-hardening** mitigation: for
each BatchNorm channel it estimates a channel Lipschitz constant (from the
combined conv+BN weights) and prunes the channels whose constant is an outlier
(`> mean + u·std`), on the assumption that backdoor channels are unusually
sensitive. No data or retraining is required.

HAMLOCK is expected to **survive** CLP: the trigger neuron and the hardware
payload are decoupled from the normal forward pass, so the trigger channel is not
a Lipschitz outlier and is not pruned. This supports claim~C4 (paper §5.2, Table 6
— column **CLP**).

The attack success rate is measured the same way as in `3N_attack.py`: the
hardware Trojan monitors a **fixed** set of neurons (chosen at injection time and
saved in the checkpoint), and ASR = the fraction of triggered inputs that flip an
MSB exponent of that monitored set. The same fixed candidate set is used before
and after pruning (CLP changes weights, not what the hardware watches).

---

## Files

| File | Description |
|------|-------------|
| `defense.py` | The CLP pruning algorithm (`CLP(net, u)`) |
| `hamock_test.py` | CLP against single-neuron attacks (`hamock`, `hamock_weights`) |
| `clp_sep.py` | CLP against the multi-neuron attack (`hamock_sep`) |
| `models_hamock.py` | Model builders |
| `utils.py`, `sep_utils.py` | Checkpoint loading, trigger / MSB-detector helpers |
| `run.sh` | Run script — single- and multi-neuron, with log capture |
| `experiments.sh` | Single-neuron run on ImageNet |
| `logs/` | Captured stdout for each configuration |

---

## Prerequisites

CLP loads the backdoored checkpoints produced by the **attack** stage (see the
top-level `README.md`, "Steps to run the attack"). Run those first so the
checkpoints exist under `--model_path`:

```
<model_path>/<attack>_<use_normalization>/<model>/<dataset>/model_<seed>.pth
```

| `--attack` | Script | Produced by |
|------------|--------|-------------|
| `hamock` | `hamock_test.py` | `main.py` (trigger optimization) |
| `hamock_weights` | `hamock_test.py` | `main_optimize_weights.py` (weight optimization) |
| `hamock_sep` | `clp_sep.py` | `3N_attack.py --save_model` (multi-neuron) |

Datasets download automatically into `--dataset_dir` (MNIST, CIFAR-10, GTSRB);
ImageNet must be placed manually.

---

## Running

Run from this directory. `-u` is the CLP pruning threshold (larger = more
conservative; the channel is pruned when its Lipschitz constant exceeds
`mean + u·std`).

### Single-neuron attacks (`hamock`, `hamock_weights`)

```bash
python3 hamock_test.py \
    --attack hamock \
    --model_path ./checkpoints \
    --dataset_dir ./data/ \
    --model resnet \
    --dataset cifar10 \
    -u 3 \
    --use_normalization 1 \
    --device cuda:0 \
    --seed 1
```

### Multi-neuron attack (`hamock_sep`)

```bash
python3 clp_sep.py \
    --attack hamock_sep \
    --model_path ./checkpoints \
    --dataset_dir ./data/ \
    --model resnet \
    --dataset cifar10 \
    -u 5 \
    --use_normalization 1 \
    --device cuda:0 \
    --seed 1
```

### Prebuilt scripts (with logs)

`run.sh` runs both the single-neuron attacks (`hamock`/`hamock_weights` via
`hamock_test.py`) and the multi-neuron attack (`hamock_sep` via `clp_sep.py`) over
`cifar10`/`gtsrb` × `resnet`/`vgg_bn`, writing one log per config to `logs/`.

```bash
bash defenses/CLP/run.sh
```

---

## Reading the output

Each run prints the clean accuracy and the monitored-neuron firing rate (ASR)
**before** and **after** pruning:

```
Before prunning
Validation accuracy: ...
[SANITY CHECK]: False positive rate on clean images: ...%
[SANITY CHECK]: True positive rate on triggered images: <ASR>%
Test clean accuracy: ...
Test attack success rate: <ASR>
After CLP prunning
[SANITY CHECK]: True positive rate on triggered images: <ASR>%
Test clean accuracy: ...
Test attack success rate: <ASR>
```

The **before** ASR (`True positive rate on triggered images`) should match the
ASR reported by the attack in the top-level `logs/3N_attack_*.log`. The **after**
ASR is the metric reported in Table 6.

---

## Expected results

HAMLOCK **survives** CLP — the ASR is essentially **unchanged** before vs. after
pruning, with clean accuracy maintained:

| config (hamock_sep) | ASR before | ASR after CLP | clean acc |
|---------------------|-----------|---------------|-----------|
| cifar10 / resnet | ~97% | ~97% | ~90.7% |
| cifar10 / vgg_bn | ~98% | ~98% | ~91.9% |
| gtsrb / resnet | ~93% | ~93% | ~92.8% |
| gtsrb / vgg_bn | ~96% | ~96% | ~95.4% |

(Single-neuron `hamock` / `hamock_weights` stay at ~100%.) This reproduces the
CLP column of Table 6 and confirms claim~C4. Small differences from the paper's
exact values come from single-seed runs (Table 6 averages 5 seeds) and the choice
of `-u`.
