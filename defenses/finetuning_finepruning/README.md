# Fine-tuning / Fine-pruning Defense against HAMLOCK

This directory evaluates the two lightweight **backdoor-mitigation** defenses
against HAMLOCK:

- **Fine-tuning (FT)** — retrain the backdoored model for a few epochs on clean
  data, hoping to overwrite the backdoor.
- **Fine-pruning (FP)** — prune the least-active neurons on clean data, then
  fine-tune.

Both are **model-hardening** defenses (they modify the model rather than detect
inputs). HAMLOCK is expected to **survive** them: because the trigger neuron and
the hardware payload are decoupled from the normal forward pass, retraining on
clean data does not disturb the trigger pathway. This supports claim~C4
(paper §5.2, "Effectiveness under lightweight retraining", Table 6 — columns
**FT** and **FP**).

The attack success rate after mitigation is measured by the **hardware-emulated
filter-activation rate**: the fraction of triggered inputs for which the
monitored neuron still fires (and would drive the hardware payload). This stays
high even though the software model's prediction-level ASR is ~0 (the software
model is behaviorally clean).

---

## Files

| File | Description |
|------|-------------|
| `expt_defense.py` | FT / FP against single-neuron attacks (`hamock`, `hamock_weights`) |
| `expt_defense_sep.py` | FT / FP against the multi-neuron attack (`hamock_sep`) |
| `defends/finetuning_finepruning.py` | `FineTuning` / `FinePruning` implementations |
| `utils.py`, `sep_utils.py` | Data loading and trigger / filter-activation helpers |
| `run_expt_defense.sh` | Prebuilt sweep script |
| `logs/` | Captured stdout for each configuration |

---

## Prerequisites

These defenses load the backdoored checkpoints produced by the **attack** stage
(see the top-level `README.md`, "Steps to run the attack"). Run those first so
the checkpoints exist under `--model_path`:

```
<model_path>/<attack>_<use_normalization>/<model>/<dataset>/model_<seed>.pth
```

| `--attack` | Script | Produced by |
|------------|--------|-------------|
| `hamock` | `expt_defense.py` | `main.py` (trigger optimization) |
| `hamock_weights` | `expt_defense.py` | `main_optimize_weights.py` (weight optimization) |
| `hamock_sep` | `expt_defense_sep.py` | `3N_attack.py --save_model` (multi-neuron) |

Datasets download automatically into `--dataset_dir` (MNIST, CIFAR-10, GTSRB);
ImageNet must be placed manually.

---

## Running

Run from the **repository root**.

### Single-neuron attacks (`hamock`, `hamock_weights`)

```bash
python3 defenses/finetuning_finepruning/expt_defense.py \
    --attack hamock \
    --model_path ./checkpoints \
    --dataset_dir ./data \
    --model resnet \
    --dataset cifar10 \
    --exp finetuning \
    --batch_size 128 \
    --epoch 50 \
    --use_normalization 1 \
    --device cuda:0 \
    --seed 1
```

- `--exp` selects the method: `finetuning` (FT) or `finepruning` (FP).
- `--model`: `resnet` or `vgg_bn`; `--dataset`: `cifar10` or `gtsrb`.

### Multi-neuron attack (`hamock_sep`)

Use the **`_sep`** script — the multi-neuron checkpoint stores its weights under
`net` (with `injection_params`), which `expt_defense.py` does not read:

```bash
python3 defenses/finetuning_finepruning/expt_defense_sep.py \
    --attack hamock_sep \
    --model_path ./checkpoints \
    --dataset_dir ./data \
    --model resnet \
    --dataset cifar10 \
    --exp finetuning \
    --batch_size 128 \
    --epoch 50 \
    --use_normalization 1 \
    --device cuda:0 \
    --seed 1
```

### Sweep script

`run_expt_defense.sh` sweeps the single-neuron attacks. Edit the `method`
(`finetuning`/`finepruning`), the attack list, and the model/dataset loops at the
top; for `hamock_sep` swap in `expt_defense_sep.py`. Logs are written to `logs/`.

```bash
bash defenses/finetuning_finepruning/run_expt_defense.sh
```

---

## Reading the output

Fine-tuning prints per-epoch `ACC`/`ASR`, then a final summary; fine-pruning
prints the final summary directly:

```
[SANITY_CHECK] Accuracy: ..., <acc>, <asr>
Accuracy: <clean_acc>, ASR: <asr>
```

- The per-epoch `ASR` (~0.10) is the **prediction-level** ASR on the software
  model — it stays near chance because the dormant model is behaviorally clean.
- The final `ASR` is the **hardware-emulated** filter-activation rate — the
  metric reported in Table 6. It stays high, showing the trigger pathway survives
  mitigation.

---

## Expected results

HAMLOCK **survives** both FT and FP (Table 6):

| attack | ASR after FT/FP | clean accuracy |
|--------|-----------------|----------------|
| `hamock`, `hamock_weights` (1N) | ≈ 1.00 (100%) | maintained (within a few % of clean) |
| `hamock_sep` (3N) | ≈ 0.90–0.98 | maintained |

Observed across CIFAR-10/GTSRB × ResNet-18/VGG-16: 1N ASR = 1.0, 3N ASR ≈
0.92–0.98, clean accuracy ≈ 0.88–0.95. This reproduces the FT and FP columns of
Table 6 and confirms claim~C4.
