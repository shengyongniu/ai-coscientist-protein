# Multi-GPU ESM2 Fitness Predictor

This directory contains the **multi-GPU training** component of the AI
Co-Scientist. It fine-tunes an [ESM2](https://huggingface.co/facebook/esm2_t12_35M_UR50D)
protein language model into a scalar **fitness/stability predictor**. The
resulting checkpoint plugs into the agent loop as the highest-signal protein
`Scorer`, so the Elo tournament is grounded in a fine-tuned model.

## Why fine-tune ESM2?

ESM2 is pretrained on ~65M protein sequences and "understands" protein space,
but is not specialized for any property. A small regression head on top of the
pooled embeddings turns it into a predictor of a concrete target (binding /
stability / fitness). This is the standard, fast-to-converge way to get a useful
protein model in hours rather than days, and it is a textbook **DDP / FSDP**
workload.

## Distributed strategies

- **DDP** (`--strategy ddp`): replicates the model on each GPU and all-reduces
  gradients. Best for models that fit on one GPU (e.g. ESM2-35M / 150M).
- **FSDP** (`--strategy fsdp`): sharded data parallel; shards params, grads, and
  optimizer state across GPUs. Use for the larger backbones (ESM2-650M / 3B)
  that don't fit replicated.

## Quickstart

CPU / single-process smoke test (synthetic data, no downloads):

```bash
pip install -r ../requirements-protein.txt
python -m training.train --dataset synthetic --epochs 1 --max-train 256 --max-valid 128 \
  --base-model facebook/esm2_t12_35M_UR50D --out checkpoints/smoke
```

Multi-GPU on a `g5.12xlarge` (4x A10G):

```bash
# DDP with ESM2-35M
./training/launch_multi_gpu.sh 4 ddp facebook/esm2_t12_35M_UR50D

# FSDP with ESM2-650M (sharded)
./training/launch_multi_gpu.sh 4 fsdp facebook/esm2_t33_650M_UR50D
```

Plot curves and evaluate:

```bash
python -m training.plot_curves --metrics checkpoints/esm2_fitness/metrics.json
python -m training.eval --ckpt checkpoints/esm2_fitness --dataset synthetic --split test
```

Use the trained model in the co-scientist:

```bash
export COSCIENTIST_SCORER=predictor
export COSCIENTIST_PREDICTOR_CKPT=checkpoints/esm2_fitness
coscientist run --config protein_binder
```

## Datasets

- `--dataset synthetic` (default): an offline, deterministic, *learnable*
  sequence->scalar task. Lets the full pipeline + CI run with no internet.
- `--dataset <hf_id>`: any HuggingFace dataset exposing a sequence column
  (`sequence`/`seq`/`primary`) and a target column (`target`/`label`/`fitness`).
  For real benchmarks see [FLIP](https://github.com/J-SNACKKB/FLIP) (binding /
  stability splits) and SKEMPI (mutation -> ddG).

## Outputs (under `--out`)

- `model.pt`, `meta.json` - the checkpoint loaded by `PredictorScorer`.
- `metrics.json` - per-epoch train loss, val RMSE/Spearman, epoch time, and
  **tokens/sec/GPU** + `world_size` (the multi-GPU scaling evidence).
- `curves.png` - training curves (after running `plot_curves`).

## Running on AWS

With 384 G/VT vCPU quota a `g5.12xlarge` (4x A10G, 48 vCPU) can be launched. See
[`../scripts/aws_train.md`](../scripts/aws_train.md) for a copy-paste launch +
train + teardown recipe that stays within the credit budget.
