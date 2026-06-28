"""Training package: multi-GPU fine-tuning of an ESM2 protein fitness predictor.

Entry points:
  - train.py : DDP / FSDP fine-tuning with throughput logging + checkpointing.
  - eval.py  : evaluate a checkpoint (Spearman / RMSE).
  - data.py  : dataset loaders (FLIP-style + synthetic offline fallback).
  - model.py : ESM2 backbone + scalar regression head, with save/load.
"""
