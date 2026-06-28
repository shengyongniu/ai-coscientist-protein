"""Datasets for protein fitness/stability regression.

Primary path: load a FLIP-style sequence->scalar benchmark from the HuggingFace
Hub. Offline fallback: generate a synthetic but learnable dataset so the
training pipeline (and CI smoke tests) run with no internet and no downloads.
The synthetic target is a deterministic function of composition + motifs, so a
model can genuinely learn it and metrics are meaningful.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

_AA = "ACDEFGHIKLMNPQRSTVWY"
# A fixed "fitness" weight per amino acid + a couple of motif bonuses.
_W = {a: ((i % 7) - 3) * 0.15 for i, a in enumerate(_AA)}
_MOTIF_BONUS = {"GG": 0.4, "WW": -0.6, "KR": 0.3, "DE": 0.25}


def _synthetic_fitness(seq: str) -> float:
    if not seq:
        return 0.0
    base = sum(_W.get(c, 0.0) for c in seq) / len(seq)
    motif = sum(seq.count(m) * b for m, b in _MOTIF_BONUS.items()) / max(len(seq), 1)
    return base + motif


def make_synthetic(n: int, seed: int = 0, min_len: int = 40, max_len: int = 120) -> list[tuple[str, float]]:
    rng = random.Random(seed)
    data = []
    for _ in range(n):
        length = rng.randint(min_len, max_len)
        seq = "".join(rng.choice(_AA) for _ in range(length))
        y = _synthetic_fitness(seq) + rng.gauss(0, 0.02)
        data.append((seq, y))
    return data


def load_pairs(dataset: str, split: str, max_examples: int | None = None) -> list[tuple[str, float]]:
    """Return a list of (sequence, target) pairs.

    dataset == "synthetic" uses the offline generator. Otherwise we try to load
    a HF dataset with `sequence`/`target` (or `seq`/`label`) columns.
    """
    if dataset == "synthetic":
        n = {"train": 4000, "valid": 800, "test": 800}.get(split, 1000)
        pairs = make_synthetic(n, seed={"train": 1, "valid": 2, "test": 3}.get(split, 0))
        return pairs[:max_examples] if max_examples else pairs

    from datasets import load_dataset  # lazy import

    ds = load_dataset(dataset, split=split)
    seq_col = next((c for c in ("sequence", "seq", "protein", "primary") if c in ds.column_names), None)
    tgt_col = next((c for c in ("target", "label", "fitness", "value", "y") if c in ds.column_names), None)
    if seq_col is None or tgt_col is None:
        raise ValueError(f"Could not infer columns from {ds.column_names}")
    pairs = [(r[seq_col], float(r[tgt_col])) for r in ds]
    return pairs[:max_examples] if max_examples else pairs


@dataclass
class Collator:
    tokenizer: object
    max_length: int = 256

    def __call__(self, batch: list[tuple[str, float]]):
        seqs = [s for s, _ in batch]
        ys = torch.tensor([y for _, y in batch], dtype=torch.float32)
        enc = self.tokenizer(
            seqs, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length
        )
        return enc["input_ids"], enc["attention_mask"], ys


class SeqRegressionDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, float]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[str, float]:
        return self.pairs[idx]
