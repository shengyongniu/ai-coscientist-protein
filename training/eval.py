"""Evaluate a fine-tuned ESM2 fitness checkpoint."""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from training.data import Collator, SeqRegressionDataset, load_pairs
from training.model import ESM2Regressor
from training.train import _spearman


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dataset", default="synthetic")
    p.add_argument("--split", default="test")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-examples", type=int, default=None)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, meta = ESM2Regressor.load(args.ckpt, device)
    model.eval()
    pairs = load_pairs(args.dataset, args.split, args.max_examples)
    loader = DataLoader(
        SeqRegressionDataset(pairs), batch_size=args.batch_size, collate_fn=Collator(tokenizer)
    )
    preds, tgts = [], []
    se = 0.0
    n = 0
    with torch.no_grad():
        for input_ids, attn, y in loader:
            out = model(input_ids.to(device), attn.to(device)).cpu()
            preds.extend(out.tolist())
            tgts.extend(y.tolist())
            se += ((out - y) ** 2).sum().item()
            n += y.numel()
    rmse = (se / max(n, 1)) ** 0.5
    print(f"[eval] n={n} rmse={rmse:.4f} spearman={_spearman(preds, tgts):.4f} base={meta['base_model']}")


if __name__ == "__main__":
    main()
