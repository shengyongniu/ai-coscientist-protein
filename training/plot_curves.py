"""Plot training curves from metrics.json (loss, val RMSE, val Spearman)."""

from __future__ import annotations

import argparse
import json
import os


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", default="checkpoints/esm2_fitness/metrics.json")
    p.add_argument("--out", default="checkpoints/esm2_fitness/curves.png")
    args = p.parse_args()

    data = json.load(open(args.metrics))
    hist = data["history"]
    epochs = [h["epoch"] for h in hist]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in hist], "-o")
    axes[0].set_title("Train loss (MSE)")
    axes[0].set_xlabel("epoch")
    axes[1].plot(epochs, [h["val_rmse"] for h in hist], "-o", color="tab:orange")
    axes[1].set_title("Val RMSE")
    axes[1].set_xlabel("epoch")
    axes[2].plot(epochs, [h["val_spearman"] for h in hist], "-o", color="tab:green")
    axes[2].set_title("Val Spearman")
    axes[2].set_xlabel("epoch")
    ws = hist[0].get("world_size", 1) if hist else 1
    tok = hist[-1].get("tokens_per_sec_per_gpu", 0) if hist else 0
    fig.suptitle(f"ESM2 fitness fine-tuning (world_size={ws}, ~{tok:,.0f} tok/s/gpu)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"[plot] wrote {args.out}")


if __name__ == "__main__":
    main()
