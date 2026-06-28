"""Multi-GPU fine-tuning of an ESM2 protein fitness predictor.

Supports two distributed strategies:
  - ddp  : DistributedDataParallel (default, replicates the model per GPU).
  - fsdp : FullyShardedDataParallel (shards params/grads/optimizer; for larger
           ESM2 like 650M/3B that don't fit replicated).

Launch with torchrun, e.g. 4 GPUs:
    torchrun --standalone --nproc_per_node=4 -m training.train \
        --dataset synthetic --base-model facebook/esm2_t12_35M_UR50D \
        --strategy ddp --epochs 3 --batch-size 16 --out checkpoints/esm2_fitness

Single process (CPU/1 GPU) also works without torchrun:
    python -m training.train --dataset synthetic --epochs 1 --max-train 256

Throughput, scaling, loss/val curves, and checkpoints are written under --out.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from training.data import Collator, SeqRegressionDataset, load_pairs
from training.model import ESM2Regressor


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank0() -> bool:
    return (not is_dist()) or dist.get_rank() == 0


def setup_distributed() -> tuple[int, int, int, torch.device]:
    """Initialize the process group if launched under torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", 0))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world)
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
            device = torch.device(f"cuda:{local}")
        else:
            device = torch.device("cpu")
        return rank, world, local, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return 0, 1, 0, device


def wrap_model(model: nn.Module, strategy: str, device: torch.device, local_rank: int) -> nn.Module:
    model = model.to(device)
    if not is_dist():
        return model
    if strategy == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

        policy = functools.partial(size_based_auto_wrap_policy, min_num_params=1_000_000)
        return FSDP(
            model,
            auto_wrap_policy=policy,
            device_id=local_rank if torch.cuda.is_available() else None,
        )
    # default DDP. find_unused_parameters=True because the ESM2 backbone's
    # `pooler` submodule is not exercised by our mean-pooled regression head,
    # so those params receive no gradient and would otherwise stall DDP's
    # gradient reduction.
    return DDP(
        model,
        device_ids=[local_rank] if torch.cuda.is_available() else None,
        find_unused_parameters=True,
    )


def all_reduce_mean(value: float, device: torch.device) -> float:
    if not is_dist():
        return value
    t = torch.tensor([value], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / dist.get_world_size()).item()


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    se = 0.0
    n = 0
    preds, tgts = [], []
    for input_ids, attn, y in loader:
        input_ids, attn, y = input_ids.to(device), attn.to(device), y.to(device)
        out = model(input_ids, attn)
        se += ((out - y) ** 2).sum().item()
        n += y.numel()
        preds.extend(out.detach().cpu().tolist())
        tgts.extend(y.detach().cpu().tolist())
    mse = se / max(n, 1)
    mse = all_reduce_mean(mse, device)
    spearman = _spearman(preds, tgts)
    return {"mse": mse, "rmse": mse**0.5, "spearman": spearman}


def _spearman(a: list[float], b: list[float]) -> float:
    if len(a) < 2:
        return 0.0
    try:
        from scipy.stats import spearmanr

        r = spearmanr(a, b).correlation
        return float(r) if r == r else 0.0  # guard NaN
    except Exception:
        # Rank-correlation fallback without scipy.
        import statistics

        def rank(x):
            order = sorted(range(len(x)), key=lambda i: x[i])
            r = [0] * len(x)
            for pos, idx in enumerate(order):
                r[idx] = pos
            return r

        ra, rb = rank(a), rank(b)
        ma, mb = statistics.mean(ra), statistics.mean(rb)
        num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=False))
        den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
        return num / den if den else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-GPU ESM2 fitness fine-tuning")
    p.add_argument("--dataset", default="synthetic", help="'synthetic' or a HF dataset id")
    p.add_argument("--base-model", default="facebook/esm2_t12_35M_UR50D")
    p.add_argument("--strategy", choices=["ddp", "fsdp"], default="ddp")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-valid", type=int, default=None)
    p.add_argument("--out", default="checkpoints/esm2_fitness")
    p.add_argument("--resume", action="store_true", help="resume from --out if present")
    p.add_argument("--log-every", type=int, default=10)
    args = p.parse_args()

    rank, world, local_rank, device = setup_distributed()
    if rank0():
        os.makedirs(args.out, exist_ok=True)
        print(f"[setup] world_size={world} strategy={args.strategy} device={device} base={args.base_model}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    collate = Collator(tokenizer, max_length=args.max_length)

    train_pairs = load_pairs(args.dataset, "train", args.max_train)
    valid_pairs = load_pairs(args.dataset, "valid", args.max_valid)
    train_ds = SeqRegressionDataset(train_pairs)
    valid_ds = SeqRegressionDataset(valid_pairs)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_dist() else None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        shuffle=train_sampler is None, collate_fn=collate, num_workers=2, drop_last=False,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=args.batch_size, collate_fn=collate, num_workers=2
    )

    model = ESM2Regressor(base_model=args.base_model)
    model = wrap_model(model, args.strategy, device, local_rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    history: list[dict] = []
    start_epoch = 0
    metrics_path = os.path.join(args.out, "metrics.json")
    if args.resume and os.path.exists(metrics_path) and rank0():
        history = json.load(open(metrics_path)).get("history", [])
        start_epoch = len(history)
        print(f"[resume] continuing from epoch {start_epoch}")

    global_seen = 0
    t_train0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        running = 0.0
        steps = 0
        ep_tokens = 0
        ep_t0 = time.time()
        for step, (input_ids, attn, y) in enumerate(train_loader):
            input_ids, attn, y = input_ids.to(device), attn.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(input_ids, attn)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            running += loss.item()
            steps += 1
            ep_tokens += int(attn.sum().item())
            global_seen += y.numel() * world
            if rank0() and step % args.log_every == 0:
                dt = time.time() - ep_t0
                tok_s = ep_tokens / dt if dt > 0 else 0.0
                print(
                    f"[train] epoch {epoch} step {step} loss {loss.item():.4f} "
                    f"tok/s/gpu {tok_s:,.0f} samples_seen {global_seen}"
                )

        ep_dt = time.time() - ep_t0
        tok_s = all_reduce_mean(ep_tokens / ep_dt if ep_dt > 0 else 0.0, device)
        val = evaluate(model, valid_loader, device)
        rec = {
            "epoch": epoch,
            "train_loss": running / max(steps, 1),
            "val_mse": val["mse"],
            "val_rmse": val["rmse"],
            "val_spearman": val["spearman"],
            "epoch_seconds": ep_dt,
            "tokens_per_sec_per_gpu": tok_s,
            "world_size": world,
        }
        history.append(rec)
        if rank0():
            print(f"[eval] epoch {epoch} {rec}")
            json.dump(
                {"args": vars(args), "history": history,
                 "total_train_seconds": time.time() - t_train0},
                open(metrics_path, "w"), indent=2,
            )
            # Save checkpoint (unwrap DDP/FSDP).
            to_save = _unwrap(model, args.strategy)
            if to_save is not None:
                to_save.save(args.out, meta={"task": args.dataset, "base_model": args.base_model})

    if rank0():
        print(f"[done] best val spearman {max((h['val_spearman'] for h in history), default=0):.4f}")
        print(f"[done] checkpoint at {args.out}")
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


def _unwrap(model, strategy: str):
    """Return a plain ESM2Regressor for saving (rank0 only)."""
    if strategy == "fsdp" and is_dist():
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            state = model.state_dict()
        base = ESM2Regressor(base_model=model.module.base_model_name if hasattr(model, "module") else model.base_model_name)
        base.load_state_dict(state)
        return base
    inner = model.module if hasattr(model, "module") else model
    return inner


if __name__ == "__main__":
    main()
