"""ESM2 backbone + scalar regression head for protein fitness prediction.

The head mean-pools ESM2 token embeddings (masked by attention) and maps to a
single scalar. Designed to be FSDP/DDP friendly: the backbone is a standard HF
transformer whose layers can be sharded/wrapped.
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class ESM2Regressor(nn.Module):
    def __init__(self, base_model: str = "facebook/esm2_t12_35M_UR50D", dropout: float = 0.1):
        super().__init__()
        self.base_model_name = base_model
        self.backbone = AutoModel.from_pretrained(base_model)
        hidden = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # [B, T, H]
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.head(pooled).squeeze(-1)  # [B]

    # ---- persistence ----
    def save(self, path: str, meta: dict[str, Any] | None = None) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))
        meta = {"base_model": self.base_model_name, **(meta or {})}
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str, device: str = "cpu"):
        with open(os.path.join(path, "meta.json")) as f:
            meta = json.load(f)
        model = cls(base_model=meta["base_model"])
        state = torch.load(os.path.join(path, "model.pt"), map_location=device)
        model.load_state_dict(state)
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(meta["base_model"])
        return model, tokenizer, meta
