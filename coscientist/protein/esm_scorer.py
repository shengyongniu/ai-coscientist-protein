"""ESM2 pseudo-log-likelihood scorer.

Loads a HuggingFace ESM2 masked-LM and scores a sequence by its average
pseudo-log-likelihood (PLL): for each position, mask it and read the log-prob of
the true residue. Higher PLL = the sequence is more 'natural' to the model,
which correlates with stability/foldability. Lazy-imports torch so the package
works without it installed.
"""

from __future__ import annotations

from coscientist.protein.scorer import Scorer, ScoreResult


class ESMScorer(Scorer):
    name = "esm"

    def __init__(self, model_name: str | None = None, device: str | None = None, batch_tokens: int = 1):
        import os

        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.model_name = model_name or os.getenv(
            "COSCIENTIST_ESM_MODEL", "facebook/esm2_t12_35M_UR50D"
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name).to(self.device).eval()

    def score(self, sequence: str) -> ScoreResult:
        seq = "".join(c for c in (sequence or "").upper() if c.isalpha())
        if not seq:
            return ScoreResult(score=-99.0, detail={"reason": "empty"})
        torch = self._torch
        with torch.no_grad():
            enc = self.tokenizer(seq, return_tensors="pt").to(self.device)
            input_ids = enc["input_ids"]
            # Score interior tokens (skip BOS/EOS).
            positions = list(range(1, input_ids.shape[1] - 1))
            total = 0.0
            mask_id = self.tokenizer.mask_token_id
            # Batch masked positions for speed.
            n = len(positions)
            if n == 0:
                return ScoreResult(score=-99.0, detail={"reason": "too_short"})
            batch = input_ids.repeat(n, 1)
            for row, pos in enumerate(positions):
                batch[row, pos] = mask_id
            logits = self.model(batch).logits
            logprobs = torch.log_softmax(logits, dim=-1)
            for row, pos in enumerate(positions):
                true_id = input_ids[0, pos]
                total += logprobs[row, pos, true_id].item()
            pll = total / n
        return ScoreResult(
            score=round(pll, 4),
            detail={"model": self.model_name, "length": len(seq), "metric": "avg_pseudo_loglik"},
        )
