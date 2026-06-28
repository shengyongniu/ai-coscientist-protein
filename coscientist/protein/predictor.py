"""Scorer backed by YOUR fine-tuned ESM2 regression head.

Loads a checkpoint produced by ``training/train.py`` and predicts a scalar
fitness/binding value per sequence. This is the highest-signal scorer and is the
point where the multi-GPU-trained model plugs into the agent loop.
"""

from __future__ import annotations

from coscientist.protein.scorer import Scorer, ScoreResult


class PredictorScorer(Scorer):
    name = "predictor"

    def __init__(self, checkpoint_path: str, device: str | None = None):
        import torch

        from training.model import ESM2Regressor

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.tokenizer, self.meta = ESM2Regressor.load(checkpoint_path, self.device)
        self.model.eval()

    def score(self, sequence: str) -> ScoreResult:
        seq = "".join(c for c in (sequence or "").upper() if c.isalpha())
        if not seq:
            return ScoreResult(score=-99.0, detail={"reason": "empty"})
        torch = self._torch
        with torch.no_grad():
            enc = self.tokenizer(
                seq, return_tensors="pt", truncation=True, max_length=1024
            ).to(self.device)
            pred = self.model(enc["input_ids"], enc["attention_mask"]).item()
        return ScoreResult(
            score=round(float(pred), 4),
            detail={
                "checkpoint": self.meta.get("task", "predictor"),
                "base_model": self.meta.get("base_model", ""),
                "metric": "predicted_fitness",
            },
        )
