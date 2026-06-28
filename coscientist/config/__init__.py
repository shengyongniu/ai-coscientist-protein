"""Configuration loading and the typed config object."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

CONFIG_DIR = Path(__file__).parent


class GenerationCfg(BaseModel):
    initial_count: int = 8
    evolve_count: int = 4
    temperature: float = 1.0
    max_mutations: int = 6


class ReflectionCfg(BaseModel):
    temperature: float = 0.4


class RankingCfg(BaseModel):
    matches_per_round: int = 12
    temperature: float = 0.3
    score_weight: float = 0.5


class ProximityCfg(BaseModel):
    dedup_threshold: float = 0.92


class EvolutionCfg(BaseModel):
    strategies: list[str] = Field(default_factory=lambda: ["refine", "combine", "simplify", "out_of_box"])
    temperature: float = 1.0


class Config(BaseModel):
    name: str = "default"
    rounds: int = 3
    goal: str | None = None
    seed_sequence: str | None = None
    protein_mode: bool = False
    scorer: str = "heuristic"
    generation: GenerationCfg = Field(default_factory=GenerationCfg)
    reflection: ReflectionCfg = Field(default_factory=ReflectionCfg)
    ranking: RankingCfg = Field(default_factory=RankingCfg)
    proximity: ProximityCfg = Field(default_factory=ProximityCfg)
    evolution: EvolutionCfg = Field(default_factory=EvolutionCfg)


def load_config(name_or_path: str = "default") -> Config:
    """Load a config by preset name (in config/) or explicit YAML path."""
    p = Path(name_or_path)
    if not p.exists():
        p = CONFIG_DIR / f"{name_or_path}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {name_or_path}")
    data: dict[str, Any] = yaml.safe_load(p.read_text()) or {}
    cfg = Config.model_validate(data)
    # Env override for scorer choice.
    env_scorer = os.getenv("COSCIENTIST_SCORER")
    if env_scorer:
        cfg.scorer = env_scorer
    return cfg
