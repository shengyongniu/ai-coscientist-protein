"""Abstract LLM provider interface and shared types.

All agents talk to the LLM through `LLMProvider.complete`, so the rest of the
system is provider-agnostic (Bedrock in production, Mock in tests/offline).
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMMessage:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    stop_reason: str = ""


@dataclass
class UsageTracker:
    """Accumulates token usage and an approximate USD cost across calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    cost_usd: float = 0.0
    # Per-model (input, output) USD per 1K tokens. Rough public list prices.
    price_table: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "sonnet": (0.003, 0.015),
            "haiku": (0.0008, 0.004),
            "opus": (0.015, 0.075),
        }
    )

    def record(self, resp: LLMResponse) -> None:
        self.calls += 1
        self.input_tokens += resp.input_tokens
        self.output_tokens += resp.output_tokens
        rate_in, rate_out = (0.003, 0.015)
        for key, rates in self.price_table.items():
            if key in resp.model.lower():
                rate_in, rate_out = rates
                break
        self.cost_usd += (resp.input_tokens / 1000.0) * rate_in
        self.cost_usd += (resp.output_tokens / 1000.0) * rate_out

    def summary(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 4),
        }


class LLMProvider(ABC):
    """Base class for LLM backends."""

    def __init__(self) -> None:
        self.usage = UsageTracker()

    @abstractmethod
    def complete(
        self,
        system: str,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Return a single completion. Implementations should record usage."""

    def complete_text(
        self,
        system: str,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        resp = self.complete(
            system,
            [LLMMessage(role="user", content=prompt)],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.text


_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Best-effort JSON extraction from an LLM response.

    Handles fenced code blocks and trailing prose. Raises ValueError if nothing
    parseable is found.
    """
    candidates: list[str] = []
    for m in _JSON_BLOCK.finditer(text):
        candidates.append(m.group(1).strip())
    candidates.append(text.strip())
    # Also try to grab the first {...} or [...] span.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end != -1 and end > start:
            candidates.append(text[start : end + 1])

    for c in candidates:
        try:
            return json.loads(c)
        except (json.JSONDecodeError, ValueError):
            continue
    raise ValueError(f"No parseable JSON in LLM output: {text[:200]!r}")
