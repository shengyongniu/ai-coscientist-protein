"""Deterministic mock LLM provider for offline runs and tests.

It inspects a lightweight directive embedded in the system prompt
(``[[MOCK:<kind>]]``) and returns valid JSON matching what each agent expects.
This lets the full multi-agent pipeline run end-to-end with no AWS access, which
keeps the repo runnable for anyone and makes CI deterministic.
"""

from __future__ import annotations

import hashlib
import json
import random
import re

from coscientist.llm.base import LLMMessage, LLMProvider, LLMResponse

_AA = "ACDEFGHIKLMNPQRSTVWY"
_DIRECTIVE = re.compile(r"\[\[MOCK:([a-z_]+)\]\]")


def _seed_from(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)


def _mutate(seq: str, rng: random.Random, n: int) -> str:
    s = list(seq)
    if not s:
        return "".join(rng.choice(_AA) for _ in range(40))
    for _ in range(n):
        i = rng.randrange(len(s))
        s[i] = rng.choice(_AA)
    return "".join(s)


class MockProvider(LLMProvider):
    """Returns plausible, schema-valid responses without any network calls."""

    def complete(
        self,
        system: str,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        prompt = "\n".join(m.content for m in messages)
        rng = random.Random(_seed_from(system + prompt))
        m = _DIRECTIVE.search(system) or _DIRECTIVE.search(prompt)
        kind = m.group(1) if m else "generation"
        text = self._dispatch(kind, system, prompt, rng)
        resp = LLMResponse(
            text=text,
            input_tokens=len(prompt) // 4,
            output_tokens=len(text) // 4,
            model=f"mock-{model or 'strong'}",
            stop_reason="end_turn",
        )
        self.usage.record(resp)
        return resp

    def _dispatch(self, kind: str, system: str, prompt: str, rng: random.Random) -> str:
        seed = self._extract_seed_sequence(prompt)
        if kind == "generation":
            return self._gen(prompt, rng, seed, count=self._extract_count(prompt, 6))
        if kind == "reflection":
            return self._reflect(rng)
        if kind == "ranking":
            return self._rank(prompt, rng)
        if kind == "evolution":
            return self._evolve(prompt, rng, seed, count=self._extract_count(prompt, 4))
        if kind == "meta_review":
            return self._meta(rng)
        if kind == "plan":
            return self._plan(rng)
        return self._gen(prompt, rng, seed, count=4)

    @staticmethod
    def _extract_count(prompt: str, default: int) -> int:
        m = re.search(r"generate\s+(?:exactly\s+)?(\d+)", prompt, re.IGNORECASE)
        return int(m.group(1)) if m else default

    @staticmethod
    def _extract_seed_sequence(prompt: str) -> str:
        m = re.search(r"SEED_SEQUENCE:\s*([A-Z]{10,})", prompt)
        return m.group(1) if m else ""

    def _gen(self, prompt: str, rng: random.Random, seed: str, count: int) -> str:
        topics = [
            "stability-enhancing core packing",
            "electrostatic complementarity at the interface",
            "CDR3 loop rigidification",
            "aromatic stacking at the paratope",
            "reduced aggregation-prone motifs",
            "improved framework thermostability",
            "salt-bridge engineering",
            "hydrophobic patch removal",
            "glycosylation-site avoidance",
            "entropy-optimized loop shortening",
        ]
        items = []
        for i in range(count):
            t = topics[(rng.randrange(len(topics)) + i) % len(topics)]
            seq = _mutate(seed, rng, rng.randint(1, 5)) if seed else None
            items.append(
                {
                    "title": f"Variant via {t}",
                    "summary": f"Introduce mutations to exploit {t}, hypothesized to improve binding/stability.",
                    "rationale": f"Targeting {t} addresses a known liability and is consistent with structural priors.",
                    "sequence": seq,
                    "experiments": [
                        "Express variant and measure thermostability (Tm) by DSF",
                        "Measure binding affinity (KD) by SPR against the target",
                    ],
                }
            )
        return "```json\n" + json.dumps({"hypotheses": items}) + "\n```"

    def _reflect(self, rng: random.Random) -> str:
        return "```json\n" + json.dumps(
            {
                "correctness": round(rng.uniform(5, 9), 1),
                "novelty": round(rng.uniform(4, 9), 1),
                "testability": round(rng.uniform(6, 9), 1),
                "safety": round(rng.uniform(7, 10), 1),
                "critique": "Plausible mechanism; main risk is the assumption holds under the target's conditions.",
                "suggestions": [
                    "Add a negative control sequence",
                    "Validate fold integrity computationally before synthesis",
                ],
            }
        ) + "\n```"

    def _rank(self, prompt: str, rng: random.Random) -> str:
        winner = "A" if rng.random() < 0.5 else "B"
        return "```json\n" + json.dumps(
            {
                "winner": winner,
                "reasoning": f"Hypothesis {winner} presents a more mechanistically grounded and testable rationale.",
            }
        ) + "\n```"

    def _evolve(self, prompt: str, rng: random.Random, seed: str, count: int) -> str:
        items = []
        for _ in range(count):
            base = self._extract_seed_sequence(prompt) or seed
            seq = _mutate(base, rng, rng.randint(1, 4)) if base else None
            items.append(
                {
                    "title": "Evolved combination variant",
                    "summary": "Combine the strongest features of top-ranked hypotheses.",
                    "rationale": "Merging complementary mutations is expected to compound their individual benefits.",
                    "sequence": seq,
                    "experiments": ["Re-score and re-rank against parents"],
                }
            )
        return "```json\n" + json.dumps({"hypotheses": items}) + "\n```"

    def _meta(self, rng: random.Random) -> str:
        return "```json\n" + json.dumps(
            {
                "summary": "The tournament converged on stability- and interface-focused variants; "
                "top candidates share core-packing and electrostatic improvements.",
                "recommended_experiments": [
                    "Synthesize the top 5 candidates and measure Tm and KD",
                    "Run structural prediction to confirm fold integrity",
                    "Counter-screen for off-target binding",
                ],
                "open_questions": [
                    "Do predicted gains transfer to wet-lab affinity?",
                    "Is developability preserved across the top candidates?",
                ],
            }
        ) + "\n```"

    def _plan(self, rng: random.Random) -> str:
        return "```json\n" + json.dumps(
            {
                "interpretation": "Design and rank improved protein binder variants for the stated target.",
                "subgoals": [
                    "Generate diverse candidate variants",
                    "Critically review each for plausibility and safety",
                    "Rank via objective scoring and simulated debate",
                    "Evolve the best candidates over multiple rounds",
                ],
            }
        ) + "\n```"
