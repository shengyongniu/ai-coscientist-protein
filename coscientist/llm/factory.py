"""Provider factory: choose Bedrock or Mock from env/config."""

from __future__ import annotations

import os

from coscientist.llm.base import LLMProvider


def get_provider(name: str | None = None) -> LLMProvider:
    """Return an LLM provider.

    Resolution order: explicit `name` arg -> COSCIENTIST_LLM_PROVIDER env ->
    "bedrock". Falls back to the mock provider if Bedrock can't be constructed
    (e.g. no AWS creds), so the system stays runnable.
    """
    name = (name or os.getenv("COSCIENTIST_LLM_PROVIDER", "bedrock")).lower()
    if name == "mock":
        from coscientist.llm.mock import MockProvider

        return MockProvider()
    if name == "bedrock":
        try:
            from coscientist.llm.bedrock import BedrockProvider

            return BedrockProvider()
        except Exception as e:  # pragma: no cover - depends on environment
            import warnings

            warnings.warn(
                f"Could not initialize Bedrock provider ({e}); falling back to mock.",
                stacklevel=2,
            )
            from coscientist.llm.mock import MockProvider

            return MockProvider()
    raise ValueError(f"Unknown LLM provider: {name}")
