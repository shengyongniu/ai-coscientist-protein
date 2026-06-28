"""AWS Bedrock LLM provider using the Converse API.

Uses boto3's `bedrock-runtime` client. Supports both buffered (`converse`) and
streaming (`converse_stream`) calls, with simple exponential-backoff retry on
throttling, and token/cost accounting via the shared UsageTracker.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator

from coscientist.llm.base import LLMMessage, LLMProvider, LLMResponse

_THROTTLE_ERRORS = {
    "ThrottlingException",
    "TooManyRequestsException",
    "ServiceUnavailableException",
    "ModelTimeoutException",
}


class BedrockProvider(LLMProvider):
    def __init__(
        self,
        region: str | None = None,
        model_strong: str | None = None,
        model_fast: str | None = None,
        max_retries: int = 5,
    ) -> None:
        super().__init__()
        import boto3  # imported lazily so the package works without AWS

        self.region = region or os.getenv("AWS_REGION", "us-west-2")
        self.model_strong = model_strong or os.getenv(
            "COSCIENTIST_MODEL_STRONG", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        )
        self.model_fast = model_fast or os.getenv(
            "COSCIENTIST_MODEL_FAST", "us.anthropic.claude-haiku-4-5-20251001-v1:0"
        )
        self.max_retries = max_retries
        self._client = boto3.client("bedrock-runtime", region_name=self.region)

    def _resolve(self, model: str | None) -> str:
        if model in (None, "strong"):
            return self.model_strong
        if model == "fast":
            return self.model_fast
        return model

    def _to_bedrock_messages(self, messages: list[LLMMessage]) -> list[dict]:
        return [{"role": m.role, "content": [{"text": m.content}]} for m in messages]

    def complete(
        self,
        system: str,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        from botocore.exceptions import ClientError

        model_id = self._resolve(model)
        kwargs = dict(
            modelId=model_id,
            messages=self._to_bedrock_messages(messages),
            inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
        )
        if system:
            kwargs["system"] = [{"text": system}]

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.converse(**kwargs)
                out = resp["output"]["message"]["content"][0]["text"]
                usage = resp.get("usage", {})
                r = LLMResponse(
                    text=out,
                    input_tokens=usage.get("inputTokens", 0),
                    output_tokens=usage.get("outputTokens", 0),
                    model=model_id,
                    stop_reason=resp.get("stopReason", ""),
                )
                self.usage.record(r)
                return r
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                last_err = e
                if code in _THROTTLE_ERRORS and attempt < self.max_retries - 1:
                    time.sleep(min(2**attempt, 16) + 0.1 * attempt)
                    continue
                raise
        raise RuntimeError(f"Bedrock call failed after retries: {last_err}")

    def stream(
        self,
        system: str,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Iterator[str]:
        """Yield text deltas as they arrive via converse_stream."""
        model_id = self._resolve(model)
        kwargs = dict(
            modelId=model_id,
            messages=self._to_bedrock_messages(messages),
            inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
        )
        if system:
            kwargs["system"] = [{"text": system}]
        resp = self._client.converse_stream(**kwargs)
        in_tok = out_tok = 0
        for event in resp["stream"]:
            if "contentBlockDelta" in event:
                yield event["contentBlockDelta"]["delta"].get("text", "")
            elif "metadata" in event:
                usage = event["metadata"].get("usage", {})
                in_tok = usage.get("inputTokens", 0)
                out_tok = usage.get("outputTokens", 0)
        self.usage.record(
            LLMResponse(text="", input_tokens=in_tok, output_tokens=out_tok, model=model_id)
        )
