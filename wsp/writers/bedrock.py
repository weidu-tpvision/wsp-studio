from __future__ import annotations

import os
from typing import Any, Optional

from wsp.writers.base import WriterLLM


class BedrockWriter(WriterLLM):
    """Writer LLM backed by AWS Bedrock Runtime Converse API."""

    provider_name = "bedrock"

    def __init__(
        self,
        model_id: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        region_name: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> None:
        self.model_id = model_id
        self.region_name = region_name or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        if self._client is None:
            import boto3

            self._client = boto3.client("bedrock-runtime", region_name=self.region_name)
        return self._client

    def generate(self, system_prompt: str, messages: list[dict[str, Any]]) -> str:
        client = self._get_client()
        converse_msgs = []
        for msg in messages:
            role = "assistant" if msg["role"] == "assistant" else "user"
            converse_msgs.append({"role": role, "content": [{"text": msg["content"]}]})

        resp = client.converse(
            modelId=self.model_id,
            system=[{"text": system_prompt}],
            messages=converse_msgs,
            inferenceConfig={
                "temperature": self.temperature,
                "maxTokens": self.max_tokens,
            },
        )
        blocks = resp["output"]["message"]["content"]
        return "\n".join(b["text"] for b in blocks if "text" in b).strip()
