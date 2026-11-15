from __future__ import annotations

from typing import Any, Callable, Optional

from wsp.writers.base import WriterLLM


class MockWriter(WriterLLM):
    """Deterministic scripted writer for offline tests and demos."""

    provider_name = "mock"

    def __init__(self, script: Optional[list[str]] = None, responder: Optional[Callable[[list[dict[str, Any]]], str]] = None) -> None:
        self.script = list(script or [])
        self.responder = responder
        self.call_count = 0

    def generate(self, system_prompt: str, messages: list[dict[str, Any]]) -> str:
        idx = self.call_count
        self.call_count += 1
        if self.responder is not None:
            return self.responder(messages)
        if idx < len(self.script):
            return self.script[idx]
        return 'REASON: Budget complete or no further edits needed.\n```finish\n{"summary": "Mock run complete."}\n```'
