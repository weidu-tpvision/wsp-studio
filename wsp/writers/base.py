from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ParsedTurn:
    reason: str
    action_type: str  # "forward" | "python" | "finish"
    payload: str


_BLOCK_RE = re.compile(r"```(forward|python|finish)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_writer_response(text: str) -> ParsedTurn:
    """Extract REASON and the single (forward | python | finish) code block."""
    match = _BLOCK_RE.search(text)
    if not match:
        raise ValueError(
            "Response must contain exactly one fenced block: ```forward```, ```python```, or ```finish```."
        )
    action_type = match.group(1).lower()
    payload = match.group(2).strip()
    reason = text[: match.start()].strip()
    if reason.upper().startswith("REASON:"):
        reason = reason[7:].strip()
    if not reason:
        reason = "(no reason provided)"
    return ParsedTurn(reason=reason, action_type=action_type, payload=payload)


class WriterLLM(ABC):
    """Interface for the frontier reasoning model that inspects and edits student weights."""

    provider_name: str = "base"

    @abstractmethod
    def generate(self, system_prompt: str, messages: list[dict[str, Any]]) -> str:
        """Return the assistant's text reply given system prompt and conversation history."""
