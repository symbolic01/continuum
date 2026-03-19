"""Token counting utility for Continuum."""

import subprocess
import json


def count_tokens(text: str) -> int:
    """Estimate token count. Uses a simple heuristic: ~4 chars per token.

    Good enough for budget allocation. Can swap in tiktoken or
    anthropic's counter later without changing the interface.
    """
    return len(text) // 4


def count_messages_tokens(messages: list[dict]) -> int:
    """Count total tokens across a messages array."""
    total = 0
    for msg in messages:
        # Role overhead ~4 tokens
        total += 4
        content = msg.get("content", "")
        if isinstance(content, str):
            total += count_tokens(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    total += count_tokens(block["text"])
    return total
