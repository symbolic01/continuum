"""Compression policies for session history.

Compression is the same operation at every timescale:
content → shorter version with refs back to the original.
A turn becomes a summary. A summary becomes a sutra.

Policies decide WHEN and HOW AGGRESSIVELY to compress.
The mechanism is always: take turns, produce a summary with ref_uids.
"""

import json
import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

from .tokens import count_tokens


class CompressedBlock:
    """A compressed summary replacing one or more turns."""

    def __init__(self, summary: str, ref_uids: list[str], ref_turns: list[int],
                 polarity: str = "neutral"):
        self.summary = summary
        self.ref_uids = ref_uids      # UIDs of original entries
        self.ref_turns = ref_turns     # turn numbers covered
        self.polarity = polarity       # "success", "failure", or "neutral"

    def to_message(self) -> dict:
        """Format as a message for the API prompt."""
        refs = ", ".join(self.ref_uids[:6])
        if len(self.ref_uids) > 6:
            refs += f" (+{len(self.ref_uids) - 6} more)"
        prefix = ""
        if self.polarity == "failure":
            prefix = "[FAILED APPROACH] "
        elif self.polarity == "success":
            prefix = "[CONFIRMED] "
        return {
            "role": "assistant",
            "content": f"{prefix}{self.summary}\n[refs: {refs}]",
        }

    def to_dict(self) -> dict:
        return {
            "type": "compressed",
            "summary": self.summary,
            "ref_uids": self.ref_uids,
            "ref_turns": self.ref_turns,
            "polarity": self.polarity,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CompressedBlock":
        return cls(
            summary=d["summary"],
            ref_uids=d.get("ref_uids", []),
            ref_turns=d.get("ref_turns", []),
            polarity=d.get("polarity", "neutral"),
        )


class CompressionPolicy(ABC):
    """Base class for compression policies."""

    @abstractmethod
    def select_for_compression(
        self, entries: list[dict], tail_budget: int
    ) -> tuple[list[dict], list[dict]]:
        """Split entries into (to_compress, to_keep_as_tail).

        Returns:
            to_compress: entries that should be compressed into summaries
            tail: entries that remain at full fidelity
        """

    @abstractmethod
    def compress(self, entries: list[dict]) -> CompressedBlock:
        """Compress a batch of entries into a single summary block."""


class FixedTailPolicy(CompressionPolicy):
    """Keep the last N turns at full fidelity, compress everything else.

    Simplest possible policy. Good baseline for benchmarking.
    """

    def __init__(self, tail_turns: int = 10, model: str = "claude-haiku-4-5-20251001"):
        self.tail_turns = tail_turns
        self.model = model

    def select_for_compression(
        self, entries: list[dict], tail_budget: int
    ) -> tuple[list[dict], list[dict]]:
        if not entries:
            return [], []

        # Find the turn number cutoff
        turns = sorted(set(e.get("turn", 0) for e in entries))
        if len(turns) <= self.tail_turns:
            return [], entries

        cutoff_turn = turns[-self.tail_turns]
        to_compress = [e for e in entries if e.get("turn", 0) < cutoff_turn]
        tail = [e for e in entries if e.get("turn", 0) >= cutoff_turn]
        return to_compress, tail

    def compress(self, entries: list[dict]) -> CompressedBlock:
        return _llm_compress(entries, self.model)


class TokenBudgetPolicy(CompressionPolicy):
    """Keep as many recent turns as fit in the tail budget, compress the rest.

    More adaptive than FixedTail — adjusts to turn size.
    """

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self.model = model

    def select_for_compression(
        self, entries: list[dict], tail_budget: int
    ) -> tuple[list[dict], list[dict]]:
        if not entries:
            return [], []

        # Walk backwards, accumulating tokens until budget is hit
        tail = []
        tokens_used = 0

        for entry in reversed(entries):
            entry_tokens = count_tokens(entry.get("content", "")) + 4
            if tokens_used + entry_tokens > tail_budget:
                break
            tail.insert(0, entry)
            tokens_used += entry_tokens

        # Everything not in tail gets compressed
        tail_set = set(id(e) for e in tail)
        to_compress = [e for e in entries if id(e) not in tail_set]

        return to_compress, tail

    def compress(self, entries: list[dict]) -> CompressedBlock:
        return _llm_compress(entries, self.model)


# ── Shared compression implementation ──────────────────────────────────

COMPRESS_PROMPT = """You are a session history compressor. Given a sequence of conversation turns, produce a concise summary that preserves:

1. KEY DECISIONS made and their reasoning
2. FACTS learned (especially corrections — what was tried and failed vs what worked)
3. OPEN QUESTIONS that weren't resolved
4. ACTION ITEMS mentioned but not completed

For any failed approaches, prefix with [FAILED]: and note the correction.
For confirmed working approaches, prefix with [CONFIRMED]:.

Be concise — aim for 2-4 sentences. Preserve technical specifics (file names, function names, error messages). Drop pleasantries and filler.

Output ONLY the summary text. No JSON, no markdown headers, no commentary."""


def _llm_compress(entries: list[dict], model: str) -> CompressedBlock:
    """Compress entries using an LLM (via claude --print)."""
    # Build the conversation text
    lines = []
    ref_uids = []
    ref_turns = set()
    has_failure = False
    has_success = False

    for e in entries:
        role = e.get("role", "?")
        content = e.get("content", "")
        uid = e.get("uid", "")
        turn = e.get("turn", 0)

        # Truncate very long entries for the compression prompt
        if len(content) > 500:
            content = content[:500] + "..."

        lines.append(f"[{role}] (turn {turn}) {content}")
        if uid:
            ref_uids.append(uid)
        ref_turns.add(turn)

        # Simple polarity detection
        content_lower = content.lower()
        if any(w in content_lower for w in ("error", "failed", "doesn't work", "bug", "wrong")):
            has_failure = True
        if any(w in content_lower for w in ("fixed", "works", "solved", "confirmed", "correct")):
            has_success = True

    conversation = "\n".join(lines)
    user_prompt = f"Compress this conversation segment:\n\n{conversation}"

    claude_bin = shutil.which("claude") or "claude"
    env = {k: v for k, v in os.environ.items()
           if k not in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")}

    try:
        result = subprocess.run(
            [claude_bin, "--print", "--model", model,
             "--no-session-persistence", "--system-prompt", COMPRESS_PROMPT,
             "-p", user_prompt],
            capture_output=True, text=True, timeout=60,
            env=env, stdin=subprocess.DEVNULL,
        )
        summary = (result.stdout or "").strip()
        if not summary or result.returncode != 0:
            # Fallback: mechanical summary
            summary = _mechanical_compress(entries)
    except (subprocess.TimeoutExpired, Exception):
        summary = _mechanical_compress(entries)

    # Determine polarity
    polarity = "neutral"
    if has_failure and not has_success:
        polarity = "failure"
    elif has_success and not has_failure:
        polarity = "success"
    elif has_failure and has_success:
        polarity = "neutral"  # mixed — contains both

    return CompressedBlock(
        summary=summary,
        ref_uids=ref_uids,
        ref_turns=sorted(ref_turns),
        polarity=polarity,
    )


def _mechanical_compress(entries: list[dict]) -> str:
    """Fallback compression without LLM — just truncate and concatenate."""
    parts = []
    for e in entries:
        role = e.get("role", "?")
        content = e.get("content", "")[:100]
        parts.append(f"[{role}] {content}")
    return " | ".join(parts)
