"""Append-only session log — ground truth that never gets modified."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def mint_uid() -> str:
    """Mint a short provenance UID in guillemet delimiters.

    6 hex chars = 16M namespace, sufficient per-session.
    Format: «a7f3b2»
    """
    return "«" + uuid4().hex[:6] + "»"


class SessionLog:
    """Append-only JSONL session log.

    Each turn is written with metadata: role, content, turn number,
    timestamp, context thread, and optional embedding vector. This is
    the full-fidelity record that compressed summaries reference back to.
    """

    def __init__(self, path: str | Path, embed: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._turn = 0
        self._entries: list[dict] = []
        self._embed = embed  # mint embeddings at write time

        # Load existing entries if resuming
        if self.path.exists():
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        self._entries.append(entry)
                        self._turn = max(self._turn, entry.get("turn", 0))

    @property
    def turn(self) -> int:
        return self._turn

    @property
    def entries(self) -> list[dict]:
        return list(self._entries)

    def append(self, role: str, content: str, thread: str = "default") -> dict:
        """Append a turn to the log. Returns the entry.

        If embed=True was set on init, mints a semantic embedding alongside
        the UID. The embedding is stored in the entry as a float vector,
        enabling proximity traversal across the entire corpus.
        """
        if role == "user":
            self._turn += 1

        entry = {
            "uid": mint_uid(),
            "role": role,
            "content": content,
            "turn": self._turn,
            "ts": datetime.now(timezone.utc).isoformat(),
            "thread": thread,
        }

        # Mint embedding at write time (async-safe: failures produce None)
        if self._embed:
            try:
                from embeddings import embed_text
                vec = embed_text(content)
                if vec is not None:
                    entry["embedding"] = vec
            except Exception:
                pass  # embedding is best-effort, never blocks writes

        self._entries.append(entry)

        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        return entry

    def get_thread_entries(self, thread: str) -> list[dict]:
        """Get all entries for a specific context thread."""
        return [e for e in self._entries if e.get("thread") == thread]

    def get_recent(self, n: int | None = None) -> list[dict]:
        """Get the most recent entries, optionally limited to n."""
        if n is None:
            return list(self._entries)
        return self._entries[-n:]

    def get_turns_range(self, start: int, end: int) -> list[dict]:
        """Get entries for a range of turn numbers (inclusive)."""
        return [
            e for e in self._entries
            if start <= e.get("turn", 0) <= end
        ]

    def get_by_uid(self, uid: str) -> dict | None:
        """Look up a single entry by its provenance UID."""
        for e in self._entries:
            if e.get("uid") == uid:
                return e
        return None

    def get_by_uids(self, uids: list[str]) -> list[dict]:
        """Look up multiple entries by UIDs, preserving order."""
        uid_set = set(uids)
        return [e for e in self._entries if e.get("uid") in uid_set]

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"SessionLog({self.path}, {len(self)} entries, turn={self._turn})"
