"""Continuum — dynamic context assembly for Claude API sessions.

Manages a per-turn context window assembled from:
  Layer 0: Identity core (identity.md — never compressed)
  Layer 1: System prompt
  Layer 2: Dynamic context (retrieved from configured sources)
  Layer 3a: Compressed older session history
  Layer 3b: Uncompressed recent session tail
  Layer 4: Current user turn
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

from core.tokens import count_tokens, count_messages_tokens
from core.session_log import SessionLog
from core.retrieval import ContextRetriever
from core.compression import CompressionPolicy, CompressedBlock, TokenBudgetPolicy
from core.actions import parse_actions, strip_actions, execute_action
from core.backend import Backend, CLIBackend, make_backend


class ContextAssembler:
    """Assembles the messages array for each API call."""

    def __init__(
        self,
        identity_path: str | Path,
        system_prompt: str,
        retriever: ContextRetriever,
        session_log: SessionLog,
        model: str = "claude-sonnet-4-6",
        token_budgets: dict | None = None,
        compression_policy: CompressionPolicy | None = None,
        backend: Backend | None = None,
    ):
        self.identity = Path(identity_path).expanduser().read_text().strip()
        self.system_prompt = system_prompt
        self.retriever = retriever
        self.log = session_log
        self.model = model
        self.policy = compression_policy or TokenBudgetPolicy()
        self.backend = backend or CLIBackend()

        defaults = {
            "total": 180_000,
            "identity": 300,
            "system": 4_000,
            "dynamic_context": 30_000,
            "compressed_history": 20_000,
            "recent_tail": 80_000,
        }
        self.budgets = {**defaults, **(token_budgets or {})}

        # Layer 3a: compressed blocks accumulate here across turns
        self._compressed: list[CompressedBlock] = []

        # Load any previously saved compressed blocks
        self._compressed_path = self.log.path.with_suffix(".compressed.json")
        if self._compressed_path.exists():
            try:
                data = json.loads(self._compressed_path.read_text())
                self._compressed = [CompressedBlock.from_dict(d) for d in data]
            except (json.JSONDecodeError, KeyError):
                pass

    def _save_compressed(self):
        """Persist compressed blocks to disk alongside the session log."""
        with open(self._compressed_path, "w") as f:
            json.dump([b.to_dict() for b in self._compressed], f, indent=2)

    def _build_system(self) -> str:
        """Combine identity core + system prompt."""
        return f"{self.identity}\n\n---\n\n{self.system_prompt}"

    def _maybe_compress(self):
        """Check if session history needs compression, and compress if so.

        Takes entries that won't fit in the tail budget, compresses them
        into a block, and adds to the 3a layer. Only compresses entries
        that haven't been compressed yet (turns newer than any existing
        compressed block).
        """
        entries = self.log.entries
        if len(entries) < 6:  # not worth compressing tiny sessions
            return

        # What turns are already compressed?
        already_compressed = set()
        for block in self._compressed:
            already_compressed.update(block.ref_turns)

        # Filter to uncompressed entries only
        uncompressed = [e for e in entries if e.get("turn", 0) not in already_compressed]
        if len(uncompressed) < 6:
            return

        # Ask the policy to split into compress/keep
        to_compress, tail = self.policy.select_for_compression(
            uncompressed, self.budgets["recent_tail"]
        )

        if not to_compress:
            return

        # Compress in batches of ~10 turns
        batch_size = 10
        i = 0
        while i < len(to_compress):
            batch = to_compress[i:i + batch_size]
            block = self.policy.compress(batch)
            self._compressed.append(block)
            i += batch_size

        self._save_compressed()

    def _build_compressed_messages(self) -> list[dict]:
        """Build Layer 3a messages from compressed blocks."""
        if not self._compressed:
            return []

        budget = self.budgets["compressed_history"]
        messages = []
        tokens_used = 0

        # Most recent compressed blocks first (reverse chronological)
        for block in reversed(self._compressed):
            msg = block.to_message()
            msg_tokens = count_tokens(msg["content"]) + 4
            if tokens_used + msg_tokens > budget:
                break
            messages.insert(0, msg)
            tokens_used += msg_tokens

        return messages

    def _build_tail(self, thread: str | None = None) -> list[dict]:
        """Build Layer 3b: recent session tail from the log.

        Only includes turns that haven't been compressed.
        """
        entries = self.log.entries
        if not entries:
            return []

        # Exclude already-compressed turns
        already_compressed = set()
        for block in self._compressed:
            already_compressed.update(block.ref_turns)

        uncompressed = [e for e in entries if e.get("turn", 0) not in already_compressed]
        if not uncompressed:
            return []

        budget = self.budgets["recent_tail"]
        tail = []
        tokens_used = 0

        # Walk backwards from most recent
        for entry in reversed(uncompressed):
            msg = {"role": entry["role"], "content": entry["content"]}
            msg_tokens = count_tokens(entry["content"]) + 4
            if tokens_used + msg_tokens > budget:
                break
            tail.insert(0, msg)
            tokens_used += msg_tokens

        return tail

    def assemble(self, user_message: str, thread: str = "default") -> tuple[str, list[dict]]:
        """Assemble the full system + messages for an API call.

        Returns:
            (system_prompt, messages) ready for the API
        """
        # Maybe compress older history before assembling
        self._maybe_compress()

        system = self._build_system()

        # Build conversation tail for context-enriched retrieval
        recent = self.log.get_recent(10)
        conversation_tail = "\n".join(
            f"[{e.get('role', '?')}] {e.get('content', '')[:300]}"
            for e in recent
        )

        # Layer 2: dynamic context (query enriched with conversation tail)
        dynamic = self.retriever.retrieve(
            query=user_message,
            token_budget=self.budgets["dynamic_context"],
            conversation_tail=conversation_tail,
        )

        messages = []

        # Inject dynamic context as a synthetic turn
        if dynamic:
            messages.append({
                "role": "user",
                "content": f"<context>\n{dynamic}\n</context>",
            })
            messages.append({
                "role": "assistant",
                "content": "Thank you, I have that context.",
            })

        # Layer 3a: compressed older history
        compressed_msgs = self._build_compressed_messages()
        if compressed_msgs:
            messages.append({
                "role": "user",
                "content": "<compressed_history>The following are summaries of earlier parts of our conversation:</compressed_history>",
            })
            messages.extend(compressed_msgs)

        # Layer 3b: recent session tail
        tail = self._build_tail(thread=thread)
        messages.extend(tail)

        # Layer 4: current user turn
        messages.append({"role": "user", "content": user_message})

        # Stash stats from this assembly for observability
        self._last_stats = {
            "identity_tokens": count_tokens(self.identity),
            "system_tokens": count_tokens(system),
            "dynamic_context_tokens": count_tokens(dynamic) if dynamic else 0,
            "dynamic_context_files": len(self.retriever._resolve_paths()) if dynamic else 0,
            "compressed_blocks": len(self._compressed),
            "compressed_tokens": sum(count_tokens(m["content"]) for m in compressed_msgs) if compressed_msgs else 0,
            "tail_entries": len(tail),
            "tail_tokens": sum(count_tokens(e["content"]) for e in tail),
            "total_tokens": count_messages_tokens(messages) + count_tokens(system),
            "total_log_entries": len(self.log),
        }

        return system, messages

    def stats(self) -> dict:
        """Return stats from the most recent assembly."""
        return getattr(self, "_last_stats", {})

    def call_api(self, system: str, messages: list[dict]) -> str:
        """Call the API via the configured backend."""
        return self.backend.call(system, messages, self.model)


class Session:
    """A Continuum session — the main turn loop."""

    def __init__(self, assembler: ContextAssembler):
        self.assembler = assembler
        self.current_thread = "default"

    def turn(self, user_message: str) -> tuple[str, dict]:
        """Process one turn: assemble context, call API, log everything.

        If the response contains <action> blocks, executes them and
        includes results in the returned stats.

        Returns:
            (response_text, stats_dict)
        """
        # Log user message
        self.assembler.log.append("user", user_message, thread=self.current_thread)

        # Assemble and call
        system, messages = self.assembler.assemble(
            user_message, thread=self.current_thread
        )
        response = self.assembler.call_api(system, messages)

        # Parse and execute any action blocks
        actions = parse_actions(response)
        action_results = []
        if actions:
            for action in actions:
                result = execute_action(action, model=self.assembler.model)
                action_results.append(result)

        # Log assistant response (with actions stripped for cleaner history)
        display_response = strip_actions(response) if actions else response
        self.assembler.log.append("assistant", display_response, thread=self.current_thread)

        stats = self.assembler.stats()
        if action_results:
            stats["actions"] = action_results

        return display_response, stats
