"""Backend abstraction — swappable API call strategies.

Backends:
  - cli: uses `claude --print` (home, Max Pro subscription, no caching control)
  - api: uses Anthropic SDK directly (work, per-token pricing, prompt caching)

The API backend structures the messages array to maximize cache hits:
stable prefix (identity + system + dynamic context) → cache_control breakpoint
→ session history → current turn. The prefix is identical across turns,
so Bedrock/Anthropic serves it from cache at ~90% token discount.
"""

import json
import os
import shutil
import subprocess
from abc import ABC, abstractmethod

from tokens import count_tokens


class Backend(ABC):
    """Base class for API backends."""

    @abstractmethod
    def call(self, system: str, messages: list[dict], model: str) -> str:
        """Send system + messages to the model, return response text."""


class CLIBackend(Backend):
    """Uses `claude --print` subprocess. No caching control.

    Best for: home use with Claude Max subscription (flat rate).
    """

    def call(self, system: str, messages: list[dict], model: str) -> str:
        claude_bin = shutil.which("claude") or "claude"

        conversation_context = ""
        prior_messages = messages[:-1]
        current_message = messages[-1]["content"]

        if prior_messages:
            parts = []
            for msg in prior_messages:
                role = msg["role"].upper()
                parts.append(f"[{role}]: {msg['content']}")
            conversation_context = "\n\n".join(parts)

        full_system = system
        if conversation_context:
            full_system = f"{system}\n\n<conversation_history>\n{conversation_context}\n</conversation_history>"

        cmd = [
            claude_bin, "--print", "--model", model,
            "--no-session-persistence",
            "--dangerously-skip-permissions",
            "--append-system-prompt", full_system,
            "-p", current_message,
        ]

        env = {k: v for k, v in os.environ.items()
               if k not in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")}

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            env=env, stdin=subprocess.DEVNULL,
        )

        if result.returncode != 0:
            error = (result.stderr or "").strip()[:500]
            raise RuntimeError(f"claude --print failed (rc={result.returncode}): {error}")

        return (result.stdout or "").strip()


class APIBackend(Backend):
    """Uses Anthropic SDK directly with prompt caching.

    Structures the request to maximize cache hits:
    - System prompt with cache_control breakpoint
    - Dynamic context as first message with cache_control
    - Session history (prefix is cached across turns)
    - Current turn (only new tokens)

    Best for: work use with API/Bedrock (per-token pricing).

    Config:
        api_key: ANTHROPIC_API_KEY env var or config
        base_url: for LiteLLM proxy (e.g. http://localhost:4000)
    """

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.base_url = base_url

    def call(self, system: str, messages: list[dict], model: str) -> str:
        try:
            import anthropic
        except ImportError:
            raise RuntimeError("anthropic SDK not installed. Run: pip install anthropic")

        client_kwargs = {}
        if self.api_key:
            client_kwargs["api_key"] = self.api_key
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        client = anthropic.Anthropic(**client_kwargs)

        # Build system with cache control
        system_blocks = [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ]

        # Mark dynamic context messages for caching
        api_messages = []
        for i, msg in enumerate(messages):
            api_msg = {"role": msg["role"], "content": msg["content"]}
            # Cache the dynamic context block (first user message if it's <context>)
            if i == 0 and msg["role"] == "user" and msg["content"].startswith("<context>"):
                api_msg["content"] = [
                    {
                        "type": "text",
                        "text": msg["content"],
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            # Cache the conversation history prefix (second-to-last message)
            elif i == len(messages) - 2:
                api_msg["content"] = [
                    {
                        "type": "text",
                        "text": msg["content"],
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            api_messages.append(api_msg)

        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_blocks,
            messages=api_messages,
        )

        # Extract text from response
        text_parts = []
        for block in response.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
        result = "\n".join(text_parts)

        # Log cache stats if available
        usage = response.usage
        cache_read = getattr(usage, "cache_read_input_tokens", 0)
        cache_create = getattr(usage, "cache_creation_input_tokens", 0)
        input_tokens = getattr(usage, "input_tokens", 0)
        output_tokens = getattr(usage, "output_tokens", 0)

        if cache_read or cache_create:
            print(f"[api] tokens: {input_tokens} in ({cache_read} cached, {cache_create} new cache) + {output_tokens} out", flush=True)
        else:
            print(f"[api] tokens: {input_tokens} in + {output_tokens} out", flush=True)

        return result


def make_backend(config: dict) -> Backend:
    """Create a backend from config.

    Config options:
        backend: "cli" (default) or "api"
        api_key: for API backend (or ANTHROPIC_API_KEY env var)
        base_url: for LiteLLM/proxy (e.g. "http://localhost:4000")
    """
    backend_type = config.get("backend", "cli")

    if backend_type == "api":
        return APIBackend(
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
        )
    else:
        return CLIBackend()
