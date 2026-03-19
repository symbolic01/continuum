"""Configuration loader for Continuum."""

import yaml
from pathlib import Path


DEFAULT_CONFIG = {
    "model": "claude-sonnet-4-6",
    "identity": "identity.md",
    "system_prompt": "You are a helpful assistant.",
    "context_sources": [],
    "token_budgets": {
        "total": 180_000,
        "identity": 300,
        "system": 4_000,
        "dynamic_context": 30_000,
        "compressed_history": 20_000,
        "recent_tail": 80_000,
    },
    "compression": {
        "policy": "token_budget",
        "model": "claude-haiku-4-5-20251001",
    },
    "session": {
        "log_dir": "~/.continuum/sessions",
    },
}


def load_config(path: str | Path | None = None) -> dict:
    """Load config from YAML file, falling back to defaults."""
    config = dict(DEFAULT_CONFIG)

    if path is None:
        # Look for continuum.yaml in current dir, then home
        candidates = [
            Path.cwd() / "continuum.yaml",
            Path.home() / ".continuum" / "continuum.yaml",
        ]
        for c in candidates:
            if c.exists():
                path = c
                break

    if path is not None:
        path = Path(path)
        if path.exists():
            with open(path) as f:
                user_config = yaml.safe_load(f) or {}
            # Merge (shallow for top-level, deep for token_budgets)
            for key in user_config:
                if key == "token_budgets" and isinstance(user_config[key], dict):
                    config["token_budgets"] = {
                        **config["token_budgets"],
                        **user_config[key],
                    }
                else:
                    config[key] = user_config[key]

    return config
