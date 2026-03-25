"""Configuration loader for Continuum."""

import os
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
    "retrieval": {
        "semantic_k": 30,
        "keyword_k": 30,
        "keyword_weight": 0.3,
        "hybrid_boost": 0.4,
        "identifier_weight": 0.6,
        "decay_half_life_days": 30,
        "decay_floor": 0.3,
        "correction_boost_max": 0.5,
        "context_boost": 1.5,
        "kernel_boost": 2.0,
        "chain_boost": 1.3,
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
            try:
                import yaml
            except ImportError:
                return config
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


def get_model(role: str, config: dict | None = None) -> str:
    """Resolve model name for a given role.

    Priority: env var > config > hardcoded default.

    Roles:
        compress   — LLM compression (spoof --compress)
        cull       — retrieval precision filtering
        decompose  — query decomposition (Ollama)
        dream      — offline integration passes (Ollama)
    """
    env_map = {
        "compress": "CONTINUUM_COMPRESS_MODEL",
        "cull": "CONTINUUM_CULL_MODEL",
        "decompose": "CONTINUUM_DECOMPOSE_MODEL",
        "dream": "CONTINUUM_DREAM_MODEL",
    }
    config_map = {
        "compress": "compress_model",
        "cull": "cull_model",
        "decompose": "decompose_model",
        "dream": "dream_model",
    }
    defaults = {
        "compress": "claude-sonnet-4-6",
        "cull": "claude-haiku-4-5-20251001",
        "decompose": "qwen2.5:7b",
        "dream": "qwen2.5:7b",
    }

    # 1. Env var
    env_key = env_map.get(role, "")
    if env_key:
        val = os.environ.get(env_key, "").strip()
        if val:
            return val

    # 2. Config file
    if config:
        config_key = config_map.get(role, "")
        if config_key and config_key in config:
            return config[config_key]

    # 3. Default
    return defaults.get(role, "claude-sonnet-4-6")
