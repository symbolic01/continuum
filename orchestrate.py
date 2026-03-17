#!/usr/bin/env python3
"""Continuum Orchestrator — manages Claude Code sessions with dynamic context.

Prepares a spoofed session JSONL with dynamically assembled context,
then launches `claude --resume` for the user. After the session ends,
captures new entries into Continuum's ground truth log.

Usage:
    python orchestrate.py                    # new session
    python orchestrate.py --continue         # continue last session
    python orchestrate.py --session my-name  # named session
"""

import argparse
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from config import load_config
from session_log import SessionLog
from retrieval import ContextRetriever
from compression import TokenBudgetPolicy
from index import load_index
from session_spoof import (
    build_spoofed_session,
    write_cc_session,
    read_cc_new_entries,
    extract_text_from_cc_entry,
)


def main():
    parser = argparse.ArgumentParser(description="Continuum — Claude Code with dynamic context")
    parser.add_argument("--config", "-c", help="Path to continuum.yaml")
    parser.add_argument("--session", "-s", help="Session name")
    parser.add_argument("--model", "-m", help="Override model")
    parser.add_argument("prompt", nargs="?", help="Initial prompt (optional)")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    if args.model:
        config["model"] = args.model

    continuum_dir = Path(__file__).parent

    # Identity
    identity_path = Path(config.get("identity", "identity.md"))
    if not identity_path.is_absolute():
        identity_path = continuum_dir / identity_path
    identity_text = identity_path.read_text().strip() if identity_path.exists() else ""

    # System prompt
    sp = config.get("system_prompt", "")
    sp_path = continuum_dir / sp
    system_prompt = sp_path.read_text().strip() if sp_path.is_file() else sp

    # Session log (ground truth)
    session_name = args.session or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = Path(config["session"]["log_dir"]).expanduser()
    log = SessionLog(log_dir / f"{session_name}.jsonl")

    # Retriever with corpus index
    idx = load_index()
    sources = config.get("context_sources", [])
    retriever = ContextRetriever(sources, index=idx)

    # Compression
    comp_config = config.get("compression", {})
    policy = TokenBudgetPolicy(model=comp_config.get("model", "claude-haiku-4-5-20251001"))

    # Claude Code session ID (stable across turns for this continuum session)
    cc_session_id = str(uuid.uuid4())
    cwd = str(Path.cwd())
    model = config.get("model", "claude-sonnet-4-6")

    print(f"continuum | session={session_name} | model={model}")
    print(f"  corpus: {len(idx)} entries")
    print(f"  log: {log.path} ({len(log)} entries)")
    print(f"  cc session: {cc_session_id[:12]}...")
    print()

    # Initial context retrieval
    query = args.prompt or "starting a new session"
    retrieved = retriever.retrieve(query, token_budget=config.get("token_budgets", {}).get("dynamic_context", 30000))

    # Build the spoofed session
    cc_entries = build_spoofed_session(
        session_id=cc_session_id,
        continuum_log=log,
        retrieved_context=retrieved,
        compressed_blocks=[],  # TODO: wire compression
        tail_budget=config.get("token_budgets", {}).get("recent_tail", 80000),
        identity_text=identity_text + "\n\n" + system_prompt if system_prompt else identity_text,
        cwd=cwd,
    )

    # Write the session file
    session_file = write_cc_session(cc_session_id, cc_entries, cwd=cwd)
    entry_count_before = len(cc_entries)

    print(f"  spoofed {len(cc_entries)} entries → {session_file}")
    print(f"  launching claude --resume {cc_session_id}")
    print()

    # Launch Claude Code
    cmd = ["claude", "--resume", cc_session_id]
    if args.prompt:
        cmd.append(args.prompt)

    env = {k: v for k, v in os.environ.items()
           if k not in ("CLAUDECODE",)}

    try:
        subprocess.run(cmd, env=env, cwd=cwd)
    except KeyboardInterrupt:
        pass

    # Capture new entries from Claude Code
    new_entries = read_cc_new_entries(session_file, entry_count_before)
    if new_entries:
        for entry in new_entries:
            entry_type = entry.get("type", "")
            if entry_type in ("user", "assistant"):
                role = entry.get("message", {}).get("role", entry_type)
                content = extract_text_from_cc_entry(entry)
                if content.strip():
                    log.append(role, content, thread="default")

        print(f"\n  captured {len(new_entries)} new entries → {log.path}")
    else:
        print("\n  no new entries to capture")

    print(f"  ground truth: {len(log)} total entries")


if __name__ == "__main__":
    main()
