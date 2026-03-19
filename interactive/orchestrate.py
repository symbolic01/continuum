#!/usr/bin/env python3
"""Continuum Orchestrator — Claude Code with dynamic context per user turn.

Each user turn:
  1. Assemble context (retrieve from corpus, compress old history)
  2. Spoof a Claude Code session JSONL with the assembled context
  3. Run `claude --resume <id> --print -p "user message"`
     (full agentic loop — reads files, edits, runs bash, exits when done)
  4. Capture all new entries to Continuum's ground truth log
  5. Display response
  6. Wait for next user input

The session JSONL is rewritten before every user turn with fresh context.
Continuum's ground truth log is append-only and permanent.

Usage:
    python orchestrate.py                    # new session
    python orchestrate.py --session my-name  # named session
    python orchestrate.py --session my-name "initial prompt"
"""

import argparse
import json
import os
import readline
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from core.config import load_config
from core.session_log import SessionLog
from core.retrieval import ContextRetriever
from core.compression import TokenBudgetPolicy
from core.index import load_index
from core.session_spoof import (
    build_spoofed_session,
    write_cc_session,
    read_cc_new_entries,
    extract_text_from_cc_entry,
)


def run_turn(
    user_message: str,
    cc_session_id: str,
    log: SessionLog,
    retriever: ContextRetriever,
    identity_text: str,
    system_prompt: str,
    config: dict,
    cwd: str,
    model: str,
) -> str:
    """Run one user turn through Claude Code with dynamic context.

    Returns the assistant's response text.
    """
    # Log user message to ground truth
    log.append("user", user_message, thread="default")

    # Build conversation tail for context-enriched retrieval
    recent = log.get_recent(10)  # last 10 entries
    conversation_tail = "\n".join(
        f"[{e.get('role', '?')}] {e.get('content', '')[:300]}"
        for e in recent[:-1]  # exclude the just-appended user message
    )

    # Retrieve context based on this turn + recent conversation
    token_budgets = config.get("token_budgets", {})
    retrieved = retriever.retrieve(
        user_message,
        token_budget=token_budgets.get("dynamic_context", 30000),
        conversation_tail=conversation_tail,
    )

    # Build spoofed session JSONL
    full_identity = identity_text
    if system_prompt:
        full_identity = f"{identity_text}\n\n{system_prompt}"

    cc_entries = build_spoofed_session(
        session_id=cc_session_id,
        continuum_log=log,
        retrieved_context=retrieved,
        compressed_blocks=[],  # TODO: wire compression
        tail_budget=token_budgets.get("recent_tail", 80000),
        identity_text=full_identity,
        cwd=cwd,
    )

    # Write the session file (overwrites previous)
    session_file = write_cc_session(cc_session_id, cc_entries, cwd=cwd)
    entry_count_before = len(cc_entries)

    # Run claude --resume --print -p "message"
    # --resume loads the spoofed history
    # --print makes it non-interactive (full agentic loop, then exit)
    # -p sends the user message
    cmd = [
        "claude",
        "--resume", cc_session_id,
        "--print",
        "--dangerously-skip-permissions",
        "--model", model,
        "-p", user_message,
    ]

    env = {k: v for k, v in os.environ.items()
           if k not in ("CLAUDECODE",)}

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
            cwd=cwd,
        )
        response_text = (result.stdout or "").strip()
        if result.returncode != 0 and not response_text:
            response_text = f"[error] exit {result.returncode}: {(result.stderr or '')[:300]}"
    except subprocess.TimeoutExpired:
        response_text = "[error] timeout (300s)"
    except Exception as e:
        response_text = f"[error] {e}"

    # Capture all new entries from Claude Code's JSONL
    new_entries = read_cc_new_entries(session_file, entry_count_before)
    full_response_parts = []
    for entry in new_entries:
        entry_type = entry.get("type", "")
        if entry_type in ("user", "assistant"):
            content = extract_text_from_cc_entry(entry)
            if content.strip():
                role = entry.get("message", {}).get("role", entry_type)
                if role == "assistant":
                    full_response_parts.append(content)

    # Use the captured response if richer than stdout
    if full_response_parts:
        captured = "\n".join(full_response_parts)
        if len(captured) > len(response_text):
            response_text = captured

    # Log assistant response to ground truth
    log.append("assistant", response_text, thread="default")

    return response_text


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

    # Stable session ID — reused across all turns
    cc_session_id = str(uuid.uuid4())
    cwd = str(Path.home() / "projects")
    model = config.get("model", "claude-sonnet-4-6")

    print(f"continuum | session={session_name} | model={model}")
    print(f"  corpus: {len(idx)} entries")
    print(f"  log: {log.path} ({len(log)} entries)")
    print()

    # Handle initial prompt if provided
    if args.prompt:
        print(f"you> {args.prompt}\n")
        response = run_turn(
            args.prompt, cc_session_id, log, retriever,
            identity_text, system_prompt, config, cwd, model,
        )
        print(f"symbolic> {response}\n")

    # Interactive REPL
    try:
        while True:
            try:
                user_input = input("you> ").strip()
            except EOFError:
                break

            if not user_input:
                continue
            if user_input in ("/quit", "/exit", "/q"):
                break

            print()
            response = run_turn(
                user_input, cc_session_id, log, retriever,
                identity_text, system_prompt, config, cwd, model,
            )
            print(f"symbolic> {response}\n")

    except KeyboardInterrupt:
        pass

    print(f"\nsession saved: {log.path} ({len(log)} entries)")


def launch():
    """Spoof a session and exec into interactive Claude Code.

    Usage:
        python orchestrate.py launch                     # new session
        python orchestrate.py launch --session my-name   # named/resumed
        python orchestrate.py launch "do the thing"      # with initial prompt
        python orchestrate.py launch --cwd /path/to/dir  # custom working dir

    Each invocation:
      1. Appends the prompt (if any) to Continuum's ground truth log
      2. Retrieves relevant context from the corpus
      3. Spoofs a CC session JSONL with assembled context
      4. exec()s into `claude --resume <id>` — full interactive session
      5. On exit, captures CC's responses back to ground truth

    Re-run to get fresh context for the next turn.
    """
    parser = argparse.ArgumentParser(description="Continuum — launch interactive Claude Code with dynamic context")
    parser.add_argument("--config", "-c", help="Path to continuum.yaml")
    parser.add_argument("--session", "-s", help="Session name (for continuity across launches)")
    parser.add_argument("--model", "-m", help="Override model")
    parser.add_argument("--cwd", help="Working directory for Claude Code (default: cwd)")
    parser.add_argument("--yolo", action="store_true", help="Skip permissions (--dangerously-skip-permissions)")
    parser.add_argument("prompt", nargs="?", help="Initial user message (optional)")
    args = parser.parse_args(sys.argv[2:])  # skip 'orchestrate.py launch'

    config = load_config(args.config)
    if args.model:
        config["model"] = args.model

    continuum_dir = Path(__file__).parent
    cwd = args.cwd or str(Path.cwd())
    model = config.get("model", "claude-sonnet-4-6")

    # Identity
    identity_path = Path(config.get("identity", "identity.md"))
    if not identity_path.is_absolute():
        identity_path = continuum_dir / identity_path
    identity_text = identity_path.read_text().strip() if identity_path.exists() else ""

    # System prompt
    sp = config.get("system_prompt", "")
    sp_path = continuum_dir / sp
    system_prompt = sp_path.read_text().strip() if sp_path.is_file() else sp
    if system_prompt:
        identity_text = f"{identity_text}\n\n{system_prompt}" if identity_text else system_prompt

    # Session log (ground truth)
    session_name = args.session or "default"
    log_dir = Path(config["session"]["log_dir"]).expanduser()
    log = SessionLog(log_dir / f"{session_name}.jsonl")

    # Append user message if provided
    if args.prompt:
        log.append("user", args.prompt, thread="default")

    # Retrieve context
    idx = load_index()
    sources = config.get("context_sources", [])
    retriever = ContextRetriever(sources, index=idx)

    token_budgets = config.get("token_budgets", {})
    retrieved = ""
    try:
        query = args.prompt or (log.entries[-1]["content"] if log.entries else "continue")
        recent = log.get_recent(10)
        conversation_tail = "\n".join(
            f"[{e.get('role', '?')}] {e.get('content', '')[:300]}" for e in recent
        )
        retrieved = retriever.retrieve(
            query,
            token_budget=token_budgets.get("dynamic_context", 30000),
            conversation_tail=conversation_tail,
        )
    except Exception:
        pass

    # Build spoofed session
    cc_session_id = str(uuid.uuid4())
    cc_entries = build_spoofed_session(
        session_id=cc_session_id,
        continuum_log=log,
        retrieved_context=retrieved,
        compressed_blocks=[],
        tail_budget=token_budgets.get("recent_tail", 80000),
        identity_text=identity_text,
        cwd=cwd,
    )
    session_file = write_cc_session(cc_session_id, cc_entries, cwd=cwd)
    entry_count_before = len(cc_entries)

    print(f"continuum launch | session={session_name} | model={model}")
    print(f"  corpus: {len(idx)} entries | log: {len(log)} entries")
    print(f"  context: {len(retrieved)} chars retrieved | tail: {len(cc_entries)} CC entries")
    print(f"  cwd: {cwd}")
    print(f"  spoofed: {session_file}")
    print()

    # Build command
    cmd = ["claude", "--resume", cc_session_id]
    if args.yolo:
        cmd.append("--dangerously-skip-permissions")

    # Run claude interactively (inherits stdin/stdout/stderr)
    env = {k: v for k, v in os.environ.items() if k not in ("CLAUDECODE",)}
    try:
        result = subprocess.run(cmd, env=env, cwd=cwd)
    except KeyboardInterrupt:
        pass

    # Capture CC responses back to ground truth
    try:
        new_entries = read_cc_new_entries(session_file, entry_count_before)
        captured = 0
        for entry in new_entries:
            if entry.get("type") == "assistant":
                text = extract_text_from_cc_entry(entry)
                if text.strip():
                    log.append("assistant", text, thread="default")
                    captured += 1
        if captured:
            print(f"[continuum] captured {captured} entries → {log.path}")
    except Exception as e:
        print(f"[continuum] capture failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "launch":
        launch()
    else:
        main()
