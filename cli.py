#!/usr/bin/env python3
"""Continuum CLI — interactive session with dynamic context."""

import sys
import readline
import argparse
from pathlib import Path
from datetime import datetime, timezone

from config import load_config
from session_log import SessionLog
from retrieval import ContextRetriever
from compression import FixedTailPolicy, TokenBudgetPolicy
from continuum import ContextAssembler, Session


def make_session(config: dict, session_name: str | None = None) -> Session:
    """Create a Session from config."""
    # Resolve identity path relative to config or codebase
    identity_path = Path(config["identity"]).expanduser()
    if not identity_path.is_absolute():
        identity_path = Path(__file__).parent / config["identity"]

    # System prompt: file path or inline string
    system_prompt = config.get("system_prompt", "You are a helpful assistant.")
    sp_path = Path(system_prompt).expanduser()
    if not sp_path.is_absolute():
        sp_path = Path(__file__).parent / system_prompt
    if sp_path.is_file():
        system_prompt = sp_path.read_text().strip()

    # Retriever
    sources = config.get("context_sources", [])
    retriever = ContextRetriever(sources)

    # Session log
    log_dir = Path(config["session"]["log_dir"]).expanduser()
    if session_name is None:
        session_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"{session_name}.jsonl"
    session_log = SessionLog(log_path)

    # Compression policy
    comp_config = config.get("compression", {})
    policy_name = comp_config.get("policy", "token_budget")
    comp_model = comp_config.get("model", "claude-haiku-4-5-20251001")
    if policy_name == "fixed_tail":
        policy = FixedTailPolicy(model=comp_model)
    else:
        policy = TokenBudgetPolicy(model=comp_model)

    # Assembler
    assembler = ContextAssembler(
        identity_path=identity_path,
        system_prompt=system_prompt,
        retriever=retriever,
        session_log=session_log,
        model=config.get("model", "claude-sonnet-4-6"),
        token_budgets=config.get("token_budgets"),
        compression_policy=policy,
    )

    return Session(assembler)


def main():
    parser = argparse.ArgumentParser(description="Continuum — dynamic context CLI")
    parser.add_argument("--config", "-c", help="Path to continuum.yaml")
    parser.add_argument("--session", "-s", help="Session name (for resume)")
    parser.add_argument("--model", "-m", help="Override model")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.model:
        config["model"] = args.model

    session = make_session(config, args.session)
    log = session.assembler.log

    compressed = session.assembler._compressed
    print(f"continuum | model={config['model']} | session={log.path.stem}")
    print(f"  sources: {len(session.assembler.retriever.sources)}")
    print(f"  compression: {policy_name} (via {comp_model})")
    print(f"  log: {log.path}")
    if log.entries:
        print(f"  resumed: {len(log)} entries, turn {log.turn}, {len(compressed)} compressed blocks")
    print()

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

            try:
                response, stats = session.turn(user_input)
                print(f"\nsymbolic> {response}\n")
                # Observability line
                s = stats
                print(f"  L0:{s.get('identity_tokens',0)}t │ "
                      f"L2:{s.get('dynamic_context_tokens',0)//1000}K({s.get('dynamic_context_files',0)}f) │ "
                      f"L3a:{s.get('compressed_blocks',0)}blk {s.get('compressed_tokens',0)//1000}K │ "
                      f"L3b:{s.get('tail_entries',0)}ent {s.get('tail_tokens',0)//1000}K │ "
                      f"total:{s.get('total_tokens',0)//1000}K\n")
            except Exception as e:
                print(f"\n[error] {e}\n", file=sys.stderr)

    except KeyboardInterrupt:
        pass

    print(f"\nsession saved: {log.path} ({len(log)} entries)")


if __name__ == "__main__":
    main()
