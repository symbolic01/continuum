#!/usr/bin/env python3
"""Standalone session spoofing tool — build a resumable CC session from an existing one.

Usage:
    python ~/+/continuum/spoof_tool.py                          # spoof most recent session for CWD
    python ~/+/continuum/spoof_tool.py --session abc123          # specific session
    python ~/+/continuum/spoof_tool.py --context "retrieved text" # inject context
    python ~/+/continuum/spoof_tool.py --identity path/to/id.md  # inject identity
"""

import argparse
import glob
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path

# Ensure continuum modules are importable
_CONTINUUM_DIR = Path(__file__).resolve().parent
if str(_CONTINUUM_DIR) not in sys.path:
    sys.path.insert(0, str(_CONTINUUM_DIR))

from core.auto_ingest import auto_ingest
from core.session_spoof import (
    build_spoofed_session,
    write_cc_session,
    extract_text_from_cc_entry,
)
from core.session_log import SessionLog

import shlex

_CC_PROJECTS_DIR = Path.home() / ".claude" / "projects"


def _quote_for_shell(text: str) -> str:
    """Quote text for safe shell embedding."""
    return shlex.quote(text)


def _mangle_cwd(cwd: str) -> str:
    mangled = cwd.replace("/", "-")
    if not mangled.startswith("-"):
        mangled = "-" + mangled
    return mangled


def _find_source_session(cwd: str, session_id: str | None = None) -> Path | None:
    """Find a CC session JSONL file.

    If session_id is given, look for that specific file.
    Otherwise, find the most recent session for the given CWD.
    """
    cc_dir = _CC_PROJECTS_DIR / _mangle_cwd(cwd)
    if not cc_dir.is_dir():
        return None

    if session_id:
        # Try exact match
        candidate = cc_dir / f"{session_id}.jsonl"
        if candidate.is_file():
            return candidate
        # Try prefix match
        matches = sorted(glob.glob(str(cc_dir / f"{session_id}*.jsonl")), key=os.path.getmtime, reverse=True)
        return Path(matches[0]) if matches else None

    # Most recent by mtime
    candidates = sorted(glob.glob(str(cc_dir / "*.jsonl")), key=os.path.getmtime, reverse=True)
    return Path(candidates[0]) if candidates else None


def _read_cc_conversation(session_file: Path) -> list[dict]:
    """Read user/assistant turns from a CC session JSONL.

    Returns a list of {role, content} dicts — clean text only, no tool use.
    Consecutive same-role entries are preserved (CC renders each as a separate bullet).
    """
    turns = []
    try:
        with open(session_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                entry_type = entry.get("type", "")
                if entry_type not in ("user", "assistant"):
                    continue
                msg = entry.get("message", {})
                role = msg.get("role", "")
                if role == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str) and content.strip():
                        turns.append({"role": "user", "content": content})
                    elif isinstance(content, list):
                        # Extract text blocks from list content
                        texts = [b.get("text", "") for b in content
                                 if isinstance(b, dict) and b.get("type") == "text"]
                        text = "\n".join(t for t in texts if t.strip())
                        if text.strip():
                            turns.append({"role": "user", "content": text})
                elif role == "assistant":
                    text = extract_text_from_cc_entry(entry)
                    if text.strip() and not text.startswith("["):
                        turns.append({"role": "assistant", "content": text})
    except Exception:
        pass
    return turns


def main():
    parser = argparse.ArgumentParser(description="Spoof a CC session with optional context injection")
    parser.add_argument("--session", default=None, help="Source CC session ID (default: most recent for CWD)")
    parser.add_argument("--context", default="", help="Retrieved context text to inject")
    parser.add_argument("--identity", default=None, help="Path to identity markdown file")
    parser.add_argument("--cwd", default=None, help="Working directory (default: current)")
    parser.add_argument("--compress", action="store_true", help="LLM-compress session into clean narrative")
    parser.add_argument("--prompt", default="", help="Steering prompt for compression (used with --compress)")
    parser.add_argument("--no-ingest", action="store_true", help="Skip auto-ingest check")
    args = parser.parse_args()

    cwd = args.cwd or os.getcwd()

    if not args.no_ingest:
        auto_ingest()

    # Find source session
    source_file = _find_source_session(cwd, args.session)
    if source_file is None:
        print(f"[continuum:spoof] no CC session found for {cwd}", file=sys.stderr)
        sys.exit(1)

    source_id = source_file.stem
    print(f"[continuum:spoof] source={source_id[:10]}", file=sys.stderr, end="")

    # Read conversation turns
    cc_turns = _read_cc_conversation(source_file)

    # Count raw entries for reporting
    raw_count = 0
    try:
        with open(source_file) as f:
            raw_count = sum(1 for line in f if line.strip())
    except Exception:
        pass

    print(f" ({len(cc_turns)} turns from {raw_count} entries)", file=sys.stderr)

    # Compress if requested — keep recent tail raw
    if args.compress:
        from core.session_compress import compress_session

        # Split: compress the bulk, keep the recent tail verbatim
        # ~50 lines of screen content ≈ last few turns
        tail_chars = 0
        tail_start = len(cc_turns)
        for i in range(len(cc_turns) - 1, -1, -1):
            tail_chars += len(cc_turns[i].get("content", "")) + 20  # +20 for role overhead
            if tail_chars >= 3000:  # ~50 lines × 60 chars
                tail_start = i
                break

        head = cc_turns[:tail_start]
        tail = cc_turns[tail_start:]
        raw_count_turns = len(cc_turns)

        if len(head) > 20:
            head = compress_session(head, user_prompt=args.prompt)

        cc_turns = head + tail
        print(f"  compressed {raw_count_turns}→{len(cc_turns)} turns ({len(tail)} raw tail)", file=sys.stderr)

    # Build an in-memory SessionLog
    tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
    tmp.close()
    try:
        log = SessionLog(tmp.name)
        for turn in cc_turns:
            log.append(turn["role"], turn["content"])

        # Load identity
        identity_text = ""
        if args.identity:
            identity_path = Path(args.identity).expanduser()
            if identity_path.is_file():
                identity_text = identity_path.read_text(encoding="utf-8").strip()
        else:
            # Default identity
            default_identity = _CONTINUUM_DIR / "identity.md"
            if default_identity.is_file():
                identity_text = default_identity.read_text(encoding="utf-8").strip()

        # Retrieved context
        retrieved_context = args.context or ""

        # Generate session and build spoofed JSONL
        cc_session_id = str(uuid.uuid4())
        entries = build_spoofed_session(
            session_id=cc_session_id,
            continuum_log=log,
            retrieved_context=retrieved_context,
            identity_text=identity_text,
            cwd=cwd,
        )

        write_cc_session(cc_session_id, entries, cwd=cwd)

        # Save last spoof for easy resume
        last_spoof_dir = Path.home() / ".continuum"
        last_spoof_dir.mkdir(parents=True, exist_ok=True)
        (last_spoof_dir / ".last_spoof").write_text(cc_session_id)

        # Save identity for system prompt injection
        if identity_text:
            (last_spoof_dir / ".last_identity").write_text(identity_text)

        # Report
        ctx_chars = len(retrieved_context) if retrieved_context else 0
        tail_count = max(0, len(entries) - (2 if (identity_text or retrieved_context) else 0))
        print(f"  context: {ctx_chars / 1000:.1f}K chars | tail: {tail_count} entries", file=sys.stderr)

        # Build resume command — include --append-system-prompt if identity exists
        resume_cmd = f"claude --resume {cc_session_id}"
        if identity_text:
            resume_cmd += f" --append-system-prompt {_quote_for_shell(identity_text)}"
        print(f"  {resume_cmd}", file=sys.stderr)

        # Print session ID to stdout for scripting
        print(cc_session_id)

    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


if __name__ == "__main__":
    main()
