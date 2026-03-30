#!/usr/bin/env python3
"""Standalone session spoofing tool — build a resumable CC session from an existing one.

Usage:
    python ~/+/continuum/spoof_tool.py                          # spoof most recent session for CWD
    python ~/+/continuum/spoof_tool.py --session abc123          # specific session
    python ~/+/continuum/spoof_tool.py --context "retrieved text" # inject context
    python ~/+/continuum/spoof_tool.py --identity path/to/id.md  # inject identity
    python ~/+/continuum/spoof_tool.py --compress --local        # compress via local Ollama
"""

import argparse
import glob
import json
import os
import re
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

    If session_id is given, search ALL CC project directories (not just CWD).
    Otherwise, find the most recent session for the given CWD.
    """
    if session_id:
        # Search across all project directories — session ID is globally unique
        for project_dir in _CC_PROJECTS_DIR.iterdir():
            if not project_dir.is_dir():
                continue
            candidate = project_dir / f"{session_id}.jsonl"
            if candidate.is_file():
                return candidate
            # Try prefix match
            matches = sorted(glob.glob(str(project_dir / f"{session_id}*.jsonl")), key=os.path.getmtime, reverse=True)
            if matches:
                return Path(matches[0])
        return None

    # No session ID — most recent session for the given CWD
    cc_dir = _CC_PROJECTS_DIR / _mangle_cwd(cwd)
    if not cc_dir.is_dir():
        return None
    candidates = sorted(glob.glob(str(cc_dir / "*.jsonl")), key=os.path.getmtime, reverse=True)
    return Path(candidates[0]) if candidates else None


def _get_session_cwd(session_file: Path) -> str | None:
    """Read the session's starting CWD from its first entry."""
    try:
        with open(session_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                cwd = entry.get("cwd", "")
                if cwd:
                    return cwd
    except Exception:
        pass
    return None


def _is_spoof_invocation(content: str) -> bool:
    """Detect if a user turn is the /spoof skill invocation or its expansion."""
    if "/spoof" in content[:100]:
        return True
    # Skill expansions contain YAML frontmatter with skill-specific fields
    if ("user-invocable:" in content or "allowed-tools:" in content) and "---" in content[:50]:
        return True
    return False


def _has_identity_exchange(turns: list[dict]) -> bool:
    """Detect if turns already contain a spoofed identity exchange.

    Looks for the "Who are you?" / identity pattern from a previous spoof.
    """
    for i, turn in enumerate(turns[:6]):  # only check first few turns
        content = turn.get("content", "")
        if turn.get("role") == "user" and "Who are you?" in content:
            return True
        if turn.get("role") == "user" and "What do you recall" in content:
            return True
    return False


def _abbreviate_code_in_content(content: str, max_lines: int = 6) -> str:
    """Abbreviate long code/diff blocks in content to first few lines."""
    def _shorten_block(m):
        fence_open = m.group(1)
        body = m.group(2)
        lines = body.split("\n")
        if len(lines) <= max_lines + 2:
            return m.group(0)
        kept = "\n".join(lines[:max_lines])
        omitted = len(lines) - max_lines
        return f"{fence_open}\n{kept}\n[... {omitted} more lines ...]\n```"

    return re.sub(r"(```\w*)\n([\s\S]*?)\n```", _shorten_block, content)


def _read_cc_conversation(session_file: Path, preserve_tail_tools: bool = False) -> list[dict]:
    """Read user/assistant turns from a CC session JSONL.

    Returns a list of {role, content, ts} dicts.
    Cuts off at the LAST /spoof invocation — earlier spoofs in the session
    don't truncate subsequent work.
    """
    turns = []
    last_spoof_idx = -1  # track where the last /spoof is

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
                ts = entry.get("timestamp", "")
                msg = entry.get("message", {})
                role = msg.get("role", "")
                if role == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str) and content.strip():
                        if _is_spoof_invocation(content):
                            last_spoof_idx = len(turns)
                            continue  # don't include the spoof turn itself
                        turns.append({"role": "user", "content": content, "ts": ts})
                    elif isinstance(content, list):
                        texts = [b.get("text", "") for b in content
                                 if isinstance(b, dict) and b.get("type") == "text"]
                        text = "\n".join(t for t in texts if t.strip())
                        if text.strip():
                            if _is_spoof_invocation(text):
                                last_spoof_idx = len(turns)
                                continue
                            turns.append({"role": "user", "content": text, "ts": ts})
                elif role == "assistant":
                    text = extract_text_from_cc_entry(entry)
                    if text.strip():
                        turns.append({"role": "assistant", "content": text, "ts": ts})
    except Exception:
        pass

    # If the last spoof is the final thing in the file (current invocation),
    # we already skipped it and return everything before it.
    # If there were earlier spoofs with work after them, that work is included.
    return turns


def _read_cc_raw_tail(session_file: Path, tail_count: int) -> list[dict]:
    """Read the last N user/assistant raw CC entries from a session, preserving tool calls.

    Returns raw CC JSONL entries (not simplified turns) for full-fidelity tail.
    Skips /spoof invocation entries but doesn't truncate at them — work
    after an earlier /spoof is included.
    """
    all_entries = []
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

                # Skip spoof invocation entries (don't include them in output)
                # but keep reading — work after an earlier /spoof is valid
                msg = entry.get("message", {})
                content = msg.get("content", "")
                if isinstance(content, str) and _is_spoof_invocation(content):
                    continue
                if isinstance(content, list):
                    texts = [b.get("text", "") for b in content
                             if isinstance(b, dict) and b.get("type") == "text"]
                    if any(_is_spoof_invocation(t) for t in texts):
                        continue

                all_entries.append(entry)
    except Exception:
        return []

    return all_entries[-tail_count:] if tail_count < len(all_entries) else all_entries


def main():
    parser = argparse.ArgumentParser(description="Spoof a CC session with optional context injection")
    parser.add_argument("--session", default=None, help="Source CC session ID (default: most recent for CWD)")
    parser.add_argument("--context", default="", help="Retrieved context text to inject")
    parser.add_argument("--identity", default=None, help="Path to identity markdown file")
    parser.add_argument("--cwd", default=None, help="Working directory (default: current)")
    parser.add_argument("--compress", action="store_true", help="LLM-compress session into clean narrative")
    parser.add_argument("--local", action="store_true", help="Use local Ollama model for compression instead of Claude")
    parser.add_argument("--local-model", default="qwen2.5:7b", help="Ollama model for --local (default: qwen2.5:7b)")
    parser.add_argument("--prompt", default="", help="Steering prompt for compression (used with --compress)")
    parser.add_argument("--no-ingest", action="store_true", help="Skip auto-ingest check")
    parser.add_argument("--tail-entries", type=int, default=20, help="Number of raw tail entries to preserve (default: 20)")
    args = parser.parse_args()

    cwd = args.cwd or os.getcwd()

    if not args.no_ingest:
        auto_ingest()

    # Find source session
    source_file = _find_source_session(cwd, args.session)
    if source_file is None:
        print(f"[continuum:spoof] no CC session found for {cwd}", file=sys.stderr)
        sys.exit(1)

    # Use the session's own starting CWD, not the current working directory
    session_cwd = _get_session_cwd(source_file)
    if session_cwd:
        cwd = session_cwd

    source_id = source_file.stem
    print(f"[continuum:spoof] source={source_id[:12]}", file=sys.stderr)
    print(f"  source_dir: {source_file.parent}", file=sys.stderr)
    print(f"  session_cwd: {cwd}", file=sys.stderr)

    # Read conversation turns (cuts at /spoof invocation)
    cc_turns = _read_cc_conversation(source_file)

    # Read raw tail entries (preserves tool_use, tool_result, code diffs)
    raw_tail = _read_cc_raw_tail(source_file, args.tail_entries)

    # Count raw entries for reporting
    raw_count = 0
    try:
        with open(source_file) as f:
            raw_count = sum(1 for line in f if line.strip())
    except Exception:
        pass

    print(f"  text turns: {len(cc_turns)} (from {raw_count} raw entries)", file=sys.stderr)
    print(f"  raw tail: {len(raw_tail)} entries (--tail-entries={args.tail_entries})", file=sys.stderr)

    # Detect if identity is already present (from a previous spoof)
    already_has_identity = _has_identity_exchange(cc_turns)
    print(f"  identity_present: {already_has_identity}", file=sys.stderr)

    # Capture source time range from all turns
    source_timestamps = [t.get("ts", "") for t in cc_turns if t.get("ts")]
    source_time_range = None
    if source_timestamps:
        source_time_range = (source_timestamps[0], source_timestamps[-1])

    # Figure out where the raw tail starts in the text-extracted turns
    # so we can exclude those from the head (avoid double-counting)
    raw_tail_uuids = {e.get("uuid", "") for e in raw_tail if e.get("uuid")}

    # Split: head (for compression) + raw tail (full fidelity)
    # The raw tail is already extracted. Remove those turns from the text-extracted
    # list to get the head. Use raw_tail count as the split point.
    head_time_range = None
    if raw_tail and len(raw_tail) < len(cc_turns):
        head_turns = cc_turns[:-len(raw_tail)]
    elif raw_tail:
        head_turns = []
    else:
        head_turns = cc_turns

    print(f"  head_turns: {len(head_turns)}, compress={args.compress}", file=sys.stderr)

    if args.compress and len(head_turns) > 20:
        from core.session_compress import compress_session

        raw_count_turns = len(head_turns)

        # Capture head time window before compression destroys timestamps
        head_ts = [t.get("ts", "") for t in head_turns if t.get("ts")]
        if head_ts:
            head_time_range = (head_ts[0], head_ts[-1])

        head_turns = compress_session(
            head_turns,
            user_prompt=args.prompt,
            use_local=args.local,
            local_model=args.local_model,
        )

        print(f"  compressed {raw_count_turns}→{len(head_turns)} head turns + {len(raw_tail)} raw tail entries", file=sys.stderr)
    else:
        print(f"  {len(head_turns)} head turns + {len(raw_tail)} raw tail entries (no compression)", file=sys.stderr)

    # head_turns become the SessionLog; raw_tail gets appended as raw CC entries
    cc_turns = head_turns

    # Build an in-memory SessionLog from the head turns only
    tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
    tmp.close()
    try:
        log = SessionLog(tmp.name)
        for turn in cc_turns:
            log.append(turn["role"], turn["content"], ts=turn.get("ts", ""))

        # Load identity — skip if source already has it
        identity_text = ""
        if not already_has_identity:
            if args.identity:
                identity_path = Path(args.identity).expanduser()
                if identity_path.is_file():
                    identity_text = identity_path.read_text(encoding="utf-8").strip()
            else:
                # Default identity
                default_identity = _CONTINUUM_DIR / "identity.md"
                if default_identity.is_file():
                    identity_text = default_identity.read_text(encoding="utf-8").strip()

            if identity_text:
                print(f"  injecting identity ({len(identity_text)} chars)", file=sys.stderr)
        else:
            print(f"  identity already present, skipping injection", file=sys.stderr)

        # Retrieved context
        retrieved_context = args.context or ""

        # Generate session and build spoofed JSONL
        cc_session_id = str(uuid.uuid4())
        print(f"  building spoofed session: log={len(log.entries)} entries, raw_tail={len(raw_tail) if raw_tail else 0}", file=sys.stderr)
        entries = build_spoofed_session(
            session_id=cc_session_id,
            continuum_log=log,
            retrieved_context=retrieved_context,
            identity_text=identity_text,
            cwd=cwd,
            source_time_range=source_time_range,
            head_time_range=head_time_range,
            raw_tail_entries=raw_tail if raw_tail else None,
        )

        # Write to the same CC project directory as the source session
        # so `claude --resume <id>` works from the same working directory
        out_path = write_cc_session(cc_session_id, entries, cwd=cwd,
                                    target_dir=source_file.parent)
        print(f"  wrote {len(entries)} entries to {out_path}", file=sys.stderr)

        # Save last spoof for easy resume
        last_spoof_dir = Path.home() / ".continuum"
        last_spoof_dir.mkdir(parents=True, exist_ok=True)
        (last_spoof_dir / ".last_spoof").write_text(cc_session_id)

        # Save identity for system prompt injection
        if identity_text:
            (last_spoof_dir / ".last_identity").write_text(identity_text)

        # Report — breakdown of what's in the spoofed session
        identity_entries = 2 if identity_text else 0
        context_entries = (1 + len(retrieved_context.split("\n\n"))) if retrieved_context else 0
        print(f"  output breakdown: {identity_entries} identity + {len(log.entries)} head + {len(raw_tail) if raw_tail else 0} raw_tail = {len(entries)} total", file=sys.stderr)

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
