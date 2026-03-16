#!/usr/bin/env python3
"""Continuum Ingest — convert external session logs into the Continuum corpus.

Reusable tool for ingesting Claude Code session logs (or any JSONL source)
into Continuum's format with UIDs, thread detection, and optional embeddings.

Usage:
    # Ingest all Claude Code sessions
    python ingest.py claude-code

    # Ingest a specific session file
    python ingest.py file ~/.claude/projects/-home-symbolic-projects/SESSION_ID.jsonl

    # Ingest with embeddings (slower but enables semantic search)
    python ingest.py claude-code --embed

    # Dry run — show what would be ingested
    python ingest.py claude-code --dry-run
"""

import argparse
import json
import re
import sys
from pathlib import Path

from session_log import mint_uid


# ── Claude Code format parsing ─────────────────────────────────────────

def detect_project_from_path(session_path: Path) -> str:
    """Infer project name from Claude Code's directory structure.

    ~/.claude/projects/-home-symbolic-projects-bridge/SESSION.jsonl
    → "bridge"
    """
    parent = session_path.parent.name
    # Pattern: -home-symbolic-projects-PROJECT or -home-symbolic-projects
    match = re.match(r"-home-symbolic-projects?-?(.*)", parent)
    if match:
        project = match.group(1).replace("-", "/") if match.group(1) else "home"
        return project
    # Pattern: -home-symbolic---bridge etc.
    match = re.match(r"-home-symbolic---?(.*)", parent)
    if match:
        return match.group(1).replace("-", "/") if match.group(1) else "unknown"
    return "unknown"


def extract_assistant_text(content) -> str:
    """Extract readable text from Claude Code assistant message content.

    Content can be:
    - str: plain text
    - list: array of blocks (thinking, text, tool_use, tool_result)
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            if btype == "text":
                parts.append(block.get("text", ""))
            elif btype == "tool_use":
                name = block.get("name", "?")
                inp = block.get("input", {})
                # Compact tool use summary
                if name in ("Read", "Glob", "Grep"):
                    target = inp.get("file_path") or inp.get("pattern") or inp.get("path", "")
                    parts.append(f"[{name}: {target}]")
                elif name == "Edit":
                    parts.append(f"[Edit: {inp.get('file_path', '?')}]")
                elif name == "Write":
                    parts.append(f"[Write: {inp.get('file_path', '?')}]")
                elif name == "Bash":
                    cmd = inp.get("command", "")[:100]
                    parts.append(f"[Bash: {cmd}]")
                else:
                    parts.append(f"[{name}]")
            # Skip thinking and tool_result — too noisy for corpus
        return "\n".join(parts)

    return str(content)[:500]


def convert_claude_code_session(
    source_path: Path,
    project: str,
    embed: bool = False,
) -> list[dict]:
    """Convert a single Claude Code session JSONL to Continuum format.

    Returns list of Continuum-format entries.
    """
    entries = []
    turn = 0

    with open(source_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue

            entry_type = raw.get("type", "")
            if entry_type not in ("user", "assistant"):
                continue

            msg = raw.get("message", {})
            role = msg.get("role", entry_type)
            content_raw = msg.get("content", "")

            if role == "user":
                turn += 1
                content = content_raw if isinstance(content_raw, str) else str(content_raw)
            else:
                content = extract_assistant_text(content_raw)

            # Skip empty entries
            if not content.strip():
                continue

            entry = {
                "uid": mint_uid(),
                "role": role,
                "content": content,
                "turn": turn,
                "ts": raw.get("timestamp", ""),
                "thread": project,
                "source_session": raw.get("sessionId", source_path.stem),
                "source_uuid": raw.get("uuid", ""),
            }

            # Optional embedding
            if embed:
                try:
                    from embeddings import embed_text
                    vec = embed_text(content)
                    if vec is not None:
                        entry["embedding"] = vec
                except Exception:
                    pass

            entries.append(entry)

    return entries


# ── Corpus writer ──────────────────────────────────────────────────────

def write_corpus(entries: list[dict], output_path: Path):
    """Write entries to a corpus JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


# ── CLI ────────────────────────────────────────────────────────────────

def ingest_claude_code(args):
    """Ingest all Claude Code session logs."""
    source_dir = Path.home() / ".claude" / "projects"
    if not source_dir.exists():
        print(f"No Claude Code sessions found at {source_dir}")
        return

    output_dir = Path.home() / ".continuum" / "corpus"
    session_files = sorted(source_dir.rglob("*.jsonl"))

    # Skip subagent logs
    session_files = [f for f in session_files if "subagent" not in str(f)]

    print(f"Found {len(session_files)} session logs")

    total_entries = 0
    total_sessions = 0
    skipped = 0

    for sf in session_files:
        project = detect_project_from_path(sf)
        out_path = output_dir / project / f"{sf.stem}.jsonl"

        # Skip if already converted (unless --force)
        if out_path.exists() and not args.force:
            skipped += 1
            continue

        if args.dry_run:
            print(f"  [dry-run] {sf.name} → {project} ({sf.stat().st_size // 1024}KB)")
            total_sessions += 1
            continue

        entries = convert_claude_code_session(sf, project, embed=args.embed)
        if not entries:
            continue

        write_corpus(entries, out_path)
        total_entries += len(entries)
        total_sessions += 1
        print(f"  {sf.stem[:12]}... → {project} ({len(entries)} entries)")

    print(f"\nIngested: {total_sessions} sessions, {total_entries} entries")
    if skipped:
        print(f"Skipped: {skipped} (already converted, use --force to re-convert)")
    if not args.dry_run:
        print(f"Corpus: {output_dir}")


def ingest_file(args):
    """Ingest a single JSONL file."""
    source = Path(args.path).expanduser()
    if not source.exists():
        print(f"File not found: {source}")
        return

    project = args.project or detect_project_from_path(source)
    output_dir = Path.home() / ".continuum" / "corpus"
    out_path = output_dir / project / f"{source.stem}.jsonl"

    if args.dry_run:
        print(f"[dry-run] {source} → {project}")
        return

    entries = convert_claude_code_session(source, project, embed=args.embed)
    write_corpus(entries, out_path)
    print(f"Ingested: {len(entries)} entries → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Continuum Ingest — convert session logs to corpus")

    sub = parser.add_subparsers(dest="command")

    cc_parser = sub.add_parser("claude-code", help="Ingest all Claude Code session logs")
    cc_parser.add_argument("--embed", action="store_true", help="Mint embeddings per entry (slower)")
    cc_parser.add_argument("--dry-run", action="store_true", help="Show what would be ingested")
    cc_parser.add_argument("--force", action="store_true", help="Re-convert already-converted sessions")

    file_parser = sub.add_parser("file", help="Ingest a single JSONL file")
    file_parser.add_argument("path", help="Path to JSONL file")
    file_parser.add_argument("--project", help="Override project detection")
    file_parser.add_argument("--embed", action="store_true", help="Mint embeddings per entry (slower)")
    file_parser.add_argument("--dry-run", action="store_true", help="Show what would be ingested")

    args = parser.parse_args()

    if args.command == "claude-code":
        ingest_claude_code(args)
    elif args.command == "file":
        ingest_file(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
