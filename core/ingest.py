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
import os
import re
import subprocess
import sys
from pathlib import Path

from .session_log import mint_uid


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

            # Skip low-signal entries that waste embeddings and pollute search.
            stripped = content.strip()
            if role == "assistant":
                # Pure tool-use bracket summaries: [Read: path], [TaskUpdate], etc.
                if stripped.startswith("[") and stripped.endswith("]") and len(stripped) < 200:
                    continue
                # Common filler responses
                if stripped in ("No response requested.",
                                "I have this context. Ready to continue."):
                    continue
            if role == "user":
                # Tool result rejection boilerplate
                if "The user doesn't want to proceed with this tool use" in stripped and len(stripped) < 400:
                    continue
                # Empty tool results
                if stripped.startswith("[{'tool_use_id':") and len(stripped) < 80:
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

            entries.append(entry)

    # Batch embed all entries at once
    if embed and entries:
        try:
            from .embeddings import embed_batch
            texts = [e["content"] for e in entries]
            vectors = embed_batch(texts)
            for entry, vec in zip(entries, vectors):
                if vec is not None:
                    entry["embedding"] = vec
        except Exception:
            pass

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

        # Skip if already converted AND source hasn't grown (unless --force)
        if out_path.exists() and not args.force:
            if sf.stat().st_mtime <= out_path.stat().st_mtime:
                skipped += 1
                continue
            # Source is newer — re-ingest (session grew since last ingest)

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


# ── Markdown ingestion ─────────────────────────────────────────────────

def chunk_markdown(text: str, source_path: str) -> list[dict]:
    """Split markdown into chunks by heading. Each chunk = one section.

    Returns list of {"content": str, "heading": str, "source": str}.
    """
    chunks = []
    current_heading = "preamble"
    current_lines = []

    for line in text.split("\n"):
        if line.startswith("#"):
            # Save previous chunk
            if current_lines:
                content = "\n".join(current_lines).strip()
                if content:
                    chunks.append({
                        "content": content,
                        "heading": current_heading,
                        "source": source_path,
                    })
            current_heading = line.lstrip("#").strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    # Save last chunk
    if current_lines:
        content = "\n".join(current_lines).strip()
        if content:
            chunks.append({
                "content": content,
                "heading": current_heading,
                "source": source_path,
            })

    return chunks


def convert_markdown_file(source_path: Path, project: str, embed: bool = False) -> list[dict]:
    """Convert a markdown file into Continuum corpus entries (one per heading section)."""
    text = source_path.read_text(encoding="utf-8", errors="replace")
    chunks = chunk_markdown(text, str(source_path))
    entries = []

    for chunk in chunks:
        entry = {
            "uid": mint_uid(),
            "role": "context",
            "content": chunk["content"],
            "turn": 0,
            "ts": "",
            "thread": project,
            "source_file": chunk["source"],
            "heading": chunk["heading"],
        }

        entries.append(entry)

    # Batch embed all entries at once
    if embed and entries:
        try:
            from .embeddings import embed_batch
            texts = [e["content"] for e in entries]
            vectors = embed_batch(texts)
            for entry, vec in zip(entries, vectors):
                if vec is not None:
                    entry["embedding"] = vec
        except Exception:
            pass

    return entries


def ingest_markdown(args):
    """Ingest markdown files from source directories into the corpus."""
    source_dirs = [Path(s).expanduser() for s in (args.sources or [str(Path.home() / "projects")])]
    output_dir = Path.home() / ".continuum" / "corpus"

    # Find all markdown files across all source dirs
    md_files = []
    source_roots = []  # track which root each file came from
    for src_dir in source_dirs:
        if not src_dir.is_dir():
            print(f"  skipping {src_dir} (not found)", file=sys.stderr)
            continue
        found = sorted(src_dir.rglob("*.md"))
        # Filter to CLAUDE.md and plans/*.md
        found = [f for f in found if
                 f.name == "CLAUDE.md" or
                 "plans/" in str(f.relative_to(src_dir)) or
                 "plans\\" in str(f.relative_to(src_dir))]
        md_files.extend(found)
        source_roots.extend([src_dir] * len(found))

    print(f"Found {len(md_files)} markdown files across {len(source_dirs)} source(s)")

    total_entries = 0
    total_files = 0

    for mf, root_dir in zip(md_files, source_roots):
        rel = mf.relative_to(root_dir)
        # Detect project from path
        parts = rel.parts
        if len(parts) > 1:
            project = str(Path(*parts[:-1]))
        else:
            project = "home"

        out_path = output_dir / "_markdown" / f"{str(rel).replace('/', '_')}.jsonl"

        if out_path.exists() and not args.force:
            continue

        if args.dry_run:
            print(f"  [dry-run] {rel} → {project}")
            total_files += 1
            continue

        entries = convert_markdown_file(mf, project, embed=args.embed)
        if not entries:
            continue

        write_corpus(entries, out_path)
        total_entries += len(entries)
        total_files += 1
        print(f"  {rel} → {project} ({len(entries)} sections)")

    print(f"\nIngested: {total_files} files, {total_entries} sections")
    if not args.dry_run:
        print(f"Corpus: {output_dir / '_markdown'}")


# ── Codebase ingestion ────────────────────────────────────────────────

# File extensions to ingest
_CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".sh", ".bash",
    ".html", ".css", ".json", ".yaml", ".yml", ".toml",
    ".go", ".rs", ".java", ".c", ".cpp", ".h",
}

# Directories to skip
_SKIP_DIRS = {
    "__pycache__", "node_modules", ".git", ".venv", "venv",
    "dist", "build", ".next", ".cache", "vendor", ".tox",
    "egg-info", ".eggs", ".mypy_cache", ".pytest_cache",
}


def _chunk_python(text: str, file_path: str) -> list[dict]:
    """Chunk Python source into file-level + class/method-level entries."""
    chunks = []
    lines = text.split("\n")

    # Always add file-level chunk (truncated)
    file_preview = text[:2000] if len(text) > 2000 else text
    chunks.append({
        "content": file_preview,
        "heading": file_path,
        "source": file_path,
        "chunk_type": "file",
    })

    # Extract class and function definitions
    current_def = None
    current_lines = []
    current_indent = 0

    for i, line in enumerate(lines):
        stripped = line.lstrip()

        # Detect class/function definitions at module level or class level
        is_def = False
        if stripped.startswith("class ") or stripped.startswith("def ") or stripped.startswith("async def "):
            indent = len(line) - len(stripped)
            # New top-level or class-level definition
            if indent <= current_indent or current_def is None:
                # Save previous definition
                if current_def and current_lines:
                    content = "\n".join(current_lines)
                    if len(content.strip()) > 20:  # skip trivial defs
                        chunks.append({
                            "content": content,
                            "heading": f"{file_path}:{current_def}",
                            "source": file_path,
                            "chunk_type": "class" if current_def.startswith("class ") else "function",
                        })

                # Start new definition
                if stripped.startswith("class "):
                    name = stripped.split("(")[0].split(":")[0].replace("class ", "")
                    current_def = f"class {name}"
                else:
                    name = stripped.replace("async def ", "def ").split("(")[0].replace("def ", "")
                    current_def = f"def {name}"
                current_lines = [line]
                current_indent = indent
                is_def = True

        if not is_def and current_def is not None:
            current_lines.append(line)

    # Save last definition
    if current_def and current_lines:
        content = "\n".join(current_lines)
        if len(content.strip()) > 20:
            chunks.append({
                "content": content,
                "heading": f"{file_path}:{current_def}",
                "source": file_path,
                "chunk_type": "class" if current_def.startswith("class ") else "function",
            })

    return chunks


def _chunk_js(text: str, file_path: str) -> list[dict]:
    """Chunk JS/TS source into file-level + function-level entries."""
    chunks = []

    # File-level chunk
    file_preview = text[:2000] if len(text) > 2000 else text
    chunks.append({
        "content": file_preview,
        "heading": file_path,
        "source": file_path,
        "chunk_type": "file",
    })

    # Extract function/class definitions
    func_re = re.compile(
        r'^(?:export\s+)?(?:async\s+)?(?:function\s+(\w+)|class\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s+)?\()',
        re.MULTILINE,
    )
    for m in func_re.finditer(text):
        name = m.group(1) or m.group(2) or m.group(3)
        if not name:
            continue
        # Extract ~50 lines from the match
        start = m.start()
        end = text.find("\n", start)
        line_count = 0
        pos = start
        while pos < len(text) and line_count < 50:
            nl = text.find("\n", pos)
            if nl == -1:
                break
            pos = nl + 1
            line_count += 1
        chunk_text = text[start:pos]
        if len(chunk_text.strip()) > 20:
            kind = "class" if m.group(2) else "function"
            chunks.append({
                "content": chunk_text,
                "heading": f"{file_path}:{name}",
                "source": file_path,
                "chunk_type": kind,
            })

    return chunks


def _chunk_java(text: str, file_path: str) -> list[dict]:
    """Chunk Java source into file-level + class/method-level entries."""
    chunks = []

    # File-level chunk
    file_preview = text[:2000] if len(text) > 2000 else text
    chunks.append({
        "content": file_preview,
        "heading": file_path,
        "source": file_path,
        "chunk_type": "file",
    })

    # Extract class, interface, enum, and method definitions
    # Matches: public class Foo, private void bar(, static int baz(, etc.
    def_re = re.compile(
        r'^[ \t]*(?:(?:public|private|protected|static|final|abstract|synchronized|native)\s+)*'
        r'(?:'
        r'(?:class|interface|enum|record)\s+(\w+)'  # group 1: class/interface/enum name
        r'|'
        r'(?:[\w<>\[\],\s]+?)\s+(\w+)\s*\('         # group 2: method name (return_type methodName()
        r')',
        re.MULTILINE,
    )

    lines = text.split("\n")
    for m in def_re.finditer(text):
        class_name = m.group(1)
        method_name = m.group(2)
        name = class_name or method_name
        if not name or len(name) < 2:
            continue
        # Skip common false positives
        if name in ("if", "for", "while", "switch", "catch", "new", "return", "throw"):
            continue

        kind = "class" if class_name else "method"

        # Find the line number and extract up to the closing brace
        start = m.start()
        # Count braces to find the end of the block
        brace_depth = 0
        found_open = False
        pos = start
        end = len(text)
        char_limit = 5000  # don't extract more than 5K chars per chunk
        while pos < len(text) and (pos - start) < char_limit:
            ch = text[pos]
            if ch == '{':
                brace_depth += 1
                found_open = True
            elif ch == '}':
                brace_depth -= 1
                if found_open and brace_depth <= 0:
                    end = pos + 1
                    break
            pos += 1

        chunk_text = text[start:end]
        if len(chunk_text.strip()) > 20:
            chunks.append({
                "content": chunk_text,
                "heading": f"{file_path}:{name}",
                "source": file_path,
                "chunk_type": kind,
            })

    return chunks


def _chunk_go(text: str, file_path: str) -> list[dict]:
    """Chunk Go source into file-level + type/func-level entries."""
    chunks = []

    # File-level chunk
    file_preview = text[:2000] if len(text) > 2000 else text
    chunks.append({
        "content": file_preview,
        "heading": file_path,
        "source": file_path,
        "chunk_type": "file",
    })

    # Match: func Name(, func (r *Receiver) Name(, type Name struct/interface
    def_re = re.compile(
        r'^(?:'
        r'func\s+(?:\([^)]+\)\s+)?(\w+)\s*\('   # group 1: func name (with optional receiver)
        r'|'
        r'type\s+(\w+)\s+(?:struct|interface)\b'  # group 2: type name
        r')',
        re.MULTILINE,
    )

    for m in def_re.finditer(text):
        func_name = m.group(1)
        type_name = m.group(2)
        name = func_name or type_name
        if not name:
            continue

        kind = "type" if type_name else "function"

        # Brace-matching to find end of block
        start = m.start()
        brace_depth = 0
        found_open = False
        pos = start
        end = len(text)
        char_limit = 5000
        while pos < len(text) and (pos - start) < char_limit:
            ch = text[pos]
            if ch == '{':
                brace_depth += 1
                found_open = True
            elif ch == '}':
                brace_depth -= 1
                if found_open and brace_depth <= 0:
                    end = pos + 1
                    break
            pos += 1

        chunk_text = text[start:end]
        if len(chunk_text.strip()) > 20:
            chunks.append({
                "content": chunk_text,
                "heading": f"{file_path}:{name}",
                "source": file_path,
                "chunk_type": kind,
            })

    return chunks


def _chunk_json(text: str, file_path: str) -> list[dict]:
    """Chunk JSON/YAML into top-level keys.

    For large JSON files (configs, templates, JSON-e), each top-level key
    becomes its own chunk. Small files stay as a single chunk.
    """
    chunks = []

    # Small files: single chunk
    if len(text) <= 3000:
        chunks.append({
            "content": text,
            "heading": file_path,
            "source": file_path,
            "chunk_type": "file",
        })
        return chunks

    # Try to parse and chunk by top-level keys
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # Unparseable — file-level truncated
        chunks.append({
            "content": text[:3000],
            "heading": file_path,
            "source": file_path,
            "chunk_type": "file",
        })
        return chunks

    if isinstance(data, dict):
        # File overview: list of top-level keys
        key_list = ", ".join(list(data.keys())[:50])
        overview = f"// {file_path}\n// Top-level keys: {key_list}\n"
        if len(data) <= 3:
            # Few keys — just dump the whole thing truncated
            overview += text[:3000]
        chunks.append({
            "content": overview,
            "heading": file_path,
            "source": file_path,
            "chunk_type": "file",
        })

        # Each top-level key as its own chunk
        for key, value in data.items():
            value_str = json.dumps(value, indent=2)
            # Truncate very large values
            if len(value_str) > 3000:
                value_str = value_str[:3000] + "\n  // ... truncated"
            chunk_content = f'"{key}": {value_str}'
            chunks.append({
                "content": chunk_content,
                "heading": f"{file_path}:{key}",
                "source": file_path,
                "chunk_type": "key",
            })

    elif isinstance(data, list):
        # Arrays: overview + sample items
        overview = f"// {file_path}\n// Array with {len(data)} items\n"
        sample = json.dumps(data[:3], indent=2)
        if len(sample) > 3000:
            sample = sample[:3000]
        overview += sample
        chunks.append({
            "content": overview,
            "heading": file_path,
            "source": file_path,
            "chunk_type": "file",
        })
    else:
        chunks.append({
            "content": text[:3000],
            "heading": file_path,
            "source": file_path,
            "chunk_type": "file",
        })

    return chunks


def _chunk_generic(text: str, file_path: str) -> list[dict]:
    """File-level chunk only for unrecognized file types."""
    preview = text[:3000] if len(text) > 3000 else text
    return [{
        "content": preview,
        "heading": file_path,
        "source": file_path,
        "chunk_type": "file",
    }]


def chunk_source_file(text: str, file_path: str) -> list[dict]:
    """Chunk a source code file by file/class/method."""
    ext = Path(file_path).suffix.lower()
    if ext == ".py":
        return _chunk_python(text, file_path)
    elif ext in (".js", ".ts", ".jsx", ".tsx"):
        return _chunk_js(text, file_path)
    elif ext == ".java":
        return _chunk_java(text, file_path)
    elif ext == ".go":
        return _chunk_go(text, file_path)
    elif ext == ".json":
        return _chunk_json(text, file_path)
    else:
        return _chunk_generic(text, file_path)


def convert_codebase(
    codebase_dir: Path,
    project: str,
    embed: bool = False,
) -> list[dict]:
    """Convert a codebase directory into corpus entries (chunked by file/class/method)."""
    entries = []

    for root, dirs, files in os.walk(codebase_dir):
        # Skip unwanted directories
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]

        for fname in sorted(files):
            fpath = Path(root) / fname
            if fpath.suffix.lower() not in _CODE_EXTENSIONS:
                continue
            # Skip very large files
            try:
                if fpath.stat().st_size > 500_000:  # 500KB
                    continue
            except OSError:
                continue

            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
            except (OSError, PermissionError):
                continue

            rel_path = str(fpath.relative_to(codebase_dir))
            chunks = chunk_source_file(text, rel_path)

            for chunk in chunks:
                entries.append({
                    "uid": mint_uid(),
                    "role": "code",
                    "content": chunk["content"],
                    "turn": 0,
                    "ts": "",
                    "thread": project,
                    "source_file": str(fpath),
                    "heading": chunk["heading"],
                    "chunk_type": chunk["chunk_type"],
                })

    # Batch embed
    if embed and entries:
        try:
            from .embeddings import embed_batch
            texts = [e["content"] for e in entries]
            vectors = embed_batch(texts)
            for entry, vec in zip(entries, vectors):
                if vec is not None:
                    entry["embedding"] = vec
        except Exception:
            pass

    return entries


def _discover_codebases() -> list[tuple[str, Path]]:
    """Discover linked codebases from px."""
    codebases = []
    try:
        px_bin = Path.home() / "projects" / "px"
        result = subprocess.run(
            [str(px_bin), "list", "--flat"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return codebases
        for line in result.stdout.strip().split("\n"):
            project = line.strip()
            if not project:
                continue
            cb_result = subprocess.run(
                [str(px_bin), "codebase", project],
                capture_output=True, text=True, timeout=5,
            )
            if cb_result.returncode == 0 and cb_result.stdout.strip():
                cb_path = Path(cb_result.stdout.strip())
                if cb_path.is_dir():
                    codebases.append((project, cb_path))
    except Exception:
        pass
    return codebases


def ingest_codebase(args):
    """Ingest source code from codebase directories."""
    output_dir = Path.home() / ".continuum" / "corpus"

    if args.paths:
        # Explicit paths
        codebases = []
        for p in args.paths:
            path = Path(p).expanduser().resolve()
            if path.is_dir():
                project = path.name
                codebases.append((project, path))
            else:
                print(f"  skipping {p} (not found)", file=sys.stderr)
    else:
        # Auto-discover from px
        codebases = _discover_codebases()

    if not codebases:
        print("No codebases found. Pass paths explicitly or link via px.")
        return

    print(f"Found {len(codebases)} codebase(s): {', '.join(p for p, _ in codebases)}")

    total_entries = 0
    total_codebases = 0

    for project, cb_path in codebases:
        out_path = output_dir / "_codebase" / f"{project}.jsonl"

        if out_path.exists() and not args.force:
            print(f"  {project} (skipped, use --force)")
            continue

        if args.dry_run:
            # Count files
            count = sum(1 for _ in cb_path.rglob("*")
                       if _.suffix.lower() in _CODE_EXTENSIONS
                       and not any(s in str(_) for s in _SKIP_DIRS))
            print(f"  [dry-run] {project} @ {cb_path} ({count} files)")
            total_codebases += 1
            continue

        entries = convert_codebase(cb_path, project, embed=args.embed)
        if not entries:
            continue

        write_corpus(entries, out_path)
        total_entries += len(entries)
        total_codebases += 1
        file_chunks = sum(1 for e in entries if e.get("chunk_type") == "file")
        func_chunks = sum(1 for e in entries if e.get("chunk_type") in ("function", "class"))
        print(f"  {project} → {len(entries)} chunks ({file_chunks} files, {func_chunks} functions/classes)")

    print(f"\nIngested: {total_codebases} codebases, {total_entries} chunks")


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

    md_parser = sub.add_parser("markdown", help="Ingest CLAUDE.md + plans/*.md from source directories")
    md_parser.add_argument("--embed", action="store_true", help="Mint embeddings per section (slower)")
    md_parser.add_argument("--dry-run", action="store_true", help="Show what would be ingested")
    md_parser.add_argument("--force", action="store_true", help="Re-ingest already-ingested files")
    md_parser.add_argument("--sources", nargs="+", help="Source directories (default: ~/projects/)")

    code_parser = sub.add_parser("codebase", help="Ingest source code files from codebase directories")
    code_parser.add_argument("paths", nargs="*", help="Codebase directories (default: discover via px)")
    code_parser.add_argument("--embed", action="store_true", help="Mint embeddings per chunk (slower)")
    code_parser.add_argument("--dry-run", action="store_true", help="Show what would be ingested")
    code_parser.add_argument("--force", action="store_true", help="Re-ingest already-ingested files")

    args = parser.parse_args()

    if args.command == "claude-code":
        ingest_claude_code(args)
    elif args.command == "file":
        ingest_file(args)
    elif args.command == "markdown":
        ingest_markdown(args)
    elif args.command == "codebase":
        ingest_codebase(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
