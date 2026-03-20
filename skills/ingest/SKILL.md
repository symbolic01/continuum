---
name: ingest
description: Ingest Claude Code sessions, markdown docs, and source code into the Continuum corpus. Rebuilds search indexes and git-commits the data directory.
allowed-tools: Bash, Read
user-invocable: true
---

# Ingest Skill

Build or update the Continuum corpus from all available sources.

## Arguments

- `/ingest` — ingest everything (CC sessions + markdown + codebases + index rebuild)
- `/ingest --no-embed` — skip semantic embeddings (faster, keyword search still works)
- `/ingest --force` — re-ingest all files, not just new/changed ones
- `/ingest --no-code` — skip codebase ingestion
- `/ingest --identifiers-only` — just rebuild the identifiers index (instant)
- `/ingest --sources ~/dir1 ~/dir2` — add extra markdown source directories
- `/ingest --codebases ~/repo1 ~/repo2` — specify codebase directories

## How to Run

Use `cx` if on PATH, otherwise find and run `ingest_all.py` directly:

```bash
# Option 1: cx on PATH
cx ingest

# Option 2: read install path from breadcrumb, then run directly
CONTINUUM_DIR="$(cat ~/.continuum/.install_path 2>/dev/null)"
python3 "$CONTINUUM_DIR/ingest_all.py"
```

IMPORTANT: Always try `cx ingest` first. If that fails, use the breadcrumb file. Never hardcode a continuum path.

## What It Does

1. **CC sessions**: scans `~/.claude/projects/` for session JSONL files (incremental)
2. **Markdown**: scans `~/projects/` for CLAUDE.md and plans/*.md (always re-ingests)
3. **Codebases**: auto-discovers linked codebases, chunks by file/class/method
4. **Index rebuild**: builds embedding index, full metadata index, and identifiers index
5. **Git checkpoint**: auto-commits `~/.continuum/` so index files are recoverable

## When to Use

- First time setup: `/ingest --no-embed`
- After pulling new code: `/ingest --codebases ~/work/repo --force`
- After long sessions: `/ingest` to pick up new session data
- After editing CLAUDE.md or plans: `/ingest`
