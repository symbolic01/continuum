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

Find the continuum directory. Check these locations in order:
1. `~/continuum/`
2. `~/+/continuum/`
3. The directory containing this skill file (walk up from SKILL.md)

Then run:

```bash
python3 <continuum_dir>/ingest_all.py
```

If `cx` is on PATH:

```bash
cx ingest
cx ingest --no-embed --codebases ~/work/my-service
```

## What It Does

1. **CC sessions**: scans `~/.claude/projects/` for session JSONL files. Incremental — skips sessions already ingested unless source is newer (session grew). `--force` re-ingests all.
2. **Markdown**: scans `~/projects/` (or `--sources`) for CLAUDE.md and plans/*.md. Always re-ingests (files change in place).
3. **Codebases**: auto-discovers linked codebases or uses `--codebases`. Chunks Python by class/method, JS/TS by function/class, Java by class/method, Go by type/func, JSON by top-level key.
4. **Index rebuild**: builds embedding index (`corpus.npz`), full metadata index (`all_metadata.json`), and identifiers index (`identifiers.json`).
5. **Git checkpoint**: auto-commits `~/.continuum/` so expensive index files are recoverable.

## When to Use

- First time setup: `/ingest --no-embed` to get started fast
- After pulling new code at work: `/ingest --codebases ~/work/repo --force`
- After long sessions: `/ingest` to pick up new session data
- After editing CLAUDE.md or plans: `/ingest` (markdown always re-ingests)
