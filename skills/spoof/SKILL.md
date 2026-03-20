---
name: spoof
description: Compress a messy Claude Code session into a clean resumable one. LLM distills dead ends into narrative, injects identity and retrieved context, produces a valid CC session for claude --resume.
allowed-tools: Bash, Read
user-invocable: true
---

# Spoof Skill

Compress the current session into a clean, resumable Claude Code session.

## Arguments

- `/spoof` — spoof with raw tail (no compression)
- `/spoof --compress` — LLM-compress the bulk into a clean narrative
- `/spoof --compress --prompt "focus on X"` — steer what the compression emphasizes
- `/spoof --no-ingest` — skip auto-ingest check (faster)

## How to Run

Use `cx` if on PATH, otherwise find and run `spoof_tool.py` directly:

```bash
# Option 1: cx on PATH
cx spoof --compress

# Option 2: find spoof_tool.py
CONTINUUM_DIR="$([ -d ~/+/continuum ] && echo ~/+/continuum || [ -d ~/continuum ] && echo ~/continuum || echo "")"
python3 "$CONTINUUM_DIR/spoof_tool.py" --compress
```

IMPORTANT: Always try `cx spoof` first. If that fails, use the `CONTINUUM_DIR` resolution above. Never hardcode a path without checking it exists.

The tool prints the spoofed session ID to stdout and the resume command to stderr. Resume with `cx` or `claude --resume <id>`.

## What It Does

1. Reads the most recent CC session for the current working directory
2. If `--compress`: sends the bulk to an LLM to distill into 5-20 clean narrative turns
3. Keeps the recent tail verbatim (~3000 chars)
4. Injects `identity.md` as a first-person assistant turn
5. Retrieves relevant corpus context and injects as recall blocks
6. Writes a valid CC session JSONL to `~/.claude/projects/`

## When to Use

- Session is getting long and messy with dead ends
- Context is filling up and you need to start fresh without losing history
- User says "let's clean this up" or "compress this session"
