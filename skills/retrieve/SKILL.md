---
name: retrieve
description: Search past sessions, docs, and code via Continuum's multi-axis retrieval (semantic + keyword + identifier + temporal). Returns relevant context chunks.
allowed-tools: Bash, Read
user-invocable: true
---

# Retrieve Skill

Search the Continuum corpus for relevant context.

## Arguments

- `/retrieve <query>` — search with LLM cull (default)
- `/retrieve --no-cull <query>` — skip LLM precision filtering (faster, noisier)
- `/retrieve --budget N <query>` — limit output to N tokens (default: 30000)
- `/retrieve --code <query>` — only return code chunks (skip sessions/markdown)
- `/retrieve --project bridge <query>` — filter to a specific project/codebase
- `/retrieve --code --project bridge <query>` — combine: code from a specific project

## How to Run

Use `cx` if on PATH, otherwise find and run `retrieve_tool.py` directly:

```bash
# Option 1: cx on PATH
cx retrieve "your query here"

# Option 2: read install path from breadcrumb, then run directly
CONTINUUM_DIR="$(cat ~/.continuum/.install_path 2>/dev/null)"
python3 "$CONTINUUM_DIR/retrieve_tool.py" "your query here"
```

IMPORTANT: Always try `cx retrieve` first. If that fails, use the breadcrumb file. Never hardcode a continuum path.

## Output

The tool prints retrieved context to stdout — show this to the user. Diagnostic info goes to stderr.

## When to Use

- User asks about something from a previous session
- User references code, files, or concepts that aren't in the current context
- You need background on a project or decision that was made earlier
- User says "what did we do about X" or "remember when we fixed Y"
