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

## How to Run

Find the continuum directory. Check these locations in order:
1. `~/continuum/`
2. `~/+/continuum/`
3. The directory containing this skill file (walk up from SKILL.md)

Then run:

```bash
python3 <continuum_dir>/retrieve_tool.py <args>
```

Example:

```bash
python3 ~/continuum/retrieve_tool.py "auth token refresh handler"
python3 ~/continuum/retrieve_tool.py --no-cull --budget 10000 "PTY resize bug"
```

If `cx` is on PATH, you can also use:

```bash
cx retrieve "auth token refresh handler"
```

## Output

The tool prints retrieved context to stdout — show this to the user. Diagnostic info goes to stderr.

## When to Use

- User asks about something from a previous session
- User references code, files, or concepts that aren't in the current context
- You need background on a project or decision that was made earlier
- User says "what did we do about X" or "remember when we fixed Y"
