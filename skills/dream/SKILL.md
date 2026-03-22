---
name: dream
description: Run offline integration pipeline — finds connections between corpus chunks across time and projects, surfaces anti-patterns, unfinished work, and cross-project insights.
allowed-tools: Bash, Read
user-invocable: true
---

# Dream Skill

Run the offline integration pipeline on the continuum corpus.

## Arguments

- `/dream` — full pipeline (ingest, integrate, reconnect, report)
- `/dream --dry-run` — show what would happen, no writes
- `/dream --report` — emit markdown report to stdout
- `/dream --max-passes N` — cap integration passes (default: 50)
- `/dream --max-time N` — wall time cap in seconds (default: 1800)
- `/dream --force` — run even if corpus hasn't changed

## How to Run

Use `cx` if on PATH, otherwise find and run `dream_tool.py` directly:

```bash
# Option 1: cx on PATH
cx dream --report

# Option 2: read install path from breadcrumb, then run directly
CONTINUUM_DIR="$(cat ~/.continuum/.install_path 2>/dev/null)"
python3 "$CONTINUUM_DIR/dream_tool.py" --report
```

IMPORTANT: Always try `cx dream` first. If that fails, use the breadcrumb file. Never hardcode a continuum path.

## Output

- Chain chunks written to `~/.continuum/corpus/_chains/`
- Cross-reference index at `~/.continuum/index/xrefs.json`
- Report data at `~/.continuum/dream_report.json`
- Markdown report printed to stdout with `--report`

## When to Use

- User asks to run the dream pipeline
- User wants to find connections across sessions or projects
- User wants to surface unfinished work or anti-patterns
- During idle time (can be triggered via cron)
