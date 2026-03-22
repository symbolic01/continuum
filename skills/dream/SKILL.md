---
name: dream
description: Run offline integration pipeline — finds connections between corpus chunks across time and projects, synthesizes human-meaningful kernels (corrections, unfinished work, patterns, pain points).
allowed-tools: Bash, Read
user-invocable: true
---

# Dream Skill

Run the offline integration + synthesis pipeline on the continuum corpus.

## Arguments

- `/dream` — full pipeline (ingest, integrate, synthesize, report)
- `/dream --dry-run` — show what would happen, no writes
- `/dream --report` — print markdown report to stdout
- `/dream --max-time N` — wall time cap in seconds (default: 1800)
- `/dream --force` — run even if corpus hasn't changed
- `/dream --no-synthesis` — skip Claude synthesis (just integration)
- `/dream --no-temporal` — skip temporal reconnection

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

## Pipeline

1. **Ingest housekeeping** — pick up new sessions, docs, code
2. **Integration** (Ollama, local) — self-reinforcing cluster analysis. Multi-axis similarity (semantic + temporal + project + keyword). New chains become preferred seeds
3. **Temporal reconnection** — cross-temporal links (things weeks apart but related)
4. **Synthesis** (Claude Sonnet) — compress chains into kernels: corrections, orphans, patterns, stress points, growth, open questions
5. **Report** — HTML drill-down + markdown summary

## Output

- **Kernel chunks** written to corpus (role: "kernel") — retrievable via `cx retrieve`
- **Chain chunks** at `~/.continuum/corpus/_chains/`
- **Cross-references** at `~/.continuum/index/xrefs.json`
- **Report** at `~/.continuum/dream_report.json` + HTML viewer
- **Markdown** to stdout with `--report`

## When to Use

- User asks to find connections, patterns, or unfinished work across sessions
- User wants to surface anti-patterns or recurring corrections
- To get a high-level view of what's been happening across projects
- Runs automatically via `cx daemon` when sessions are idle

## Daemon

Auto-trigger dreaming when idle:

```bash
cx daemon --idle 15 --dream 90 --gap 120   # background
cx daemon --once                            # single check (systemd/cron)
```

Wakes up gracefully if a claude process starts.
