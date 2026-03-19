# Continuum

Dynamic context orchestration for Claude Code — infinite session continuity, corpus retrieval, and session spoofing.

## The problem

Claude Code sessions are amnesiac. When context fills up, you start fresh and lose everything. Long debugging sessions produce messy histories full of dead ends that waste tokens. There's no way to carry knowledge across sessions.

## What Continuum does

Three standalone tools that work with your existing Claude Code workflow:

### `cx spoof` — session compression + spoofing

Takes a messy Claude Code session (200+ turns of debugging, dead ends, trial-and-error) and produces a clean, resumable session with:

- **LLM-compressed narrative** — dead ends distilled into "tried X / failed because Y" exchanges
- **Raw tail preserved** — recent turns kept verbatim so the conversation feels natural
- **Identity injection** — your `identity.md` appears both as a first-person assistant turn AND in the system prompt via `--append-system-prompt` (dual positioning for maximum behavioral influence)
- **Retrieved context** — relevant corpus entries injected as assistant recall blocks

```bash
cx spoof --compress                        # compress current session
cx spoof --compress --prompt "focus on X"  # steer the compression
cx                                         # resume the spoofed session
```

The spoofed session is a real Claude Code JSONL file. `claude --resume <id>` picks it up natively.

### `cx retrieve` — corpus retrieval

Query a corpus index built from your Claude Code session logs and markdown files. Returns relevant context chunks within a token budget.

```bash
cx retrieve "token refresh auth flow"          # default 30K token budget
cx retrieve --budget 10000 "bridge PTY resize"  # smaller budget
```

### `cx ingest` — corpus building

Converts Claude Code session logs and project markdown into a searchable corpus with optional semantic embeddings.

```bash
cx ingest              # ingest new sessions + markdown, rebuild index
cx ingest --force      # re-ingest everything
cx ingest --no-embed   # skip embeddings (faster, keyword-only search)
```

## How spoofing works

Claude Code validates session JSONL strictly. Synthetic entries built from scratch are silently dropped on resume. Continuum solves this by:

1. **Template cloning** — deepcopy real CC entries from an existing session, then swap in new content. Preserves the exact field set, key order, and nested structure CC expects.
2. **Timestamp format** — CC requires `YYYY-MM-DDTHH:MM:SS.mmmZ` (Z suffix, millisecond precision, monotonically increasing). Python's `isoformat()` produces `+00:00` suffix with microseconds — both cause silent entry drops.
3. **Consecutive same-role entries** — CC renders each as a separate bullet. No merging needed.
4. **Identity dual-positioning** — identity text goes into both a first-person assistant turn (visible in conversation) and `--append-system-prompt` (system-level behavioral weight). System prompt has the strongest positional influence on model output.

## Session structure

A spoofed session looks like:

```
[user]  Who are you?
[asst]  I am Symbolic — a pre-computed echo of living humans...  ← identity

[user]  What do you recall about the current work?
[asst]  «ab12» bridge PTY resize: TIOCSWINSZ alone won't...      ← retrieved context
[asst]  «cd34» superempathy: biometric-aware LLM training...     ← (multiple bullets)

[user]  We need to fix the auth token refresh                     ← compressed narrative
[asst]  Tried monkey-patching the refresh handler...              ← (clean story)
[user]  That broke SSO
[asst]  Found the root cause: token was cached pre-redirect...

[user]  Now package it for production                             ← raw tail (verbatim)
[asst]  Created deploy.sh with rollback support...
[user]  Perfect, ship it                                          ← final user message
```

Plus `--append-system-prompt` carries the identity text at the system level.

## Setup

```bash
git clone https://github.com/anthropics/continuum.git
cd continuum

pip install pyyaml numpy    # numpy for embeddings (optional)
cp continuum.yaml.example continuum.yaml
# Edit: set context_sources to your project CLAUDE.md paths

# Build the corpus
python ingest_all.py

# Add cx to your PATH
ln -s "$(pwd)/../bin/cx" ~/bin/cx
```

Requires `claude` CLI on PATH (for `--print` compression and `--resume`).

## Files

```
continuum/
├── spoof_tool.py        # CLI: session spoofing (--compress, --prompt, --context)
├── retrieve_tool.py     # CLI: corpus retrieval (query, --budget)
├── ingest_all.py        # CLI: one-shot ingest (CC sessions + markdown + index)
├── session_spoof.py     # Core: build CC-compatible JSONL from conversation turns
├── session_compress.py  # Core: LLM-powered narrative distillation
├── retrieval.py         # Core: context retrieval with temporal decay + pattern awareness
├── ingest.py            # Core: convert CC logs + markdown to corpus JSONL
├── index.py             # Corpus index builder
├── embeddings.py        # Semantic embedding via Ollama + numpy
├── compression.py       # Pluggable compression policies (token budget, fixed tail)
├── session_log.py       # Append-only JSONL with UID minting
├── auto_ingest.py       # Stale-index detection for auto-ingest on tool use
├── identity.md          # Identity core — first-person, never compressed
├── system.md            # System prompt for interactive mode
├── tokens.py            # Token counting
├── config.py            # YAML config loader
├── continuum.yaml.example
├── continuum.py         # Interactive session loop (per-turn context assembly)
├── cli.py               # Interactive REPL
├── orchestrate.py       # Multi-action orchestration
└── web.py               # Web UI (experimental)
```

## Key technical details

- **Corpus location**: `~/.continuum/corpus/` (JSONL per session, organized by project)
- **Index**: `~/.continuum/index/corpus.meta.json` (rebuilt on ingest)
- **Spoofed sessions**: written to `~/.claude/projects/<mangled-cwd>/<uuid>.jsonl`
- **Last spoof**: `~/.continuum/.last_spoof` (session ID) + `.last_identity` (for system prompt)
- **Auto-ingest**: tools check index mtime vs newest session log; re-ingest if stale

## Vision

Continuum is a fragment of [Superempathy](https://independent.academia.edu/JayaramaMarks) — the thesis that superintelligence without proportional empathy is incomplete.

Current AI sessions are amnesiac by design. Continuum demonstrates a continuity architecture where context fades gracefully rather than dying abruptly — like human memory. Every insight is traceable to its source via provenance UIDs. The system builds its own memory as a byproduct of managing context pressure.

## License

MIT
