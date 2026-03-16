# Continuum

Dynamic context orchestration for AI sessions with infinite continuity.

## What it does

Continuum sits between you and the Claude API. Each turn, it assembles a context window from:

- **Layer 0** — Identity core (`identity.md`). Stable, never compressed.
- **Layer 1** — System prompt.
- **Layer 2** — Dynamic context retrieved from configured sources (CLAUDE.md files, plans, session logs).
- **Layer 3a** — Compressed older session history (summaries with provenance refs back to full log).
- **Layer 3b** — Uncompressed recent session tail.
- **Layer 4** — Current user turn.

**You never hit 100% context.** Older history compresses continuously. The full session log is preserved on disk as ground truth. The session runs forever.

## Key properties

- **Infinite continuity** — no context wall, no compaction cliff. Older turns fade gracefully.
- **Provenance** — every entry gets a `«xxxx»` UID at write time. Compressed summaries carry `ref_uids` back to source.
- **Semantic embeddings** — optional per-turn embedding via Ollama for proximity search across the entire corpus.
- **Pluggable compression** — swap policies (fixed tail, token budget, logarithmic) without changing code.
- **Context switches** — switch topics A → B → A; A's continuity is preserved.
- **Anti-pattern detection** — compressed blocks carry success/failure polarity. Known mistakes surface proactively.

## Quick start

```bash
# Clone
git clone https://github.com/symbolic01/continuum.git
cd continuum

# Dependencies
pip install pyyaml numpy anthropic  # numpy for embeddings, anthropic optional

# Configure
cp continuum.yaml.example continuum.yaml
# Edit continuum.yaml: set your context sources, model, budgets

# Run
python cli.py --session my-session

# Ingest existing Claude Code sessions into the corpus
python ingest.py claude-code          # fast, no embeddings
python ingest.py claude-code --embed  # with semantic embeddings (slower)
```

## Files

```
continuum/
├── identity.md          # Layer 0: who Symbolic is (never compressed)
├── continuum.py         # Context assembler + session loop
├── compression.py       # Pluggable compression policies
├── retrieval.py         # Context retrieval from configured sources
├── embeddings.py        # Semantic embedding via Ollama + numpy index
├── session_log.py       # Append-only JSONL with UID + embedding minting
├── ingest.py            # Convert Claude Code session logs to corpus
├── tokens.py            # Token counting
├── config.py            # YAML config loader
├── cli.py               # Interactive REPL
└── continuum.yaml.example
```

## Vision

Continuum is a fragment of [Superempathy](https://independent.academia.edu/JayaramaMarks) — the thesis that superintelligence without proportional empathy is incomplete.

Current AI sessions are amnesiac by design. Continuum demonstrates a continuity architecture where context fades gracefully rather than dying abruptly — like human memory. Every insight is traceable to its source via provenance addresses. The system builds its own memory as a byproduct of managing context pressure.

A system that can trace every insight to its source data is interpretable by construction — not as a post-hoc explanation, but as a structural property of how it thinks.

## License

MIT
