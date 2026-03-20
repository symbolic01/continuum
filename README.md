# Continuum

Cross-session memory for Claude Code. Ingest your past sessions, retrieve what matters, and resume with a clean compressed history instead of starting from scratch.

## Quick Start (5 minutes)

### 1. Clone and install

```bash
git clone https://github.com/symbolic01/continuum.git
cd continuum
pip install -r requirements.txt   # numpy + pyyaml (both optional, see below)
```

No dependencies are required for basic `cx spoof` / `cx retrieve` / `cx ingest` usage. `numpy` enables semantic embeddings. `pyyaml` enables `continuum.yaml` config (env vars work without it).

### 2. Configure

```bash
cp continuum.yaml.example continuum.yaml
```

Edit `continuum.yaml` — the defaults work, but point `context_sources` at your project docs:

```yaml
# Accepts files, directories (recurses *.md), and globs
context_sources:
  - ~/my-project/                        # whole directory tree
  - ~/my-project/docs/architecture.md    # single file
  - ~/work/plans/*.md                    # glob pattern
```

### 3. Register install path

```bash
mkdir -p ~/.continuum
echo "$(pwd)" > ~/.continuum/.install_path
```

This lets skills and tools find continuum regardless of where you cloned it.

### 4. Build the corpus

```bash
python ingest_all.py --no-embed
```

This ingests **everything it can find**:
- All your Claude Code session logs from `~/.claude/projects/`
- CLAUDE.md and plans/*.md files from `~/projects/`
- Source code from linked codebases (auto-discovered, chunked by file/class/method)
- Rebuilds the search index + identifiers index

First run takes 10-30 seconds. Use `--no-embed` to skip semantic embeddings (fast keyword + identifier search still works). Add `--force` to re-ingest previously processed files.

### 5. Set up the `cx` command

```bash
mkdir -p ~/bin
cp bin/cx ~/bin/cx    # or: ln -s "$(pwd)/bin/cx" ~/bin/cx
chmod +x ~/bin/cx
```

Make sure `~/bin` is on your PATH. If not, add to your `.bashrc` / `.zshrc`:
```bash
export PATH="$HOME/bin:$PATH"
```

### 6. Install Claude Code skills (optional)

This gives you `/retrieve`, `/spoof`, and `/ingest` slash commands inside any Claude Code session:

```bash
mkdir -p ~/.claude/skills
ln -s "$(pwd)/skills/retrieve" ~/.claude/skills/retrieve
ln -s "$(pwd)/skills/spoof" ~/.claude/skills/spoof
ln -s "$(pwd)/skills/ingest" ~/.claude/skills/ingest
```

### 7. Try it

```bash
# Search your past sessions
cx retrieve "that auth token bug we fixed"

# Compress your current messy session into a clean one
cx spoof --compress

# Resume the compressed session
cx
```

That's it. You now have cross-session memory.

## Prerequisites

- **Python 3.10+** with `pyyaml` (`pip install pyyaml`)
- **Claude CLI** (`claude`) on PATH — needed for `--print` (compression, culling) and `--resume` (spoofing)
- **Ollama** (optional) — for semantic embeddings and query decomposition. Without it, keyword + identifier search still works

### Local models (Ollama)

```bash
# Required for semantic embeddings (cx ingest without --no-embed)
ollama pull nomic-embed-text       # 768-dim, fast, ~270MB

# Required for query decomposition (keyword expansion, identifier extraction)
ollama pull qwen2.5:7b             # ~4.4GB
```

Without Ollama, retrieval falls back to raw keyword search (no decomposition, no embeddings). Everything else still works.

### Model configuration

Continuum uses three LLM roles. Each is configurable via env var or `continuum.yaml`:

| Role | What it does | Default | Env var |
|------|-------------|---------|---------|
| **compress** | Session narrative compression (`cx spoof --compress`) | `claude-sonnet-4-6` | `CONTINUUM_COMPRESS_MODEL` |
| **cull** | Retrieval precision filtering | `claude-haiku-4-5-20251001` | `CONTINUUM_CULL_MODEL` |
| **decompose** | Query decomposition + keyword expansion (Ollama) | `qwen2.5:7b` | `CONTINUUM_DECOMPOSE_MODEL` |

Compress and cull run via `claude --print` — they use whatever model names your Claude CLI understands. Decompose runs via Ollama's local API.

**At work with LiteLLM / Bedrock / custom model names:**

The model name is passed directly to `claude --print --model <name>`. It must match exactly what your proxy expects — not a made-up shorthand.

```bash
# In your .bashrc / .zshrc:
export CONTINUUM_COMPRESS_MODEL="bedrock/us.anthropic.claude-sonnet-4-6-v1"
export CONTINUUM_CULL_MODEL="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1"
# decompose still uses local Ollama — no change needed
```

Or in `continuum.yaml`:

```yaml
compress_model: bedrock/us.anthropic.claude-sonnet-4-6-v1
cull_model: bedrock/us.anthropic.claude-haiku-4-5-20251001-v1
decompose_model: qwen2.5:7b
```

Env vars take precedence over config file. If compression silently fails (90→90 turns with no reduction), the model name is probably wrong — check `claude --print --model <your-model> "hello"` works on its own first.

## What each command does

### `cx ingest` — build the corpus

Scans your Claude Code session logs, project markdown, and source code. Converts everything into a searchable corpus at `~/.continuum/corpus/`.

```bash
cx ingest                          # ingest everything, rebuild index
cx ingest --force                  # re-ingest all (not just new files)
cx ingest --no-embed               # skip semantic embeddings (faster)
cx ingest --sources ~/work ~/oss   # add extra markdown source directories
cx ingest --identifiers-only       # just rebuild the identifiers index (instant)
cx ingest --no-code                # skip codebase ingestion
```

What gets ingested:
- **CC sessions**: every `*.jsonl` in `~/.claude/projects/` — your full conversation history
- **Markdown**: CLAUDE.md files and `plans/*.md` from `~/projects/` (or `--sources`)
- **Codebases**: Python chunked by class/method, JS/TS by function/class, others by file. Auto-discovered via project links or pass paths explicitly

### `cx retrieve` — search the corpus

Finds relevant context from your past sessions, docs, and code.

```bash
cx retrieve "PTY resize SIGWINCH"                # keyword + semantic + LLM-filtered
cx retrieve --budget 10000 "auth token refresh"  # limit token output
cx retrieve --no-cull "session spoofing"         # skip LLM filtering (faster, noisier)
```

The retrieval pipeline:
1. **Query decomposition** — LLM splits your query into search axes (semantic, temporal, project, entity, etc.) and expands keywords
2. **Identifier resolution** — approximate file/function names fuzzy-matched against a known identifiers index (`webserverui` → `webui_server.py`)
3. **Multi-axis search** — semantic similarity + keyword matching + identifier matching + temporal decay
4. **LLM cull** (on by default) — over-retrieves 5x, asks a fast LLM which chunks actually matter. Skip with `--no-cull`

### `cx spoof` — compress and resume sessions

Takes a messy session (200+ turns of dead ends) and produces a clean, resumable one.

```bash
cx spoof --compress                        # compress current session
cx spoof --compress --prompt "focus on X"  # steer what the compression emphasizes
cx                                         # resume the spoofed session
```

What happens:
1. Reads your current CC session (most recent for the working directory)
2. LLM compresses the bulk into a clean "tried X / failed Y / solution was Z" narrative (5-20 turns)
3. Keeps the recent tail verbatim (last ~3000 chars)
4. Injects your `identity.md` as a first-person assistant turn + system prompt
5. Retrieves relevant corpus context and injects it as recall blocks
6. Writes a valid CC session JSONL that `claude --resume` picks up natively

### `cx` (no args) — resume last spoof

Runs `claude --resume <last-spoofed-session-id>` with identity in `--append-system-prompt`.

## Customization

### Identity (`identity.md`)

Edit `identity.md` to define who the AI is in first person. This gets injected into every spoofed session — both as a visible assistant turn and in the system prompt. The default is Symbolic's identity; replace it with your own or delete it to skip identity injection.

### Config (`continuum.yaml`)

```yaml
model: claude-sonnet-4-6              # model for interactive mode
backend: cli                           # "cli" (claude --print) or "api" (Anthropic SDK)

context_sources:                       # files, directories, or globs
  - ~/my-project/

token_budgets:
  total: 180000                        # overall context budget
  dynamic_context: 30000               # retrieved context budget
  recent_tail: 80000                   # uncompressed recent conversation

compression:
  policy: token_budget
  model: claude-haiku-4-5-20251001     # fast model for compression

session:
  log_dir: ~/.continuum/sessions       # where session logs live
```

### Adding codebases

Codebases are auto-discovered if you use `px` (the project CLI). Otherwise, pass them explicitly:

```bash
cx ingest --codebases ~/my-project ~/other-project
# or
python ingest.py codebase ~/my-project ~/other-project --force
```

Source files are chunked by language:
- **Python**: file overview + individual class and function/method definitions
- **JS/TS**: file overview + function and class definitions
- **Java**: file overview + class/interface/enum + individual method definitions (brace-matched)
- **Go**: file overview + type (struct/interface) + func definitions (brace-matched)
- **JSON**: small files as-is; large files split by top-level key (great for configs, JSON-e templates)
- **Everything else** (`.sh`, `.html`, `.css`, `.yaml`, etc.): file-level chunk (first 3K chars)

Supported extensions: `.py`, `.js`, `.ts`, `.jsx`, `.tsx`, `.sh`, `.html`, `.css`, `.json`, `.yaml`, `.yml`, `.toml`, `.go`, `.rs`, `.java`, `.c`, `.cpp`, `.h`

## Work setup: ingest a codebase you don't own

At work you often have large repos you want searchable but don't control. Here's the quickest path:

```bash
# 1. Ingest just the codebase (no CC sessions, no markdown, no embeddings)
python3 ~/continuum/ingest.py codebase ~/work/my-service ~/work/shared-libs --force

# 2. Rebuild the index (picks up the new corpus files)
python3 ~/continuum/ingest_all.py --identifiers-only   # instant — just identifiers
# or full rebuild:
python3 ~/continuum/ingest_all.py --no-code --no-embed  # skips re-ingesting code, rebuilds index

# 3. Search it
cx retrieve "TokenRefreshHandler credential rotation"
```

Step 1 is the targeted command — it only touches the codebases you point it at. No session logs, no markdown, no embeddings. The `--force` flag re-ingests even if a previous run exists (useful after `git pull`).

You can also add codebases to the default `cx ingest` run:

```bash
# These paths get ingested alongside CC sessions + markdown
cx ingest --codebases ~/work/my-service ~/work/shared-libs
```

Or skip codebases entirely in the default run:

```bash
cx ingest --no-code
```

**Re-ingesting after code changes**: run the same `ingest.py codebase ... --force` command again. It overwrites the previous corpus file for that project. Then `cx ingest --identifiers-only` to update the identifiers index.

## How it works under the hood

### Session spoofing

Claude Code validates session JSONL strictly. Entries built from scratch get silently dropped on resume. Continuum solves this by:

1. **Template cloning** — deepcopy real CC entries from an existing session, swap in new content. Preserves the exact field set and key order CC expects.
2. **Timestamp format** — CC requires `YYYY-MM-DDTHH:MM:SS.mmmZ` (Z suffix, millisecond precision, monotonically increasing). Python's `isoformat()` produces `+00:00` with microseconds — both cause silent drops.
3. **Consecutive same-role entries** — CC renders each as its own bullet. No merging needed.
4. **Dual identity positioning** — identity in both a first-person assistant turn (visible) and `--append-system-prompt` (strongest positional influence).

### Identifier resolution

The corpus index includes a **known identifiers index** (~10K entries) extracted from all ingested content — file paths, function names, class names. When you search for an approximate identifier:

- `webserverui` → `webui_server.py` (character similarity)
- `spoof_session` → `session_spoof` (token rearrangement: same tokens, different order)
- `replace_session` → `_replace_session` (substring containment)

Resolution uses `difflib.SequenceMatcher` (stdlib), no external dependencies.

### Retrieval pipeline

```
User query: "that webserverui file with the PTY resize bug"
     │
     ▼
  Query decomposition (Ollama/Qwen)
  → axes: [semantic, project:bridge]
  → keywords: [PTY, TIOCSWINSZ, SIGWINCH, terminal, resize]
  → identifiers: [webserverui]
     │
     ▼
  Fuzzy identifier resolution
  → webserverui → webui_server.py
     │
     ▼
  Multi-axis search (parallel):
  • semantic embedding similarity
  • expanded keyword matching
  • graduated identifier matching
  • temporal/project/entity axes
     │
     ▼
  Score + rank + temporal decay + budget fit
  (LLM precision cull — default on, --no-cull to skip)
```

## Data locations

Everything Continuum produces lives under `~/.continuum/`. The embedding index (`corpus.npz`) is the most expensive to rebuild — it takes hours for large corpora. **Do not delete it.** If it gets corrupted, re-embed with `cx ingest --force`.

### `~/.continuum/` — Continuum's data directory

```
~/.continuum/
├── corpus/                     # Ingested content (JSONL per source)
│   ├── home/                   #   CC sessions by project
│   ├── bridge/
│   ├── _markdown/              #   Markdown sections
│   └── _codebase/              #   Source code chunks
│
├── index/                      # Search indexes (rebuilt by cx ingest)
│   ├── corpus.npz              #   ⚠️ EXPENSIVE: embedding vectors (hours to rebuild)
│   ├── corpus.meta.json        #   Metadata for embedded entries only
│   ├── all_metadata.json       #   Metadata for ALL entries (keyword/identifier search)
│   └── identifiers.json        #   Known file paths, function names, class names
│
├── sessions/                   # Interactive mode session logs
├── .last_spoof                 # Session ID of last cx spoof
└── .last_identity              # Identity text for --append-system-prompt
```

### `~/.claude/projects/` — Claude Code's data (not ours)

Spoofed sessions are written here so `claude --resume` finds them:
```
~/.claude/projects/<mangled-cwd>/<spoofed-uuid>.jsonl
```

### In the repo

```
continuum.yaml                  # Your local config (git-ignored)
```

## Troubleshooting

**`cx: command not found`**
- Make sure `~/bin/cx` exists and `~/bin` is on your PATH
- Run `echo $PATH` to check, add `export PATH="$HOME/bin:$PATH"` to your shell rc

**`cx spoof` says "no CC session found"**
- You need at least one existing Claude Code session in the current directory
- Check `ls ~/.claude/projects/` — you should see directories with `.jsonl` files

**`cx retrieve` returns nothing**
- Run `cx ingest` first to build the corpus
- Check that `~/.continuum/corpus/` has `.jsonl` files

**Semantic search not working (only keyword results)**
- Install Ollama and pull the embedding model: `ollama pull nomic-embed-text`
- Re-ingest with embeddings: `cx ingest --force` (without `--no-embed`)

**`cx spoof --compress` hangs or fails**
- Compression calls `claude --print --model claude-sonnet-4-6` — make sure `claude` CLI works
- Check your API key / Claude subscription

**Identifiers not resolving**
- Run `cx ingest --identifiers-only` to rebuild the identifiers index
- Check `~/.continuum/index/identifiers.json` exists and has entries

## Files

```
continuum/
├── bin/cx                     # Shell wrapper: cx spoof, cx retrieve, cx ingest, cx (resume)
├── spoof_tool.py              # CLI: session spoofing (--compress, --prompt, --context)
├── retrieve_tool.py           # CLI: corpus retrieval (query, --budget, --no-cull)
├── ingest_all.py              # CLI: one-shot ingest (CC sessions + markdown + codebases + index)
├── identity.md                # Identity core — first-person, injected into spoofed sessions
├── system.md                  # System prompt for interactive mode
├── continuum.yaml.example     # Config template — copy to continuum.yaml
│
├── core/                      # Library modules
│   ├── ingest.py              # Convert CC logs, markdown, and source code to corpus
│   ├── retrieval.py           # Multi-axis retrieval with fuzzy identifier resolution
│   ├── query.py               # LLM query decomposition + keyword expansion
│   ├── index.py               # Corpus index + identifiers index builder
│   ├── embeddings.py          # Ollama batch embedding API
│   ├── session_spoof.py       # Build CC-compatible JSONL from conversation turns
│   ├── session_compress.py    # LLM-powered narrative distillation
│   ├── session_log.py         # Append-only JSONL with UID minting
│   ├── compression.py         # Pluggable compression policies
│   ├── auto_ingest.py         # Stale-index detection for auto-ingest on tool use
│   ├── tokens.py              # Token counting
│   ├── config.py              # YAML config loader
│   ├── backend.py             # CLI and API backends
│   └── actions.py             # Action parsing and execution
│
└── interactive/               # Interactive mode (experimental)
    ├── continuum.py           # Per-turn context assembly + session loop
    ├── cli.py                 # Interactive REPL
    ├── orchestrate.py         # Multi-action orchestration
    └── web.py                 # Web UI
```

## Vision

Continuum is a fragment of [Superempathy](https://independent.academia.edu/JayaramaMarks) — the thesis that superintelligence without proportional empathy is incomplete.

Current AI sessions are amnesiac by design. Continuum demonstrates a continuity architecture where context fades gracefully rather than dying abruptly — like human memory. Every insight is traceable to its source via provenance UIDs.

## License

MIT
