"""Corpus index builder — builds EmbeddingIndex from ingested corpus."""

import json
import os
import re
from pathlib import Path

from embeddings import EmbeddingIndex


DEFAULT_CORPUS_DIR = Path.home() / ".continuum" / "corpus"
DEFAULT_INDEX_PATH = Path.home() / ".continuum" / "index" / "corpus"
DEFAULT_IDENTIFIERS_PATH = Path.home() / ".continuum" / "index" / "identifiers.json"

# Patterns for extracting identifiers from corpus content
_FILE_PATH_RE = re.compile(r'(?:~/|/home/|/tmp/|\./)[\w./-]+\.\w{1,5}')  # ~/foo/bar.py, /home/.../file.js
_RELATIVE_FILE_RE = re.compile(r'(?<![a-zA-Z])[a-zA-Z][\w-]*(?:/[\w.-]+)+\.\w{1,5}')  # bridge/webui_server.py
_BARE_FILE_RE = re.compile(r'\b[\w-]+\.(?:py|js|ts|md|json|yaml|yml|sh|html|css|toml|cfg)\b')  # file.py
_FUNC_DEF_RE = re.compile(r'(?:def |self\.|\.)\b([a-z_]\w{2,})\b')  # def foo_bar, self.foo_bar
_CLASS_DEF_RE = re.compile(r'class\s+([A-Z]\w{2,})')  # class FooBar
_SNAKE_IDENT_RE = re.compile(r'(?<![a-zA-Z])([a-z_][a-z0-9]*(?:_[a-z0-9]+){1,})(?![a-zA-Z])')  # snake_case_name (2+ segments)


def build_index(
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    index_path: Path = DEFAULT_INDEX_PATH,
    force: bool = False,
) -> EmbeddingIndex:
    """Build or load the corpus embedding index.

    Scans all JSONL files in corpus_dir, extracts entries with embeddings,
    and builds an EmbeddingIndex. Saves to disk for fast reload.
    """
    idx = EmbeddingIndex(index_path)

    # If index exists and not forcing rebuild, just load it
    if len(idx) > 0 and not force:
        return idx

    # Scan corpus — fresh index (don't load existing when forcing rebuild)
    idx = EmbeddingIndex(None)
    idx.path = index_path
    corpus_files = sorted(corpus_dir.rglob("*.jsonl"))

    total = 0
    skipped = 0

    for cf in corpus_files:
        with open(cf) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                embedding = entry.get("embedding")
                if not embedding:
                    skipped += 1
                    continue

                meta = {
                    "uid": entry.get("uid", ""),
                    "role": entry.get("role", ""),
                    "content": entry.get("content", "")[:500],  # truncate for metadata
                    "turn": entry.get("turn", 0),
                    "ts": entry.get("ts", ""),
                    "thread": entry.get("thread", ""),
                    "source_session": entry.get("source_session", ""),
                    "source_file": str(cf),
                }

                idx.add(embedding, meta)
                total += 1

    idx.save()
    print(f"Index built: {total} entries ({skipped} skipped, no embedding)")

    # Also rebuild identifiers index (piggybacks on same corpus scan)
    build_identifiers(corpus_dir)

    return idx


def _is_noise(s: str) -> bool:
    """Filter out noisy identifiers (UUIDs, timestamps, dispatch hashes)."""
    # UUID-like patterns
    if re.search(r'[0-9a-f]{8}-[0-9a-f]{4}', s):
        return True
    # Timestamp-heavy paths
    if re.search(r'\d{4}-\d{2}-\d{2}T\d{2}', s):
        return True
    # Hash-only filenames (dispatch decisions)
    basename = os.path.basename(s)
    if re.match(r'^[0-9a-f]{10,}\.', basename):
        return True
    # Relative noise (../ prefix)
    if s.startswith('./../') or s.startswith('./..'):
        return True
    # Temp/screenshot paths
    if '/tmp/' in s or '/screenshots/' in s:
        return True
    # Very long paths (likely log output, not identifiers)
    if len(s) > 120:
        return True
    return False


def _extract_identifiers(text: str) -> set[str]:
    """Extract file paths, function names, and class names from text."""
    ids = set()
    for m in _FILE_PATH_RE.finditer(text):
        val = m.group()
        if not _is_noise(val):
            ids.add(val)
    for m in _RELATIVE_FILE_RE.finditer(text):
        val = m.group()
        if not _is_noise(val):
            ids.add(val)
    for m in _BARE_FILE_RE.finditer(text):
        val = m.group()
        # Strip leading non-alpha (from \n, punctuation, etc.)
        val = re.sub(r'^[^a-zA-Z_]+', '', val)
        if val and not _is_noise(val):
            ids.add(val)
    for m in _FUNC_DEF_RE.finditer(text):
        ids.add(m.group(1))
    for m in _CLASS_DEF_RE.finditer(text):
        ids.add(m.group(1))
    for m in _SNAKE_IDENT_RE.finditer(text):
        val = m.group(1)
        if len(val) >= 6:  # skip very short ones like "is_ok"
            ids.add(val)

    # Final cleanup: remove any identifiers with leading non-alpha (from regex boundary issues)
    cleaned = set()
    for i in ids:
        stripped = re.sub(r'^[^a-zA-Z_~/./]+', '', i)
        if stripped and len(stripped) >= 3:
            cleaned.add(stripped)
    return cleaned


def build_identifiers(
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    identifiers_path: Path = DEFAULT_IDENTIFIERS_PATH,
) -> list[str]:
    """Scan corpus and extract all identifiers (file paths, function names, etc.).

    Writes to identifiers.json. No embeddings needed — pure text extraction.
    """
    all_ids: set[str] = set()
    corpus_files = sorted(corpus_dir.rglob("*.jsonl"))

    for cf in corpus_files:
        with open(cf) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                content = entry.get("content", "")
                all_ids.update(_extract_identifiers(content))

    # Sort for stable output
    sorted_ids = sorted(all_ids)

    identifiers_path.parent.mkdir(parents=True, exist_ok=True)
    with open(identifiers_path, "w") as f:
        json.dump(sorted_ids, f, indent=None)

    print(f"Identifiers index: {len(sorted_ids)} unique identifiers")
    return sorted_ids


def load_identifiers(identifiers_path: Path = DEFAULT_IDENTIFIERS_PATH) -> list[str]:
    """Load known identifiers from disk."""
    if identifiers_path.is_file():
        with open(identifiers_path) as f:
            return json.load(f)
    return []


def load_index(index_path: Path = DEFAULT_INDEX_PATH) -> EmbeddingIndex:
    """Load existing index from disk. Returns empty index if not found."""
    return EmbeddingIndex(index_path)
