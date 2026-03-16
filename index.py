"""Corpus index builder — builds EmbeddingIndex from ingested corpus."""

import json
from pathlib import Path

from embeddings import EmbeddingIndex


DEFAULT_CORPUS_DIR = Path.home() / ".continuum" / "corpus"
DEFAULT_INDEX_PATH = Path.home() / ".continuum" / "index" / "corpus"


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

    # Scan corpus
    idx = EmbeddingIndex(index_path)
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
    return idx


def load_index(index_path: Path = DEFAULT_INDEX_PATH) -> EmbeddingIndex:
    """Load existing index from disk. Returns empty index if not found."""
    return EmbeddingIndex(index_path)
