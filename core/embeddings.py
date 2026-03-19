"""Embedding utilities for Continuum.

Embeds text via Ollama (nomic-embed-text) for semantic search and
proximity traversal. Embeddings are minted at write time alongside
UIDs — every entry in the corpus is both addressable (by UID) and
locatable (by semantic neighborhood).

Uses urllib with connection reuse and Ollama's native batch API
for 10-50x faster embedding vs serial curl subprocess calls.
"""

import json
import sys
import urllib.request
import urllib.error
import numpy as np
from pathlib import Path


# Default model — available via Ollama, 768-dim, fast
DEFAULT_MODEL = "nomic-embed-text"
OLLAMA_URL = "http://localhost:11434/api/embed"
BATCH_SIZE = 32  # texts per API call
MAX_TEXT_LEN = 8000  # truncate before embedding


def _truncate(text: str) -> str:
    return text[:MAX_TEXT_LEN] if len(text) > MAX_TEXT_LEN else text


def _post_json(payload: dict, timeout: int = 60) -> dict | None:
    """POST JSON to Ollama, return parsed response or None."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, OSError):
        return None


def embed_text(text: str, model: str = DEFAULT_MODEL) -> list[float] | None:
    """Embed a single text string via Ollama API.

    Returns a float vector, or None if embedding fails.
    """
    text = _truncate(text)
    result = _post_json({"model": model, "input": text})
    if result:
        embeddings = result.get("embeddings")
        if embeddings and len(embeddings) > 0:
            return embeddings[0]
    return None


def embed_batch(texts: list[str], model: str = DEFAULT_MODEL) -> list[list[float] | None]:
    """Embed multiple texts using Ollama's native batch API.

    Sends up to BATCH_SIZE texts per request. Returns list of vectors
    aligned with input (None for any that failed).
    """
    if not texts:
        return []

    results: list[list[float] | None] = [None] * len(texts)
    truncated = [_truncate(t) for t in texts]

    for batch_start in range(0, len(truncated), BATCH_SIZE):
        batch = truncated[batch_start:batch_start + BATCH_SIZE]
        batch_end = batch_start + len(batch)

        resp = _post_json({"model": model, "input": batch}, timeout=120)
        if resp:
            embeddings = resp.get("embeddings", [])
            for i, vec in enumerate(embeddings):
                if vec and batch_start + i < len(results):
                    results[batch_start + i] = vec
        else:
            # Batch failed — try singles as fallback
            for i, text in enumerate(batch):
                idx = batch_start + i
                results[idx] = embed_text(text, model)

        # Progress for large batches
        if len(texts) > BATCH_SIZE:
            done = min(batch_end, len(texts))
            print(f"  embedded {done}/{len(texts)}", file=sys.stderr, end="\r")

    if len(texts) > BATCH_SIZE:
        print(file=sys.stderr)  # newline after progress

    return results


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    a_np = np.array(a)
    b_np = np.array(b)
    denom = np.linalg.norm(a_np) * np.linalg.norm(b_np)
    if denom == 0:
        return 0.0
    return float(np.dot(a_np, b_np) / denom)


class EmbeddingIndex:
    """In-memory embedding index with numpy cosine similarity search.

    Stores vectors + metadata. Persists to disk as .npz + .json.
    Supports incremental additions.
    """

    def __init__(self, path: str | Path | None = None):
        self.vectors: list[np.ndarray] = []
        self.metadata: list[dict] = []  # uid, source, thread, ts, etc.
        self.path = Path(path) if path else None

        if self.path and self.path.with_suffix(".npz").exists():
            self._load()

    def add(self, vector: list[float], meta: dict):
        """Add a single vector + metadata to the index."""
        self.vectors.append(np.array(vector, dtype=np.float32))
        self.metadata.append(meta)

    def search(self, query_vector: list[float], k: int = 20) -> list[tuple[dict, float]]:
        """Find k nearest neighbors by cosine similarity.

        Returns list of (metadata, similarity_score) sorted by similarity desc.
        """
        if not self.vectors:
            return []

        q = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q = q / q_norm

        # Stack all vectors into a matrix for batch similarity
        matrix = np.stack(self.vectors)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1  # avoid division by zero
        matrix = matrix / norms

        similarities = matrix @ q
        top_k = np.argsort(similarities)[-k:][::-1]

        return [(self.metadata[i], float(similarities[i])) for i in top_k]

    def save(self):
        """Persist index to disk."""
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        npz_path = self.path.with_suffix(".npz")
        meta_path = self.path.with_suffix(".meta.json")

        if self.vectors:
            np.savez_compressed(npz_path, vectors=np.stack(self.vectors))
        with open(meta_path, "w") as f:
            json.dump(self.metadata, f)

    def _load(self):
        """Load index from disk."""
        npz_path = self.path.with_suffix(".npz")
        meta_path = self.path.with_suffix(".meta.json")

        if meta_path.exists():
            with open(meta_path) as f:
                self.metadata = json.load(f)

        if npz_path.exists():
            data = np.load(npz_path)
            matrix = data["vectors"]
            self.vectors = [matrix[i] for i in range(len(matrix))]

    def __len__(self) -> int:
        return len(self.vectors)

    def __repr__(self) -> str:
        dim = self.vectors[0].shape[0] if self.vectors else 0
        return f"EmbeddingIndex({len(self)} entries, {dim}-dim)"
