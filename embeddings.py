"""Embedding utilities for Continuum.

Embeds text via Ollama (nomic-embed-text) for semantic search and
proximity traversal. Embeddings are minted at write time alongside
UIDs — every entry in the corpus is both addressable (by UID) and
locatable (by semantic neighborhood).
"""

import json
import subprocess
import numpy as np
from pathlib import Path


# Default model — available via Ollama, 768-dim, fast
DEFAULT_MODEL = "nomic-embed-text"


def embed_text(text: str, model: str = DEFAULT_MODEL) -> list[float] | None:
    """Embed a single text string via Ollama API.

    Returns a float vector, or None if embedding fails.
    """
    # Truncate very long texts — embedding models have limits
    if len(text) > 8000:
        text = text[:8000]

    payload = json.dumps({"model": model, "input": text})
    try:
        result = subprocess.run(
            ["curl", "-s", "http://localhost:11434/api/embed", "-d", payload],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout)
        embeddings = data.get("embeddings")
        if embeddings and len(embeddings) > 0:
            return embeddings[0]
        return None
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        return None


def embed_batch(texts: list[str], model: str = DEFAULT_MODEL) -> list[list[float] | None]:
    """Embed multiple texts. Returns list of vectors (None for failures)."""
    return [embed_text(t, model) for t in texts]


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
