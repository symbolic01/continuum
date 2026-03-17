"""Context retrieval — multi-axis hybrid retrieval from corpus + configured sources.

Combines:
- Semantic search (embedding similarity)
- Keyword search (exact string matching)
- Temporal decay (facts fade, patterns persist, anti-patterns get boosted)
- LLM-routed query decomposition selects axes and weights
"""

import glob
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from tokens import count_tokens
from embeddings import embed_text, EmbeddingIndex
from query import decompose_query


class ContextRetriever:
    """Multi-axis context retriever.

    Static sources are always included (CLAUDE.md files, etc.).
    Corpus queries use LLM-decomposed retrieval across the embedding index.
    """

    def __init__(
        self,
        sources: list[str],
        index: EmbeddingIndex | None = None,
        decompose_model: str = "qwen2.5:7b",
    ):
        self.sources = sources
        self.index = index
        self.decompose_model = decompose_model

    def _resolve_paths(self) -> list[Path]:
        """Expand globs and resolve all source paths."""
        paths = []
        for source in self.sources:
            expanded = Path(source).expanduser()
            if "*" in source or "?" in source:
                for match in sorted(glob.glob(str(expanded), recursive=True)):
                    p = Path(match)
                    if p.is_file():
                        paths.append(p)
            elif expanded.is_file():
                paths.append(expanded)
            elif expanded.is_dir():
                for p in sorted(expanded.rglob("*.md")):
                    paths.append(p)
        return paths

    def _retrieve_static(self, token_budget: int) -> tuple[str, int]:
        """Read static sources (CLAUDE.md files), fit to budget.

        Returns (assembled_text, tokens_used).
        """
        paths = self._resolve_paths()
        if not paths:
            return "", 0

        chunks = []
        total_tokens = 0

        for path in paths:
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except (OSError, PermissionError):
                continue

            chunk = f"--- {path} ---\n{content}\n"
            chunk_tokens = count_tokens(chunk)

            if total_tokens + chunk_tokens > token_budget:
                remaining = token_budget - total_tokens
                if remaining > 100:
                    truncated = content[: remaining * 4]
                    chunks.append(f"--- {path} (truncated) ---\n{truncated}\n")
                break

            chunks.append(chunk)
            total_tokens += chunk_tokens

        return "\n".join(chunks), total_tokens

    def _retrieve_corpus(self, query: str, decomposition: dict, token_budget: int) -> str:
        """Retrieve from corpus using decomposed query axes."""
        if not self.index or len(self.index) == 0:
            return ""

        axes = decomposition.get("axes", [])
        rewritten = decomposition.get("rewritten_query", query)

        # Collect candidates from each axis
        all_candidates = {}  # uid → (meta, score)

        for axis_spec in axes:
            axis = axis_spec.get("axis", "semantic")
            weight = axis_spec.get("weight", 0.5)
            filt = axis_spec.get("filter", "")

            if axis == "semantic":
                results = self._search_semantic(rewritten, k=30)
                for meta, score in results:
                    uid = meta.get("uid", "")
                    if uid in all_candidates:
                        all_candidates[uid] = (meta, all_candidates[uid][1] + score * weight)
                    else:
                        all_candidates[uid] = (meta, score * weight)

            elif axis == "temporal":
                results = self._search_temporal(filt, k=20)
                for meta, score in results:
                    uid = meta.get("uid", "")
                    if uid in all_candidates:
                        all_candidates[uid] = (meta, all_candidates[uid][1] + score * weight)
                    else:
                        all_candidates[uid] = (meta, score * weight)

            elif axis == "project":
                results = self._search_project(filt, rewritten, k=20)
                for meta, score in results:
                    uid = meta.get("uid", "")
                    if uid in all_candidates:
                        all_candidates[uid] = (meta, all_candidates[uid][1] + score * weight)
                    else:
                        all_candidates[uid] = (meta, score * weight)

            elif axis == "entity":
                results = self._search_entity(filt, k=20)
                for meta, score in results:
                    uid = meta.get("uid", "")
                    if uid in all_candidates:
                        all_candidates[uid] = (meta, all_candidates[uid][1] + score * weight)
                    else:
                        all_candidates[uid] = (meta, score * weight)

            elif axis == "anti_pattern":
                results = self._search_semantic(rewritten + " error failed wrong bug", k=20)
                for meta, score in results:
                    uid = meta.get("uid", "")
                    if uid in all_candidates:
                        all_candidates[uid] = (meta, all_candidates[uid][1] + score * weight)
                    else:
                        all_candidates[uid] = (meta, score * weight)

            elif axis == "causal":
                results = self._search_semantic(rewritten + " because decided reason why", k=20)
                for meta, score in results:
                    uid = meta.get("uid", "")
                    if uid in all_candidates:
                        all_candidates[uid] = (meta, all_candidates[uid][1] + score * weight)
                    else:
                        all_candidates[uid] = (meta, score * weight)

            # emotional, polarity, cycle_state — future axes

        # Hybrid: add keyword search results (always runs alongside semantic)
        keyword_results = self._search_keyword(rewritten, k=30)
        for meta, score in keyword_results:
            uid = meta.get("uid", "")
            if uid in all_candidates:
                # Boost entries found by BOTH semantic and keyword
                existing_meta, existing_score = all_candidates[uid]
                all_candidates[uid] = (existing_meta, existing_score + score * 0.4)
            else:
                all_candidates[uid] = (meta, score * 0.3)

        # Apply temporal decay + role-based scoring
        decayed = []
        for meta, score in all_candidates.values():
            adjusted = self._apply_temporal_decay(meta, score)
            decayed.append((meta, adjusted))

        ranked = sorted(decayed, key=lambda x: -x[1])

        # Assemble into text, fitting budget
        chunks = []
        tokens_used = 0

        for meta, score in ranked:
            content = meta.get("content", "")
            role = meta.get("role", "?")
            thread = meta.get("thread", "?")
            ts = meta.get("ts", "?")[:10]
            uid = meta.get("uid", "")

            chunk = f"[{uid} {thread} {ts} {role}] {content}"
            chunk_tokens = count_tokens(chunk)

            if tokens_used + chunk_tokens > token_budget:
                break

            chunks.append(chunk)
            tokens_used += chunk_tokens

        return "\n".join(chunks)

    def _search_semantic(self, query: str, k: int = 30) -> list[tuple[dict, float]]:
        """Semantic search via embedding similarity."""
        vec = embed_text(query)
        if vec is None:
            return []
        return self.index.search(vec, k=k)

    def _search_temporal(self, filt: str, k: int = 20) -> list[tuple[dict, float]]:
        """Temporal search — find entries by time."""
        all_meta = self.index.metadata
        if not all_meta:
            return []

        if filt == "oldest":
            sorted_meta = sorted(all_meta, key=lambda m: m.get("ts", ""))
            return [(m, 1.0 - i * 0.02) for i, m in enumerate(sorted_meta[:k])]
        elif filt == "newest":
            sorted_meta = sorted(all_meta, key=lambda m: m.get("ts", ""), reverse=True)
            return [(m, 1.0 - i * 0.02) for i, m in enumerate(sorted_meta[:k])]
        elif filt.startswith("after:"):
            date = filt[6:]
            filtered = [m for m in all_meta if m.get("ts", "") >= date]
            filtered.sort(key=lambda m: m.get("ts", ""))
            return [(m, 0.8) for m in filtered[:k]]
        elif filt.startswith("before:"):
            date = filt[7:]
            filtered = [m for m in all_meta if m.get("ts", "") < date]
            filtered.sort(key=lambda m: m.get("ts", ""), reverse=True)
            return [(m, 0.8) for m in filtered[:k]]
        else:
            # Default: most recent
            sorted_meta = sorted(all_meta, key=lambda m: m.get("ts", ""), reverse=True)
            return [(m, 1.0 - i * 0.02) for i, m in enumerate(sorted_meta[:k])]

    def _search_project(self, project: str, query: str, k: int = 20) -> list[tuple[dict, float]]:
        """Project-filtered search — semantic within a project."""
        if not project:
            return self._search_semantic(query, k)

        project_lower = project.lower()
        # Filter index metadata by project
        matching_indices = [
            i for i, m in enumerate(self.index.metadata)
            if project_lower in m.get("thread", "").lower()
        ]

        if not matching_indices:
            return []

        # Semantic search within project
        vec = embed_text(query)
        if vec is None:
            return [(self.index.metadata[i], 0.5) for i in matching_indices[:k]]

        import numpy as np
        q = np.array(vec, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1)

        scores = []
        for i in matching_indices:
            v = self.index.vectors[i]
            v_norm = np.linalg.norm(v)
            if v_norm == 0:
                scores.append((i, 0.0))
            else:
                scores.append((i, float(np.dot(v / v_norm, q))))

        scores.sort(key=lambda x: -x[1])
        return [(self.index.metadata[i], s) for i, s in scores[:k]]

    def _search_keyword(self, query: str, k: int = 30) -> list[tuple[dict, float]]:
        """Keyword search — exact term matching against content.

        Splits query into terms, scores by how many terms match.
        Critical for: file paths, function names, error messages, UIDs.
        """
        if not self.index or not self.index.metadata:
            return []

        # Extract meaningful terms (3+ chars, skip common words)
        stop_words = {"the", "and", "for", "are", "but", "not", "you", "all", "can",
                      "had", "her", "was", "one", "our", "out", "has", "have", "been",
                      "this", "that", "with", "from", "they", "will", "what", "when",
                      "how", "who", "which", "their", "about", "would", "there", "could"}
        terms = [t.lower() for t in re.split(r'\W+', query) if len(t) >= 3 and t.lower() not in stop_words]

        if not terms:
            return []

        matches = []
        for m in self.index.metadata:
            content = m.get("content", "").lower()
            hits = sum(1 for t in terms if t in content)
            if hits > 0:
                score = hits / len(terms)  # fraction of terms found
                matches.append((m, score))

        matches.sort(key=lambda x: -x[1])
        return matches[:k]

    def _apply_temporal_decay(self, meta: dict, base_score: float) -> float:
        """Apply temporal decay — facts fade, patterns and corrections persist.

        - context entries (CLAUDE.md): no decay (actively maintained)
        - anti-pattern content: negative decay (gets boosted with age)
        - normal session turns: gentle decay over weeks
        """
        role = meta.get("role", "")
        ts = meta.get("ts", "")

        # Context entries (CLAUDE.md, plans) — no decay, boosted
        if role == "context":
            return base_score * 1.5

        # No timestamp → no decay
        if not ts:
            return base_score

        # Calculate age in days
        try:
            entry_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            age_days = (datetime.now(timezone.utc) - entry_time).total_seconds() / 86400
        except (ValueError, TypeError):
            return base_score

        # Check for anti-pattern / correction content
        content = meta.get("content", "").lower()
        is_correction = any(w in content for w in (
            "fixed", "the fix", "correct approach", "solved", "the solution",
            "don't do", "instead use", "the right way", "confirmed working",
        ))
        is_failure = any(w in content for w in (
            "failed", "error", "bug", "wrong", "broken", "doesn't work",
        ))

        if is_correction:
            # Corrections get BOOSTED with age — they're established knowledge
            return base_score * (1.0 + min(age_days / 30, 0.5))  # up to 1.5x boost
        elif is_failure and is_correction:
            # Anti-pattern with correction — very valuable, boost
            return base_score * 1.3
        elif is_failure:
            # Pure failure without correction — mild decay
            return base_score * max(0.7, 1.0 - age_days / 60)
        else:
            # Normal content — gentle decay
            # Half-life of ~30 days: score * 0.5^(age/30)
            decay = math.pow(0.5, age_days / 30)
            # Floor at 0.3 — very old content can still be retrieved, just ranked lower
            return base_score * max(0.3, decay)

    def _search_entity(self, entity: str, k: int = 20) -> list[tuple[dict, float]]:
        """Entity search — find mentions of a specific name/term."""
        if not entity:
            return []

        entity_lower = entity.lower()
        matches = []
        for m in self.index.metadata:
            content = m.get("content", "").lower()
            if entity_lower in content:
                matches.append((m, 0.8))
            if len(matches) >= k:
                break

        return matches

    def retrieve(self, query: str, token_budget: int) -> str:
        """Retrieve context from index via LLM-routed query decomposition.

        If index is available, all retrieval is semantic — no files loaded wholesale.
        Falls back to static source reading if no index is available.
        """
        if self.index and len(self.index) > 0:
            decomposition = decompose_query(query, model=self.decompose_model)
            return self._retrieve_corpus(query, decomposition, token_budget)

        # Fallback: static sources if no index
        static_text, _ = self._retrieve_static(token_budget)
        return static_text
