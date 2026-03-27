"""Context retrieval — multi-axis hybrid retrieval from corpus + configured sources.

Combines:
- Semantic search (embedding similarity)
- Keyword search (exact string matching)
- Temporal decay (facts fade, patterns persist, anti-patterns get boosted)
- LLM-routed query decomposition selects axes and weights
"""

import difflib
import glob
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .tokens import count_tokens
from .embeddings import embed_text, EmbeddingIndex
from .index import load_identifiers, load_all_metadata
from .query import decompose_query


class ContextRetriever:
    """Multi-axis context retriever.

    Static sources are always included (CLAUDE.md files, etc.).
    Corpus queries use LLM-decomposed retrieval across the embedding index.
    """

    # Default retrieval params — overridden by config["retrieval"]
    DEFAULT_PARAMS = {
        "semantic_k": 30,
        "keyword_k": 30,
        "keyword_weight": 0.3,
        "hybrid_boost": 0.4,
        "identifier_weight": 0.6,
        "decay_half_life_days": 30,
        "decay_floor": 0.3,
        "correction_boost_max": 0.5,
        "context_boost": 1.5,
        "kernel_boost": 2.0,
        "chain_boost": 1.3,
        # Reranking weights — applied after initial sort to rescore top candidates
        "rerank_query_overlap": 1.0,   # boost for query word overlap with chunk
        "rerank_identifier_hit": 2.0,  # boost for exact identifier match in chunk
        "rerank_specificity": 0.5,     # boost for information density (shorter = more specific)
    }

    def __init__(
        self,
        sources: list[str],
        index: EmbeddingIndex | None = None,
        decompose_model: str = "",
        config: dict | None = None,
    ):
        self.sources = sources
        self.index = index
        if not decompose_model:
            from .config import get_model
            decompose_model = get_model("decompose")
        self.decompose_model = decompose_model
        self._known_identifiers: list[str] | None = None
        self._all_metadata: list[dict] | None = None

        # Load retrieval params: config overrides defaults
        self.params = dict(self.DEFAULT_PARAMS)
        if config and "retrieval" in config:
            self.params.update(config["retrieval"])

    def _get_known_identifiers(self) -> list[str]:
        if self._known_identifiers is None:
            self._known_identifiers = load_identifiers()
        return self._known_identifiers

    def _get_all_metadata(self) -> list[dict]:
        """Get full metadata (all corpus entries, not just embedded ones)."""
        if self._all_metadata is None:
            self._all_metadata = load_all_metadata()
        return self._all_metadata

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

    def _retrieve_corpus(self, query: str, decomposition: dict, token_budget: int,
                         role_filter: str = "", project_filter: str = "",
                         exclude_roles: list[str] | None = None) -> str:
        """Retrieve from corpus using decomposed query axes."""

        axes = decomposition.get("axes", [])
        rewritten = decomposition.get("rewritten_query", query)
        expanded_keywords = decomposition.get("keywords", [])
        raw_identifiers = decomposition.get("identifiers", [])

        # Collect candidates from each axis
        all_candidates = {}  # uid → (meta, score)

        for axis_spec in axes:
            axis = axis_spec.get("axis", "semantic")
            weight = axis_spec.get("weight", 0.5)
            filt = axis_spec.get("filter", "")

            if axis == "semantic":
                results = self._search_semantic(rewritten, k=self.params["semantic_k"])
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

            elif axis == "emotional":
                results = self._search_emotion(filt, rewritten, k=20)
                for meta, score in results:
                    uid = meta.get("uid", "")
                    if uid in all_candidates:
                        all_candidates[uid] = (meta, all_candidates[uid][1] + score * weight)
                    else:
                        all_candidates[uid] = (meta, score * weight)

            # polarity, cycle_state — future axes

        # Hybrid: add keyword search results (always runs alongside semantic)
        # Merge expanded keywords from decomposition with raw query terms
        keyword_query = rewritten
        if expanded_keywords:
            keyword_query = rewritten + " " + " ".join(expanded_keywords)
        keyword_results = self._search_keyword(keyword_query, k=self.params["keyword_k"])
        for meta, score in keyword_results:
            uid = meta.get("uid", "")
            if uid in all_candidates:
                existing_meta, existing_score = all_candidates[uid]
                all_candidates[uid] = (existing_meta, existing_score + score * self.params["hybrid_boost"])
            else:
                all_candidates[uid] = (meta, score * self.params["keyword_weight"])

        # Identifier search: fuzzy-resolve then graduated exact match
        if raw_identifiers:
            resolved = self._resolve_identifiers(raw_identifiers)
            if resolved:
                id_results = self._search_identifier(resolved, k=self.params["keyword_k"])
                for meta, score in id_results:
                    uid = meta.get("uid", "")
                    if uid in all_candidates:
                        existing_meta, existing_score = all_candidates[uid]
                        all_candidates[uid] = (existing_meta, existing_score + score * self.params["identifier_weight"])
                    else:
                        all_candidates[uid] = (meta, score * self.params["identifier_weight"])

        # Apply filters
        if role_filter or project_filter or exclude_roles:
            role_f = role_filter.lower() if role_filter else ""
            proj_f = project_filter.lower() if project_filter else ""
            excl = {r.lower() for r in (exclude_roles or [])}
            filtered = {}
            for uid, (meta, score) in all_candidates.items():
                role = meta.get("role", "").lower()
                if role_f and role != role_f:
                    continue
                if excl and role in excl:
                    continue
                if proj_f and proj_f not in meta.get("thread", "").lower():
                    continue
                filtered[uid] = (meta, score)
            all_candidates = filtered

        # Apply temporal decay + role-based scoring
        decayed = []
        for meta, score in all_candidates.values():
            adjusted = self._apply_temporal_decay(meta, score)
            decayed.append((meta, adjusted))

        ranked = sorted(decayed, key=lambda x: -x[1])

        # Rerank: rescore top candidates using query-specific signals
        ranked = self._rerank(query, decomposition, ranked)

        # Assemble into text, fitting budget
        chunks = []
        tokens_used = 0

        for meta, score in ranked:
            content = meta.get("content", "")
            role = meta.get("role", "?")
            thread = meta.get("thread", "?")
            ts = meta.get("ts", "?")[:10]
            uid = meta.get("uid", "")
            heading = meta.get("heading", "")
            chunk_type = meta.get("chunk_type", "")

            # Code entries: show heading (file:function) instead of generic label
            if role == "code" and heading:
                label = f"[{uid} {thread} {chunk_type}] {heading}"
                chunk = f"{label}\n{content}"
            else:
                chunk = f"[{uid} {thread} {ts} {role}] {content}"

            chunk_tokens = count_tokens(chunk)

            if tokens_used + chunk_tokens > token_budget:
                # Skip oversized entries instead of stopping — don't let one
                # giant tool result block all smaller relevant entries
                continue

            chunks.append(chunk)
            tokens_used += chunk_tokens

        return "\n".join(chunks)

    def _rerank(self, query: str, decomposition: dict,
                ranked: list[tuple[dict, float]]) -> list[tuple[dict, float]]:
        """Rerank candidates using query-specific signals that affect ORDER.

        Applied after initial axis scoring + temporal decay. The initial sort
        gets candidates into the pool; reranking determines their final order.
        """
        if not ranked:
            return ranked

        query_lower = query.lower()
        query_words = set(re.split(r'\W+', query_lower)) - {'', 'the', 'a', 'an', 'in', 'on', 'of', 'to', 'for', 'is', 'and', 'or'}
        identifiers = [id.lower() for id in decomposition.get("identifiers", [])]
        keywords = [kw.lower() for kw in decomposition.get("keywords", [])]

        w_overlap = self.params.get("rerank_query_overlap", 1.0)
        w_ident = self.params.get("rerank_identifier_hit", 2.0)
        w_spec = self.params.get("rerank_specificity", 0.5)

        reranked = []
        for meta, score in ranked:
            content = meta.get("content", "")
            content_lower = content.lower()
            bonus = 0.0

            # 1. Query word overlap: what fraction of query words appear in chunk?
            if query_words:
                overlap = sum(1 for w in query_words if w in content_lower) / len(query_words)
                bonus += overlap * w_overlap

            # 2. Identifier hits: exact function/file names in the chunk
            if identifiers:
                ident_hits = sum(1 for ident in identifiers if ident in content_lower)
                bonus += (ident_hits / len(identifiers)) * w_ident

            # 3. Keyword hits from decomposition
            if keywords:
                kw_hits = sum(1 for kw in keywords if kw in content_lower)
                bonus += (kw_hits / len(keywords)) * w_overlap * 0.5

            # 4. Specificity: shorter chunks with hits are more specific/useful
            if bonus > 0 and w_spec > 0:
                # Normalize: 100 chars = 1.0, 1000 chars = 0.1
                specificity = min(1.0, 100 / max(len(content), 1))
                bonus += specificity * w_spec

            reranked.append((meta, score + bonus))

        reranked.sort(key=lambda x: -x[1])
        return reranked

    def _search_semantic(self, query: str, k: int = 30) -> list[tuple[dict, float]]:
        """Semantic search via embedding similarity."""
        if not self.index or len(self.index) == 0:
            return []
        vec = embed_text(query)
        if vec is None:
            return []
        return self.index.search(vec, k=k)

    def _search_temporal(self, filt: str, k: int = 20) -> list[tuple[dict, float]]:
        """Temporal search — find entries by time."""
        all_meta = self._get_all_metadata()
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
        # Filter all metadata by project
        all_meta = self._get_all_metadata()
        matching = [
            m for m in all_meta
            if project_lower in m.get("thread", "").lower()
        ]

        if not matching:
            return []

        # Semantic search within project (if embeddings available)
        vec = embed_text(query)
        if vec is not None and self.index and len(self.index) > 0:
            import numpy as np
            q = np.array(vec, dtype=np.float32)
            q = q / (np.linalg.norm(q) or 1)

            # Find project entries that are also in the embedding index
            project_uids = {m.get("uid") for m in matching}
            scores = []
            for i, m in enumerate(self.index.metadata):
                if m.get("uid") in project_uids:
                    v = self.index.vectors[i]
                    v_norm = np.linalg.norm(v)
                    if v_norm > 0:
                        scores.append((m, float(np.dot(v / v_norm, q))))
            scores.sort(key=lambda x: -x[1])
            return scores[:k]

        # Fallback: return project entries ranked by recency
        matching.sort(key=lambda m: m.get("ts", ""), reverse=True)
        return [(m, 0.5) for m in matching[:k]]

    def _search_keyword(self, query: str, k: int = 30) -> list[tuple[dict, float]]:
        """Keyword search — exact term matching against content.

        Splits query into terms, scores by how many terms match.
        Critical for: file paths, function names, error messages, UIDs.
        Searches ALL corpus entries (not just those with embeddings).
        """
        all_meta = self._get_all_metadata()
        if not all_meta:
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
        for m in all_meta:
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

        # Kernel entries (dream synthesis output) — highest value, no decay
        if role == "kernel":
            return base_score * self.params["kernel_boost"]

        # Chain entries (dream integration output) — valuable, no decay
        if role == "chain":
            return base_score * self.params["chain_boost"]

        # Context entries (CLAUDE.md, plans) — no decay, boosted
        if role == "context":
            return base_score * self.params["context_boost"]

        # No timestamp → no decay
        if not ts:
            return base_score

        # Calculate age in days
        try:
            entry_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            age_days = (datetime.now(timezone.utc) - entry_time).total_seconds() / 86400
        except (ValueError, TypeError):
            return base_score

        # Detect polarity — prefer emotion metadata, fall back to keywords
        emotion_class = meta.get("emotion_class", "")
        content = meta.get("content", "").lower()

        if emotion_class and emotion_class != "neutral":
            # Use model-derived emotion
            is_failure = emotion_class in ("anger", "disgust", "fear", "sadness")
            is_correction = any(w in content for w in (
                "fixed", "the fix", "correct approach", "solved", "the solution",
                "don't do", "instead use", "the right way", "confirmed working",
            ))
        else:
            # Keyword fallback for entries without emotion metadata
            is_correction = any(w in content for w in (
                "fixed", "the fix", "correct approach", "solved", "the solution",
                "don't do", "instead use", "the right way", "confirmed working",
            ))
            is_failure = any(w in content for w in (
                "failed", "error", "bug", "wrong", "broken", "doesn't work",
            ))

        half_life = self.params["decay_half_life_days"]
        floor = self.params["decay_floor"]
        corr_max = self.params["correction_boost_max"]

        if is_correction:
            # Corrections get BOOSTED with age — they're established knowledge
            return base_score * (1.0 + min(age_days / half_life, corr_max))
        elif is_failure and is_correction:
            # Anti-pattern with correction — very valuable, boost
            return base_score * self.params["chain_boost"]
        elif is_failure:
            # Pure failure without correction — mild decay
            return base_score * max(0.7, 1.0 - age_days / (half_life * 2))
        else:
            # Normal content — gentle decay
            decay = math.pow(0.5, age_days / half_life)
            return base_score * max(floor, decay)

    def _search_entity(self, entity: str, k: int = 20) -> list[tuple[dict, float]]:
        """Entity search — find mentions of a specific name/term.

        Searches ALL corpus entries (not just those with embeddings).
        """
        if not entity:
            return []

        all_meta = self._get_all_metadata()
        entity_lower = entity.lower()
        matches = []
        for m in all_meta:
            content = m.get("content", "").lower()
            if entity_lower in content:
                matches.append((m, 0.8))
            if len(matches) >= k:
                break

        return matches

    def _search_emotion(self, filt: str, query: str, k: int = 20) -> list[tuple[dict, float]]:
        """Search by emotion class or valence direction.

        filt can be:
        - An emotion class: "anger", "joy", "sadness", etc.
        - A valence direction: "positive", "negative"
        - Empty: infer from query via semantic heuristic
        """
        all_meta = self._get_all_metadata()
        if not all_meta:
            return []

        # Known emotion classes
        emotion_classes = {"anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"}

        matches = []
        filt_lower = filt.lower().strip() if filt else ""

        for m in all_meta:
            emo_class = m.get("emotion_class", "neutral")
            valence = m.get("emotion_valence", 0.0)
            arousal = m.get("emotion_arousal", 0.0)

            # Skip neutral entries — not interesting for emotion search
            if emo_class == "neutral" and abs(valence) < 0.1:
                continue

            matched = False
            if filt_lower in emotion_classes:
                matched = (emo_class == filt_lower)
            elif filt_lower == "positive":
                matched = (valence > 0.2)
            elif filt_lower == "negative":
                matched = (valence < -0.2)
            else:
                # No filter — return all non-neutral, ranked by arousal
                matched = True

            if matched:
                # Score by arousal (intensity) — more intense = more interesting
                score = 0.5 + arousal * 0.5
                matches.append((m, score))

        matches.sort(key=lambda x: -x[1])
        return matches[:k]

    def _resolve_identifiers(self, raw_identifiers: list[str]) -> list[str]:
        """Fuzzy-match approximate identifiers against the known identifiers index.

        Returns resolved (correct) identifiers sorted by confidence.
        """
        known = self._get_known_identifiers()
        if not known or not raw_identifiers:
            return raw_identifiers  # pass through if no index

        resolved = set()

        for raw in raw_identifiers:
            raw_lower = raw.lower().strip()
            if not raw_lower:
                continue

            # 1. Exact match
            if raw in known or raw_lower in (k.lower() for k in known):
                resolved.add(raw)
                continue

            # Score candidates
            candidates: list[tuple[str, float]] = []

            # Pre-compute raw tokens for rearrangement check
            raw_tokens = set(re.split(r'[_./\-]', raw_lower))
            raw_tokens.discard("")

            for k in known:
                k_lower = k.lower()

                # 2. Token rearrangement — same tokens, different order
                #    spoof_session → session_spoof
                k_tokens = set(re.split(r'[_./\-]', k_lower))
                k_tokens.discard("")
                if raw_tokens and k_tokens and raw_tokens == k_tokens:
                    candidates.append((k, 0.95))
                    continue

                # 3. Substring containment — replace_session matches _replace_session
                #    Require coverage > 60% to avoid "session" matching "do_session"
                if raw_lower in k_lower:
                    coverage = len(raw_lower) / len(k_lower)
                    if coverage >= 0.6:
                        candidates.append((k, 0.5 + coverage * 0.4))
                        continue
                if k_lower in raw_lower:
                    coverage = len(k_lower) / len(raw_lower)
                    if coverage >= 0.6:
                        candidates.append((k, 0.4 + coverage * 0.3))
                        continue

                # 4. SequenceMatcher — handles typos, character reordering
                #    Penalize large length mismatches to avoid "do_session" matching "spoof_session"
                len_ratio = min(len(raw_lower), len(k_lower)) / max(len(raw_lower), len(k_lower))
                if len_ratio < 0.5:
                    continue
                ratio = difflib.SequenceMatcher(None, raw_lower, k_lower).ratio()
                if ratio >= 0.7:
                    candidates.append((k, ratio * len_ratio))

            # Take top matches above threshold
            candidates.sort(key=lambda x: -x[1])
            for cand, score in candidates[:3]:
                if score >= 0.55:
                    resolved.add(cand)

        if resolved - set(raw_identifiers):
            print(f"  identifiers resolved: {list(resolved - set(raw_identifiers))}", file=sys.stderr)

        return list(resolved)

    def _search_identifier(self, identifiers: list[str], k: int = 30) -> list[tuple[dict, float]]:
        """Search corpus for entries containing specific identifiers.

        Graduated scoring: exact filename > path suffix > stem > substring.
        Searches ALL corpus entries (not just those with embeddings).
        """
        all_meta = self._get_all_metadata()
        if not identifiers or not all_meta:
            return []

        matches: dict[str, tuple[dict, float]] = {}  # uid → (meta, best_score)

        for ident in identifiers:
            ident_lower = ident.lower()
            # Extract just the filename if it's a path
            basename = os.path.basename(ident)
            stem = os.path.splitext(basename)[0] if "." in basename else basename

            for m in all_meta:
                uid = m.get("uid", "")
                content = m.get("content", "")
                content_lower = content.lower()

                best = 0.0

                # Exact identifier match
                if ident in content:
                    best = max(best, 1.0)
                elif ident_lower in content_lower:
                    best = max(best, 0.9)

                # Stem match (no extension)
                if stem and len(stem) >= 4 and stem.lower() in content_lower:
                    best = max(best, 0.7)

                # Word-boundary function/var match
                if re.search(r'\b' + re.escape(ident_lower) + r'\b', content_lower):
                    best = max(best, 0.9)

                if best > 0:
                    if uid not in matches or matches[uid][1] < best:
                        matches[uid] = (m, best)

        ranked = sorted(matches.values(), key=lambda x: -x[1])
        return ranked[:k]

    def _cull_with_ollama(self, query: str, numbered_prompt: str) -> list[int] | None:
        """Try culling via local Ollama (fast, free). Returns indices or None."""
        import urllib.request
        import urllib.error

        payload = json.dumps({
            "model": self.decompose_model,  # reuse the decompose model (qwen)
            "messages": [
                {"role": "user", "content": numbered_prompt},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1},
        })

        try:
            req = urllib.request.Request(
                "http://localhost:11434/api/chat",
                data=payload.encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            content = data.get("message", {}).get("content", "")
            match = re.search(r'\[[\d,\s]+\]', content)
            if match:
                return json.loads(match.group())
        except Exception:
            pass
        return None

    def _cull_with_llm(self, query: str, chunks: list[str], budget_tokens: int,
                       model: str = "") -> list[str]:
        """Over-retrieve then LLM-cull: keep only chunks relevant to the query.

        Tries local Ollama first (fast, free). Falls back to claude --print.
        """
        if not model:
            from .config import get_model
            model = get_model("cull")

        if not chunks or len(chunks) <= 5:
            return chunks  # too few to bother culling

        # Limit chunks sent to culler — top 50 is enough to find signal
        cull_limit = min(50, len(chunks))
        cull_candidates = chunks[:cull_limit]
        remainder = chunks[cull_limit:]

        # Number each chunk for reference
        numbered = []
        for i, chunk in enumerate(cull_candidates):
            # Truncate individual chunks for the cull prompt (save tokens)
            preview = chunk[:200] + "..." if len(chunk) > 200 else chunk
            numbered.append(f"[{i}] {preview}")

        prompt = f"""Given this query: "{query}"

Which of these retrieved context chunks are relevant? Return ONLY the numbers of relevant chunks as a JSON array, e.g. [0, 2, 5, 7]. Drop anything that is noise, off-topic, or redundant.

{chr(10).join(numbered)}"""

        # Try local Ollama first (fast)
        indices = self._cull_with_ollama(query, prompt)
        if indices is not None:
            print(f"  culled via local model", file=sys.stderr)
        else:
            # Fall back to claude --print (slower, API)
            try:
                result = subprocess.run(
                    ["claude", "--print", "--model", model],
                    input=prompt,
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode != 0:
                    return chunks

                output = result.stdout.strip()
                match = re.search(r'\[[\d,\s]+\]', output)
                if match:
                    indices = json.loads(match.group())
                else:
                    return chunks
            except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
                return chunks

        if indices:
            kept = [cull_candidates[i] for i in indices if 0 <= i < len(cull_candidates)]
            if kept:
                final = []
                tokens_used = 0
                for chunk in kept:
                    t = count_tokens(chunk)
                    if tokens_used + t > budget_tokens:
                        break
                    final.append(chunk)
                    tokens_used += t
                culled = len(cull_candidates) - len(final)
                if culled > 0:
                    print(f"  culled {culled}/{len(cull_candidates)} chunks", file=sys.stderr)
                return final

        return chunks  # fallback: return all

    def retrieve(self, query: str, token_budget: int, conversation_tail: str = "",
                 cull: bool = False, cull_factor: int = 5,
                 role_filter: str = "", project_filter: str = "",
                 exclude_roles: list[str] | None = None,
                 skip_decompose: bool = False) -> str:
        """Retrieve context from index via LLM-routed query decomposition.

        Args:
            query: the current user message
            token_budget: max tokens for assembled context
            conversation_tail: recent conversation for context-enriched embedding
            skip_decompose: skip LLM decomposition (fast mode — keyword only, no expansion)
        """
        has_embeddings = self.index and len(self.index) > 0
        has_metadata = len(self._get_all_metadata()) > 0

        if has_embeddings or has_metadata:
            # Enrich the query with conversation tail for better embeddings
            enriched_query = query
            if conversation_tail:
                tail_truncated = conversation_tail[-2000:]
                enriched_query = f"{tail_truncated}\n\nCurrent question: {query}"

            if skip_decompose:
                # Fast mode: skip LLM decomposition, use fallback (pure semantic + keyword)
                from .query import _fallback
                decomposition = _fallback(enriched_query)
            else:
                decomposition = decompose_query(enriched_query, model=self.decompose_model)

            if cull:
                # Over-retrieve then LLM-cull for precision
                raw = self._retrieve_corpus(enriched_query, decomposition, token_budget * cull_factor,
                                            role_filter=role_filter, project_filter=project_filter,
                                            exclude_roles=exclude_roles)
                chunks = [line for line in raw.split("\n") if line.strip()]
                culled = self._cull_with_llm(query, chunks, token_budget)
                return "\n".join(culled)
            else:
                return self._retrieve_corpus(enriched_query, decomposition, token_budget,
                                            role_filter=role_filter, project_filter=project_filter,
                                            exclude_roles=exclude_roles)

        # Fallback: static sources if no index at all
        static_text, _ = self._retrieve_static(token_budget)
        return static_text
