"""Dream Engine — offline integration pipeline for the continuum corpus.

Runs during idle time to pre-compute connections between corpus chunks:
- Integration passes find thematic, causal, temporal chains
- Temporal reconnection surfaces cross-temporal links
- Writes chain chunks back into the corpus as first-class entries
- Generates report data for the drill-down UI

Adapted from the AI Dreams hackathon pipeline (pipeline.py).
"""

import json
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .config import load_config, get_model
from .embeddings import EmbeddingIndex, embed_text, embed_batch
from .index import (
    DEFAULT_CORPUS_DIR, DEFAULT_INDEX_PATH, DEFAULT_ALL_META_PATH,
    build_index, load_all_metadata, load_index,
)
from .session_log import mint_uid
from .tokens import count_tokens

# ── Paths ──────────────────────────────────────────────────────────────

CHAINS_DIR = DEFAULT_CORPUS_DIR / "_chains"
DREAM_STATE_PATH = Path.home() / ".continuum" / "dream_state.json"
DREAM_REPORT_PATH = Path.home() / ".continuum" / "dream_report.json"
XREFS_PATH = Path.home() / ".continuum" / "index" / "xrefs.json"

# ── Prompts (adapted from hackathon pipeline.py) ───────────────────────

INTEGRATION_SYSTEM_PROMPT = """You are an integration analyst. You find connections between chunks of session logs, notes, and code from a developer's work across multiple projects.

Given a cluster of corpus chunks, identify:

1. CHAINS: Groups that belong together thematically, causally, or temporally.
   Each chain has: type (thematic|causal|temporal|correction|orphan), synthesis (1 sentence), member_uids (which chunks belong)

2. CORRECTIONS: Failure→success pairs. If one chunk describes a bug/error and another describes the fix, link them.
   Type: "correction"

3. ORPHANS: Incomplete cycles — something that was started (a question asked, a task begun, an intention stated) but never finished or resolved in any chunk in this cluster.
   Type: "orphan"

4. CROSS_PROJECT: Patterns that span multiple projects (thread field differs). Flag these explicitly.

Rules:
- Only create chains for GENUINE connections — not everything is related
- Each chain needs at least 2 member chunks
- The synthesis should be a single sentence capturing WHY these chunks belong together
- CRITICAL: Only cite UIDs that actually appear in the input chunks
- Prefer fewer, high-quality chains over many weak ones
- "orphan" chains can have a single member if it represents an unresolved question/task

Output valid JSON:
{
  "chains": [
    {
      "type": "thematic|causal|temporal|correction|orphan",
      "synthesis": "one sentence explaining the connection",
      "member_uids": ["«uid1»", "«uid2»"],
      "cross_project": true/false
    }
  ]
}

If no meaningful connections exist, return: {"chains": []}"""

INTEGRATION_USER_TEMPLATE = """Here are {count} corpus chunks from different times and projects. Find connections.

CHUNKS:
{chunks}

Output JSON only."""


# ── DreamEngine ────────────────────────────────────────────────────────

class DreamEngine:
    """Offline integration engine for the continuum corpus."""

    def __init__(
        self,
        config: dict | None = None,
        dry_run: bool = False,
        max_passes: int = 50,
        max_wall_time: int = 1800,
        max_llm_tokens: int = 500_000,
        max_chains: int = 500,
        cluster_size_min: int = 3,
        cluster_size_max: int = 8,
        model: str = "",
        verbose: bool = False,
    ):
        self.config = config or load_config()
        self.dry_run = dry_run
        self.max_passes = max_passes
        self.max_wall_time = max_wall_time
        self.max_llm_tokens = max_llm_tokens
        self.max_chains = max_chains
        self.cluster_size_min = cluster_size_min
        self.cluster_size_max = cluster_size_max
        self.model = model or get_model("dream", self.config)
        self.verbose = verbose

        # State
        self.all_metadata: list[dict] = []
        self.index: EmbeddingIndex | None = None
        self.existing_chain_uids: set[str] = set()
        self.existing_member_sets: list[frozenset[str]] = []
        self.new_chains: list[dict] = []
        self.pass_history: list[int] = []
        self.tokens_used: int = 0
        self.start_time: float = 0.0

    def load_corpus(self):
        """Load the corpus metadata and embedding index."""
        self.all_metadata = load_all_metadata()
        self.index = load_index()
        self._load_existing_chains()
        if self.verbose:
            print(f"[dream] Loaded {len(self.all_metadata)} metadata entries, "
                  f"{len(self.index)} embedded, "
                  f"{len(self.existing_chain_uids)} existing chains",
                  file=sys.stderr)

    def _load_existing_chains(self):
        """Load existing chain chunks for idempotency."""
        self.existing_chain_uids.clear()
        self.existing_member_sets.clear()
        if not CHAINS_DIR.exists():
            return
        for f in CHAINS_DIR.glob("*.jsonl"):
            for line in open(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                uid = entry.get("uid", "")
                if uid:
                    self.existing_chain_uids.add(uid)
                members = entry.get("member_uids", [])
                if members:
                    self.existing_member_sets.append(frozenset(members))

    def _is_duplicate_chain(self, member_uids: list[str]) -> bool:
        """Check if a chain with these members already exists."""
        candidate = frozenset(member_uids)
        for existing in self.existing_member_sets:
            # Exact match or high overlap (Jaccard > 0.5)
            overlap = len(candidate & existing)
            union = len(candidate | existing)
            if union > 0 and overlap / union > 0.5:
                return True
        return False

    def _should_stop(self) -> tuple[bool, str]:
        """Check termination conditions."""
        elapsed = time.time() - self.start_time
        if elapsed >= self.max_wall_time:
            return True, f"wall time ({elapsed:.0f}s >= {self.max_wall_time}s)"
        if self.tokens_used >= self.max_llm_tokens:
            return True, f"token budget ({self.tokens_used} >= {self.max_llm_tokens})"
        if len(self.new_chains) >= self.max_chains:
            return True, f"chain cap ({len(self.new_chains)} >= {self.max_chains})"
        return False, ""

    def _is_converged(self) -> bool:
        """Check if recent passes produced minimal new chains (informational)."""
        window = self.config.get("dream", {}).get("convergence_window", 3)
        threshold = self.config.get("dream", {}).get("convergence_threshold", 2)
        if len(self.pass_history) < window:
            return False
        recent = self.pass_history[-window:]
        return all(count <= threshold for count in recent)

    # ── Cluster building ───────────────────────────────────────────────

    def _is_low_content(self, meta: dict) -> bool:
        """Check if a chunk has too little semantic content for integration.

        Matches the same filter applied during ingest — these entries
        should not have been embedded in the first place.
        """
        content = meta.get("content", "").strip()
        if not content:
            return True
        # Pure tool-use bracket summaries: [Read: path], [TaskUpdate], etc.
        if content.startswith("[") and content.endswith("]") and len(content) < 200:
            return True
        # Common filler responses
        if content in ("No response requested.",
                       "I have this context. Ready to continue."):
            return True
        return False

    def _build_cluster_for_seed(self, seed_meta: dict, seed_vec) -> list[dict] | None:
        """Build a single cluster around a seed using multi-axis similarity.

        Axes (matching retrieval system):
        - Semantic: cosine similarity of embeddings (primary)
        - Temporal: same session or nearby timestamps
        - Project: same thread/project affinity
        - Keyword: shared identifiers or terms
        """
        # Semantic neighbors (primary axis)
        neighbors = self.index.search(seed_vec.tolist(), k=self.cluster_size_max * 3)

        seed_thread = seed_meta.get("thread", "")
        seed_ts = seed_meta.get("ts", "")

        scored: list[tuple[dict, float]] = []
        for meta, sem_sim in neighbors:
            if meta.get("role") == "chain" and meta.get("uid") not in self._chain_seed_uids:
                continue
            if self._is_low_content(meta):
                continue

            # Multi-axis scoring
            score = sem_sim * 0.5  # semantic: 50% weight

            # Project affinity: same project gets a boost
            if meta.get("thread") == seed_thread and seed_thread:
                score += 0.15
            # Cross-project is also interesting — don't penalize, just don't boost

            # Temporal proximity: nearby timestamps get a boost
            meta_ts = meta.get("ts", "")
            if seed_ts and meta_ts and len(seed_ts) >= 10 and len(meta_ts) >= 10:
                try:
                    from datetime import datetime
                    dt_seed = datetime.fromisoformat(seed_ts.replace("Z", "+00:00"))
                    dt_meta = datetime.fromisoformat(meta_ts.replace("Z", "+00:00"))
                    days_apart = abs((dt_meta - dt_seed).total_seconds()) / 86400
                    if days_apart < 1:
                        score += 0.2   # same day
                    elif days_apart < 7:
                        score += 0.1   # same week
                except (ValueError, TypeError):
                    pass

            # Keyword overlap: shared words in content
            seed_words = set(seed_meta.get("content", "").lower().split())
            meta_words = set(meta.get("content", "").lower().split())
            if seed_words and meta_words:
                overlap = len(seed_words & meta_words)
                union = len(seed_words | meta_words)
                if union > 0:
                    score += 0.15 * (overlap / union)  # Jaccard

            scored.append((meta, score))

        # Sort by combined score, take top cluster_size_max
        scored.sort(key=lambda x: x[1], reverse=True)
        cluster = [meta for meta, _ in scored[:self.cluster_size_max]]

        if len(cluster) >= self.cluster_size_min:
            return cluster
        return None

    def _prepare_seed_pool(self) -> list[tuple[dict, any, float]]:
        """Build the initial seed pool with interleaved priorities.

        Strategy: start with newest corpus chunks (most likely to have
        unprocessed connections), then interleave chain seeds to build
        higher-order connections from lower-order ones. Avoids pure
        chain inbreeding by mixing fresh corpus throughout.

        Returns list of (meta, vector, priority) tuples.
        Priority levels:
          3.0 — newly created chains (self-reinforcing, inserted during run)
          2.0 — existing chain chunks
          1.5 — newest unchained corpus (last 7 days)
          1.0 — unchained corpus
          0.5 — already-chained corpus
        """
        import random

        chained_uids = set()
        for ms in self.existing_member_sets:
            chained_uids.update(ms)
        for chain in self.new_chains:
            chained_uids.update(chain.get("member_uids", []))

        # Determine recency threshold (7 days)
        from datetime import datetime, timedelta, timezone
        recent_cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()

        chain_seeds = []
        fresh_seeds = []
        corpus_seeds = []
        stale_seeds = []

        for i, meta in enumerate(self.index.metadata):
            if self._is_low_content(meta):
                continue

            role = meta.get("role", "")
            uid = meta.get("uid", "")
            ts = meta.get("ts", "")

            if role == "chain":
                chain_seeds.append((meta, self.index.vectors[i], 2.0))
            elif uid not in chained_uids:
                if ts >= recent_cutoff:
                    fresh_seeds.append((meta, self.index.vectors[i], 1.5))
                else:
                    corpus_seeds.append((meta, self.index.vectors[i], 1.0))
            else:
                stale_seeds.append((meta, self.index.vectors[i], 0.5))

        # Shuffle each band
        for band in [chain_seeds, fresh_seeds, corpus_seeds, stale_seeds]:
            random.shuffle(band)

        # Interleave: fresh first, then alternate chain/corpus seeds
        # Pattern: [fresh...] [chain, corpus, chain, corpus, ...] [stale...]
        result = list(fresh_seeds)

        # Interleave chains and corpus (not pure chain-first)
        ci, co = 0, 0
        while ci < len(chain_seeds) or co < len(corpus_seeds):
            if ci < len(chain_seeds):
                result.append(chain_seeds[ci])
                ci += 1
            if co < len(corpus_seeds):
                result.append(corpus_seeds[co])
                co += 1

        result.extend(stale_seeds)

        if self.verbose:
            print(f"[dream] Seed pool: {len(fresh_seeds)} fresh, "
                  f"{len(chain_seeds)} chain, "
                  f"{len(corpus_seeds)} corpus, "
                  f"{len(stale_seeds)} stale",
                  file=sys.stderr)

        return result

    # ── LLM integration call ───────────────────────────────────────────

    def _format_chunks_for_prompt(self, cluster: list[dict]) -> str:
        """Format cluster chunks for the integration prompt."""
        lines = []
        for meta in cluster:
            uid = meta.get("uid", "")
            thread = meta.get("thread", "?")
            ts = meta.get("ts", "")[:10]
            role = meta.get("role", "?")
            content = meta.get("content", "")[:300]
            heading = meta.get("heading", "")
            prefix = f"[{uid} {thread} {ts} {role}]"
            if heading:
                prefix += f" {heading}"
            lines.append(f"{prefix} {content}")
        return "\n".join(lines)

    def _call_llm(self, cluster: list[dict]) -> list[dict]:
        """Send a cluster to the dream model and parse chain results."""
        chunks_text = self._format_chunks_for_prompt(cluster)
        user_prompt = INTEGRATION_USER_TEMPLATE.format(
            count=len(cluster),
            chunks=chunks_text,
        )

        # Estimate tokens for budget tracking
        prompt_tokens = count_tokens(INTEGRATION_SYSTEM_PROMPT + user_prompt)

        # Call Ollama
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": INTEGRATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.3},
            "format": "json",
        }

        try:
            result = subprocess.run(
                ["curl", "-s", "http://localhost:11434/api/chat",
                 "-d", json.dumps(payload)],
                capture_output=True, text=True, timeout=120,
            )
            response = json.loads(result.stdout)
            content = response.get("message", {}).get("content", "")
            response_tokens = count_tokens(content)
            self.tokens_used += prompt_tokens + response_tokens
        except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError) as e:
            if self.verbose:
                print(f"  [error] LLM call failed: {e}", file=sys.stderr)
            self.tokens_used += prompt_tokens
            return []

        # Parse response
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed.get("chains", [])
            return []
        except json.JSONDecodeError:
            if self.verbose:
                print(f"  [warn] JSON parse failed: {content[:100]}", file=sys.stderr)
            return []

    # ── Chain chunk creation ───────────────────────────────────────────

    def _clean_member_uids(self, member_uids: list[str]) -> list[str]:
        """Strip hallucinated UIDs not present in corpus."""
        known_uids = {m.get("uid", "") for m in self.all_metadata}
        cleaned = []
        for uid in member_uids:
            # Normalize: strip brackets if LLM added them
            uid = uid.strip().strip("[]")
            if uid in known_uids:
                cleaned.append(uid)
        return cleaned

    def _create_chain_chunk(self, chain_data: dict, pass_num: int) -> dict | None:
        """Create a chain corpus entry from LLM output."""
        member_uids = self._clean_member_uids(chain_data.get("member_uids", []))
        if len(member_uids) < 1:
            return None

        # Dedup check
        if self._is_duplicate_chain(member_uids):
            return None

        synthesis = chain_data.get("synthesis", "").strip()
        if not synthesis:
            return None

        # Determine primary project from members
        member_threads = []
        for meta in self.all_metadata:
            if meta.get("uid") in member_uids:
                t = meta.get("thread", "")
                if t:
                    member_threads.append(t)

        primary_thread = max(set(member_threads), key=member_threads.count) if member_threads else "unknown"

        uid = mint_uid()
        now = datetime.now(timezone.utc).isoformat()

        entry = {
            "uid": uid,
            "role": "chain",
            "content": synthesis,
            "turn": 0,
            "ts": now,
            "thread": primary_thread,
            "source_file": "dream",
            "heading": f"chain: {synthesis[:60]}",
            "chunk_type": "chain",
            "chain_type": chain_data.get("type", "thematic"),
            "member_uids": member_uids,
            "member_projects": list(set(member_threads)),
            "cross_project": chain_data.get("cross_project", False),
            "dream_pass": pass_num,
            "dream_run": now,
        }

        # Embed the chain chunk
        embedding = embed_text(synthesis)
        if embedding:
            entry["embedding"] = embedding

        return entry

    # ── Integration passes ─────────────────────────────────────────────

    def run_integration_passes(self) -> dict:
        """Run self-reinforcing integration until time/token/chain cap fires.

        Dreamlike: new chain chunks become preferred seeds for the next
        cluster. The dream feeds on its own discoveries, building
        higher-order connections from lower-order ones.

        Multi-axis clustering: semantic + temporal + project + keyword.
        """
        self.start_time = time.time()
        self.new_chains = []
        self.pass_history = []
        self.tokens_used = 0
        self._chain_seed_uids = set()  # track which chains we've used as seeds

        print(f"[dream] Starting self-reinforcing integration (model: {self.model})",
              file=sys.stderr)
        print(f"[dream] Limits: {self.max_wall_time}s wall, "
              f"{self.max_llm_tokens} tokens, "
              f"{self.max_chains} chains", file=sys.stderr)

        # Build prioritized seed pool: chains > unchained > chained
        seed_pool = self._prepare_seed_pool()
        if not seed_pool:
            print(f"[dream] No seeds available", file=sys.stderr)
            return self._make_stats(0)

        chain_seeds = sum(1 for _, _, p in seed_pool if p >= 2.0)
        corpus_seeds = len(seed_pool) - chain_seeds
        print(f"[dream] Seed pool: {chain_seeds} chain seeds + "
              f"{corpus_seeds} corpus seeds", file=sys.stderr)

        clusters_processed = 0
        seed_idx = 0
        last_progress = self.start_time

        while seed_idx < len(seed_pool):
            stop, reason = self._should_stop()
            if stop:
                print(f"[dream] Stopped: {reason}", file=sys.stderr)
                break

            seed_meta, seed_vec, priority = seed_pool[seed_idx]
            seed_idx += 1

            # Build cluster around this seed (multi-axis)
            cluster = self._build_cluster_for_seed(seed_meta, seed_vec)
            if not cluster:
                continue

            if self.dry_run:
                uids = [m.get("uid", "?") for m in cluster]
                kind = "chain→" if priority >= 2.0 else ""
                print(f"  [dry-run] {kind}Cluster {clusters_processed+1}: "
                      f"{len(cluster)} chunks ({', '.join(uids[:3])}...)")
                clusters_processed += 1
                continue

            raw_chains = self._call_llm(cluster)
            new_in_cluster = 0
            for chain_data in raw_chains:
                entry = self._create_chain_chunk(chain_data, clusters_processed)
                if entry:
                    self.new_chains.append(entry)
                    self.existing_member_sets.append(
                        frozenset(entry["member_uids"]))
                    new_in_cluster += 1

                    # Self-reinforcing: add new chain as a preferred seed
                    embedding = entry.get("embedding")
                    if embedding:
                        import numpy as np
                        chain_meta = {
                            "uid": entry["uid"],
                            "role": "chain",
                            "content": entry["content"],
                            "thread": entry["thread"],
                            "ts": entry["ts"],
                        }
                        vec = np.array(embedding, dtype=np.float32)
                        # Insert near the front of remaining seeds (preferred)
                        seed_pool.insert(seed_idx, (chain_meta, vec, 3.0))
                        self._chain_seed_uids.add(entry["uid"])

            clusters_processed += 1

            # Logging
            kind = "chain→" if priority >= 2.0 else ""
            if self.verbose:
                threads = set(m.get("thread", "?") for m in cluster)
                remaining = len(seed_pool) - seed_idx
                print(f"[dream] {kind}Cluster {clusters_processed}: "
                      f"{len(cluster)} chunks → {len(raw_chains)} raw, "
                      f"{new_in_cluster} new | "
                      f"projects: {', '.join(threads)} | "
                      f"{remaining} seeds left",
                      file=sys.stderr)

            # Progress every 30 seconds (non-verbose)
            now = time.time()
            if not self.verbose and now - last_progress >= 30:
                elapsed = now - self.start_time
                remaining = len(seed_pool) - seed_idx
                print(f"[dream] Progress: {clusters_processed} clusters, "
                      f"{len(self.new_chains)} chains, "
                      f"{self.tokens_used} tokens, "
                      f"{remaining} seeds left, "
                      f"{elapsed:.0f}s",
                      file=sys.stderr)
                last_progress = now

        return self._make_stats(clusters_processed)

    def _make_stats(self, clusters_processed: int) -> dict:
        """Build stats dict for the integration run."""
        return {
            "passes": clusters_processed,
            "chains_created": len(self.new_chains),
            "tokens_used": self.tokens_used,
            "elapsed_seconds": time.time() - self.start_time,
            "converged": False,
            "clusters_total": 0,
        }

    # ── Temporal reconnection ──────────────────────────────────────────

    def run_temporal_reconnection(
        self,
        min_distance_days: int = 14,
        similarity_threshold: float = 0.7,
    ) -> list[dict]:
        """Find cross-temporal connections in the corpus."""
        if not self.index or len(self.index) == 0:
            return []

        print(f"[dream] Running temporal reconnection "
              f"(min {min_distance_days}d gap, sim > {similarity_threshold})",
              file=sys.stderr)

        # Group metadata by week
        from collections import defaultdict
        weeks: dict[str, list[int]] = defaultdict(list)
        for i, meta in enumerate(self.index.metadata):
            ts = meta.get("ts", "")
            if len(ts) >= 10 and meta.get("role") != "chain" and not self._is_low_content(meta):
                # Week key: YYYY-WNN
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    week_key = f"{dt.year}-W{dt.isocalendar()[1]:02d}"
                    weeks[week_key] = weeks.get(week_key, [])
                    weeks[week_key].append(i)
                except (ValueError, TypeError):
                    continue

        week_keys = sorted(weeks.keys())
        if len(week_keys) < 2:
            print(f"[dream] Only {len(week_keys)} weeks — skipping temporal reconnection",
                  file=sys.stderr)
            return []

        # Compute week centroids
        import numpy as np
        centroids = {}
        for wk in week_keys:
            indices = weeks[wk]
            if not indices:
                continue
            vecs = [self.index.vectors[i] for i in indices]
            centroids[wk] = np.mean(np.stack(vecs), axis=0)

        # Find cross-temporal pairs
        temporal_links = []
        for i, wk_a in enumerate(week_keys):
            for wk_b in week_keys[i+1:]:
                if wk_a not in centroids or wk_b not in centroids:
                    continue

                # Check temporal distance
                try:
                    dt_a = datetime.strptime(wk_a + "-1", "%Y-W%W-%w")
                    dt_b = datetime.strptime(wk_b + "-1", "%Y-W%W-%w")
                    days_apart = abs((dt_b - dt_a).days)
                except ValueError:
                    continue

                if days_apart < min_distance_days:
                    continue

                # Cosine similarity between centroids
                ca, cb = centroids[wk_a], centroids[wk_b]
                norm_a, norm_b = np.linalg.norm(ca), np.linalg.norm(cb)
                if norm_a == 0 or norm_b == 0:
                    continue
                sim = float(np.dot(ca, cb) / (norm_a * norm_b))

                if sim >= similarity_threshold:
                    # Drill into specific chunk pairs
                    best_pairs = self._find_best_cross_pairs(
                        weeks[wk_a], weeks[wk_b], top_k=3)

                    for pair_meta_a, pair_meta_b, pair_sim in best_pairs:
                        if self.dry_run:
                            print(f"  [dry-run] Temporal link: {wk_a} ↔ {wk_b} "
                                  f"(sim={pair_sim:.2f}, {days_apart}d apart)")
                            continue

                        member_uids = [
                            pair_meta_a.get("uid", ""),
                            pair_meta_b.get("uid", ""),
                        ]
                        member_uids = [u for u in member_uids if u]
                        if len(member_uids) < 2:
                            continue
                        if self._is_duplicate_chain(member_uids):
                            continue

                        uid = mint_uid()
                        now = datetime.now(timezone.utc).isoformat()
                        synthesis = (
                            f"Temporal link ({days_apart}d): "
                            f"{pair_meta_a.get('content', '')[:80]} ↔ "
                            f"{pair_meta_b.get('content', '')[:80]}"
                        )

                        entry = {
                            "uid": uid,
                            "role": "chain",
                            "content": synthesis,
                            "turn": 0,
                            "ts": now,
                            "thread": pair_meta_a.get("thread", "unknown"),
                            "source_file": "dream",
                            "heading": f"temporal: {wk_a} ↔ {wk_b}",
                            "chunk_type": "chain",
                            "chain_type": "temporal_link",
                            "member_uids": member_uids,
                            "member_projects": list({
                                pair_meta_a.get("thread", ""),
                                pair_meta_b.get("thread", ""),
                            } - {""}),
                            "cross_project": (
                                pair_meta_a.get("thread") != pair_meta_b.get("thread")
                            ),
                            "temporal_distance_days": days_apart,
                            "similarity": round(pair_sim, 3),
                            "dream_pass": 0,
                            "dream_run": now,
                        }
                        embedding = embed_text(synthesis)
                        if embedding:
                            entry["embedding"] = embedding
                        temporal_links.append(entry)
                        self.existing_member_sets.append(frozenset(member_uids))

        print(f"[dream] Temporal reconnection: {len(temporal_links)} links found",
              file=sys.stderr)
        return temporal_links

    def _find_best_cross_pairs(
        self, indices_a: list[int], indices_b: list[int], top_k: int = 3
    ) -> list[tuple[dict, dict, float]]:
        """Find the most similar chunk pairs across two sets of indices."""
        import numpy as np
        if not indices_a or not indices_b:
            return []

        vecs_a = np.stack([self.index.vectors[i] for i in indices_a])
        vecs_b = np.stack([self.index.vectors[i] for i in indices_b])

        # Normalize
        norms_a = np.linalg.norm(vecs_a, axis=1, keepdims=True)
        norms_b = np.linalg.norm(vecs_b, axis=1, keepdims=True)
        norms_a[norms_a == 0] = 1
        norms_b[norms_b == 0] = 1
        vecs_a = vecs_a / norms_a
        vecs_b = vecs_b / norms_b

        # Cross-similarity matrix
        sim_matrix = vecs_a @ vecs_b.T
        flat = sim_matrix.flatten()
        top_indices = np.argsort(flat)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            row, col = divmod(idx, len(indices_b))
            meta_a = self.index.metadata[indices_a[row]]
            meta_b = self.index.metadata[indices_b[col]]
            sim = float(flat[idx])
            if sim > 0.5:  # minimum threshold for pairs
                results.append((meta_a, meta_b, sim))

        return results

    # ── Post-processing ────────────────────────────────────────────────

    def write_chains(self, chains: list[dict]) -> Path | None:
        """Write chain chunks to corpus."""
        if not chains:
            return None

        CHAINS_DIR.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = CHAINS_DIR / f"dream_{now}.jsonl"

        with open(out_path, "w") as f:
            for chain in chains:
                f.write(json.dumps(chain) + "\n")

        print(f"[dream] Wrote {len(chains)} chains to {out_path}", file=sys.stderr)
        return out_path

    def build_xrefs(self) -> dict[str, list[str]]:
        """Build cross-reference index from all chain chunks."""
        xrefs: dict[str, set[str]] = defaultdict(set)

        # Load all chain chunks
        all_chains = list(self.new_chains)
        if CHAINS_DIR.exists():
            for f in CHAINS_DIR.glob("*.jsonl"):
                for line in open(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if entry.get("role") == "chain":
                        all_chains.append(entry)

        for chain in all_chains:
            chain_uid = chain.get("uid", "")
            members = chain.get("member_uids", [])
            for member in members:
                xrefs[member].add(chain_uid)
                for other in members:
                    if other != member:
                        xrefs[member].add(other)

        result = {k: sorted(v) for k, v in xrefs.items()}

        XREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(XREFS_PATH, "w") as f:
            json.dump(result, f)

        print(f"[dream] Built xrefs: {len(result)} entries", file=sys.stderr)
        return result

    def save_state(self, stats: dict):
        """Save dream state for idempotency."""
        state = {
            "last_run": datetime.now(timezone.utc).isoformat(),
            "pass_count": stats.get("passes", 0),
            "chains_created": stats.get("chains_created", 0),
            "tokens_used": stats.get("tokens_used", 0),
            "converged": stats.get("converged", False),
            "corpus_mtime_at_run": self._corpus_mtime(),
        }
        DREAM_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DREAM_STATE_PATH, "w") as f:
            json.dump(state, f, indent=2)

    def _corpus_mtime(self) -> float:
        """Get the newest mtime across the corpus."""
        newest = 0.0
        if DEFAULT_CORPUS_DIR.exists():
            for f in DEFAULT_CORPUS_DIR.rglob("*.jsonl"):
                mt = f.stat().st_mtime
                if mt > newest:
                    newest = mt
        return newest

    def should_dream(self) -> bool:
        """Check if dreaming is needed (new corpus content since last dream)."""
        if not DREAM_STATE_PATH.exists():
            return True
        try:
            state = json.loads(DREAM_STATE_PATH.read_text())
            last_mtime = state.get("corpus_mtime_at_run", 0)
            current_mtime = self._corpus_mtime()
            return current_mtime > last_mtime
        except (json.JSONDecodeError, KeyError):
            return True

    # ── Report generation ──────────────────────────────────────────────

    def _load_all_chains(self) -> list[dict]:
        """Load ALL chain chunks from disk (not just current run)."""
        all_chains = []
        if CHAINS_DIR.exists():
            for f in sorted(CHAINS_DIR.glob("*.jsonl")):
                for line in open(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if entry.get("role") == "chain":
                        all_chains.append(entry)
        return all_chains

    def git_commit(self):
        """Git-commit the continuum data directory (like ingest does)."""
        data_dir = Path.home() / ".continuum"
        try:
            # Init if needed
            subprocess.run(
                ["git", "init"], cwd=str(data_dir),
                capture_output=True, timeout=10,
            )
            subprocess.run(
                ["git", "add", "-A"], cwd=str(data_dir),
                capture_output=True, timeout=30,
            )
            result = subprocess.run(
                ["git", "commit", "-m", f"dream: {len(self.new_chains)} chains"],
                cwd=str(data_dir),
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                print(f"[dream] Git committed", file=sys.stderr)
            # returncode 1 = nothing to commit, that's fine
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    def generate_report(self, stats: dict, temporal_links: list[dict]) -> dict:
        """Generate report data JSON from ALL chains (not just current run)."""
        all_chains = self._load_all_chains()

        # Categorize chains
        by_type = defaultdict(list)
        for chain in all_chains:
            by_type[chain.get("chain_type", "thematic")].append(chain)

        # Build report
        report = {
            "profile": {
                "name": "Symbolic",
                "generated": datetime.now(timezone.utc).isoformat(),
                "stats": {
                    "corpus_entries": len(self.all_metadata),
                    "embedded_entries": len(self.index) if self.index else 0,
                    "chains_created": len(all_chains),
                    "passes": stats.get("passes", 0),
                    "tokens_used": stats.get("tokens_used", 0),
                },
            },
            "chains": {
                "thematic": [self._chain_to_report(c) for c in by_type.get("thematic", [])],
                "causal": [self._chain_to_report(c) for c in by_type.get("causal", [])],
                "correction": [self._chain_to_report(c) for c in by_type.get("correction", [])],
                "orphan": [self._chain_to_report(c) for c in by_type.get("orphan", [])],
                "temporal_link": [self._chain_to_report(c) for c in by_type.get("temporal_link", [])],
            },
            "cross_project": [
                self._chain_to_report(c) for c in all_chains
                if c.get("cross_project")
            ],
            "unfinished": [
                self._chain_to_report(c) for c in by_type.get("orphan", [])
            ],
        }

        DREAM_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DREAM_REPORT_PATH, "w") as f:
            json.dump(report, f, indent=2)

        print(f"[dream] Report saved to {DREAM_REPORT_PATH}", file=sys.stderr)
        return report

    def _chain_to_report(self, chain: dict) -> dict:
        """Convert a chain chunk to report format."""
        # Resolve member details
        members = []
        uid_lookup = {m.get("uid"): m for m in self.all_metadata}
        for uid in chain.get("member_uids", []):
            meta = uid_lookup.get(uid, {})
            members.append({
                "uid": uid,
                "thread": meta.get("thread", ""),
                "ts": meta.get("ts", ""),
                "content": meta.get("content", "")[:200],
                "role": meta.get("role", ""),
            })

        return {
            "uid": chain.get("uid", ""),
            "type": chain.get("chain_type", "thematic"),
            "synthesis": chain.get("content", ""),
            "cross_project": chain.get("cross_project", False),
            "member_count": len(members),
            "members": members,
            "projects": chain.get("member_projects", []),
        }

    def print_report_markdown(self, report: dict):
        """Print a markdown summary of the dream report."""
        stats = report.get("profile", {}).get("stats", {})
        print(f"\n# Dream Report")
        print(f"\n**Corpus**: {stats.get('corpus_entries', 0)} entries "
              f"({stats.get('embedded_entries', 0)} embedded)")
        print(f"**Chains created**: {stats.get('chains_created', 0)}")
        print(f"**Clusters processed**: {stats.get('passes', 0)} "
              f"({stats.get('tokens_used', 0)} tokens)")

        chains = report.get("chains", {})

        for chain_type in ["correction", "orphan", "thematic", "causal", "temporal_link"]:
            items = chains.get(chain_type, [])
            if not items:
                continue
            label = chain_type.replace("_", " ").title()
            print(f"\n## {label} ({len(items)})")
            for item in items[:10]:
                xp = " [cross-project]" if item.get("cross_project") else ""
                projects = ", ".join(item.get("projects", []))
                print(f"- {item.get('synthesis', '')}{xp}")
                if projects:
                    print(f"  Projects: {projects}")

        cross = report.get("cross_project", [])
        if cross:
            print(f"\n## Cross-Project ({len(cross)})")
            for item in cross[:10]:
                print(f"- {item.get('synthesis', '')}")
                print(f"  Projects: {', '.join(item.get('projects', []))}")

        unfinished = report.get("unfinished", [])
        if unfinished:
            print(f"\n## Unfinished Business ({len(unfinished)})")
            for item in unfinished[:10]:
                print(f"- {item.get('synthesis', '')}")
