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


# ── Synthesis prompt (second stage — human-meaningful insights) ─────────

SYNTHESIS_SYSTEM_PROMPT = """You are a reflective analyst synthesizing patterns from a developer's work sessions. You take raw chain connections and compress them into human-meaningful insights — the kind that make someone say "oh right, I forgot about that" or "we keep making that mistake."

Your job is to produce KERNELS — dense, actionable insights organized by significance.

For each kernel, provide:
- type: one of [correction, orphan, pattern, stress, growth, question]
  - correction: "we keep doing X wrong, the fix is Y" — anti-pattern → correct approach
  - orphan: "this was started but never finished, and it matters because..." — unfinished business
  - pattern: "across projects/time, there's a recurring theme of..." — behavioral/workflow patterns
  - stress: "this situation causes friction/frustration/rework" — pain points
  - growth: "this approach worked well and should be repeated" — positive patterns to reinforce
  - question: "this remains unresolved and keeps coming up" — persistent open questions
- content: 1-2 sentences, specific (use names, projects, dates when available)
- importance: 1-10 (10 = "must not forget this", 1 = "mildly interesting")
- chain_refs: which chain UIDs this kernel synthesizes from
- cross_project: true if the insight spans multiple projects

Focus on:
- NON-OBVIOUS insights — not "code was edited" but "the same bug was fixed 3 times because the root cause was never addressed"
- HUMAN significance — would this matter to the person living this life, or just to a log parser?
- UNFINISHED BUSINESS — what was started and deserves attention?
- RECURRING CORRECTIONS — what keeps going wrong despite being fixed before?
- EMOTIONAL/BEHAVIORAL patterns — stress, avoidance, enthusiasm clusters

Do NOT include:
- Trivial file edits or routine tool use
- Chains that are just "these chunks are about the same project" with no deeper insight
- Anything that wouldn't pass the "so what?" test

Output valid JSON:
{
  "kernels": [
    {
      "type": "correction|orphan|pattern|stress|growth|question",
      "content": "specific, human-meaningful insight",
      "importance": 1-10,
      "chain_refs": ["«chain_uid1»", "«chain_uid2»"],
      "cross_project": true/false
    }
  ],
  "data_story": "2-3 sentence narrative of the overall arc — what's the person focused on, struggling with, excited about?",
  "top_insights": ["the 3 most important things to remember"]
}"""

SYNTHESIS_USER_TEMPLATE = """Here are {count} chains found by analyzing {corpus_size} corpus entries across {project_count} projects.

Synthesize these into human-meaningful kernels. Be specific and ruthless — only include insights that pass the "so what?" test.

IMPORTANT: The chains span different time periods. Some items marked as "planned" or "unfinished" in older chains may have been completed since. Cross-reference with the CURRENT PROJECT STATE below before claiming something is unfinished.

CURRENT PROJECT STATE:
{project_state}

CHAINS:
{chains}

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
        focus_project: str = "",
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
        self.focus_project = focus_project  # project-biased seeding

        # State
        self.all_metadata: list[dict] = []
        self.index: EmbeddingIndex | None = None
        self.existing_chain_uids: set[str] = set()
        self.existing_member_sets: list[frozenset[str]] = []
        self.new_chains: list[dict] = []
        self.pass_history: list[int] = []
        self.tokens_used: int = 0
        self.start_time: float = 0.0
        self._check_wake_up: bool = False  # set by daemon to enable user-activity detection

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
        """Check termination conditions, including user-activity wake-up."""
        elapsed = time.time() - self.start_time
        if elapsed >= self.max_wall_time:
            return True, f"wall time ({elapsed:.0f}s >= {self.max_wall_time}s)"
        if self.tokens_used >= self.max_llm_tokens:
            return True, f"token budget ({self.tokens_used} >= {self.max_llm_tokens})"
        if len(self.new_chains) >= self.max_chains:
            return True, f"chain cap ({len(self.new_chains)} >= {self.max_chains})"
        # Wake-up: stop if user activity or GPU contention detected
        if self._check_wake_up and elapsed > 30:
            wake_reason = self._detect_wake_up()
            if wake_reason:
                return True, f"woke up ({wake_reason})"
        return False, ""

    def _detect_wake_up(self) -> str | None:
        """Detect user activity or GPU contention.

        Uses session file writes (not process detection) because bridge
        keeps persistent claude PTY processes alive even when the user
        is away. A session JSONL write means someone is actually typing.

        Returns reason string if dream should stop, None otherwise.
        """
        import glob
        import os

        # 1. Session file activity — a write in the last 2 minutes means
        #    someone is actively using Claude (not just an idle PTY)
        cc_pattern = str(Path.home() / ".claude" / "projects" / "*" / "*.jsonl")
        session_files = glob.glob(cc_pattern)
        if session_files:
            newest = max(os.path.getmtime(f) for f in session_files)
            age_seconds = time.time() - newest
            if age_seconds < 120:  # active within 2 minutes
                return f"session activity ({age_seconds:.0f}s ago)"

        # 2. GPU contention — new process using VRAM that isn't Ollama
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,name",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if not line.strip():
                        continue
                    parts = line.split(",")
                    if len(parts) >= 2:
                        proc_name = parts[1].strip().lower()
                        # Ollama shows as full path or just binary name
                        basename = proc_name.rsplit("/", 1)[-1]
                        known_procs = {"python", "python3",
                                       "ollama_llama_server", "ollama"}
                        if basename not in known_procs:
                            return f"GPU contention ({basename})"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass  # no nvidia-smi = no GPU to contend

        return None

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
        """Check if a chunk has too little semantic content for integration."""
        role = meta.get("role", "")
        # Never filter kernels or chains
        if role in ("kernel", "chain"):
            return False
        content = meta.get("content", "").strip()
        if not content:
            return True
        # Pure tool-use bracket summaries
        if content.startswith("[") and content.endswith("]") and len(content) < 200:
            return True
        # Common filler
        if content in ("No response requested.",
                       "I have this context. Ready to continue."):
            return True
        # Tool result rejection boilerplate
        if "The user doesn't want to proceed with this tool use" in content and len(content) < 400:
            return True
        # Empty tool results
        if content.startswith("[{'tool_use_id':") and len(content) < 80:
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

            # Emotion affinity: similar emotional tone clusters together
            seed_emo = seed_meta.get("emotion_class", "neutral")
            meta_emo = meta.get("emotion_class", "neutral")
            if seed_emo == meta_emo and seed_emo != "neutral":
                score += 0.1
            seed_v = seed_meta.get("emotion_valence", 0.0)
            meta_v = meta.get("emotion_valence", 0.0)
            if abs(seed_v - meta_v) < 0.3:
                score += 0.05

            scored.append((meta, score))

        # Sort by combined score, take top cluster_size_max
        scored.sort(key=lambda x: x[1], reverse=True)
        cluster = [meta for meta, _ in scored[:self.cluster_size_max]]

        if len(cluster) >= self.cluster_size_min:
            return cluster
        return None

    def _select_focus_project(self) -> str:
        """Select a focus project for this dream cycle.

        If self.focus_project is set, use it. Otherwise auto-select:
        round-robin through all projects, prioritizing those with
        fewest existing chains (least-dreamed-about).
        """
        if self.focus_project:
            return self.focus_project

        # Count chains per project
        from collections import Counter
        chain_projects = Counter()
        for chain in self._load_all_chains():
            for p in chain.get("member_projects", []):
                chain_projects[p] += 1

        # Count corpus entries per project
        all_projects = set()
        for meta in self.all_metadata:
            t = meta.get("thread", "")
            if t and meta.get("role") not in ("chain", "kernel"):
                all_projects.add(t)

        if not all_projects:
            return ""

        # Score: projects with low chain coverage get priority
        # coverage = chains / corpus_entries for that project
        project_counts = Counter(
            m.get("thread", "") for m in self.all_metadata
            if m.get("role") not in ("chain", "kernel") and m.get("thread")
        )
        scored = []
        for proj in all_projects:
            corpus_count = project_counts.get(proj, 0)
            chain_count = chain_projects.get(proj, 0)
            if corpus_count == 0:
                continue
            coverage = chain_count / corpus_count
            scored.append((proj, coverage, corpus_count))

        # Sort by coverage ascending (least-dreamed first)
        scored.sort(key=lambda x: x[1])

        if scored:
            chosen = scored[0][0]
            if self.verbose:
                print(f"[dream] Auto-selected focus project: {chosen} "
                      f"(coverage: {scored[0][1]:.2%}, "
                      f"{scored[0][2]} entries)",
                      file=sys.stderr)
            return chosen
        return ""

    def _prepare_seed_pool(self) -> list[tuple[dict, any, float]]:
        """Build project-biased seed pool.

        Dream cycles start with a heavy bias toward the focus project.
        Cross-project integration remains — just not the default starting
        point. Ensures small but important projects (fence, family, health)
        get adequate dream attention despite low corpus volume.

        Returns list of (meta, vector, priority) tuples.
        Priority levels:
          3.0 — newly created chains (self-reinforcing, inserted during run)
          2.5 — focus project seeds (project-biased)
          2.0 — existing chain chunks
          1.5 — newest unchained corpus (last 7 days)
          1.0 — unchained corpus (other projects)
          0.5 — already-chained corpus
        """
        import random

        focus = self._select_focus_project()

        chained_uids = set()
        for ms in self.existing_member_sets:
            chained_uids.update(ms)
        for chain in self.new_chains:
            chained_uids.update(chain.get("member_uids", []))

        # Determine recency threshold (7 days)
        from datetime import datetime, timedelta, timezone
        recent_cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()

        focus_seeds = []
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
            thread = meta.get("thread", "")

            if role == "chain":
                chain_seeds.append((meta, self.index.vectors[i], 2.0))
            elif focus and (thread == focus or thread.startswith(focus + "/")):
                # Focus project gets highest corpus priority
                focus_seeds.append((meta, self.index.vectors[i], 2.5))
            elif uid not in chained_uids:
                if ts >= recent_cutoff:
                    fresh_seeds.append((meta, self.index.vectors[i], 1.5))
                else:
                    corpus_seeds.append((meta, self.index.vectors[i], 1.0))
            else:
                stale_seeds.append((meta, self.index.vectors[i], 0.5))

        # Shuffle each band
        for band in [focus_seeds, chain_seeds, fresh_seeds, corpus_seeds, stale_seeds]:
            random.shuffle(band)

        # Build pool: focus seeds first, then interleave chain+fresh+corpus
        result = list(focus_seeds)

        # Interleave remaining bands
        bands = [chain_seeds, fresh_seeds, corpus_seeds]
        indices = [0] * len(bands)
        while any(indices[j] < len(bands[j]) for j in range(len(bands))):
            for j in range(len(bands)):
                if indices[j] < len(bands[j]):
                    result.append(bands[j][indices[j]])
                    indices[j] += 1

        result.extend(stale_seeds)

        if self.verbose:
            print(f"[dream] Focus project: {focus or '(global)'}", file=sys.stderr)
            print(f"[dream] Seed pool: {len(focus_seeds)} focus, "
                  f"{len(chain_seeds)} chain, "
                  f"{len(fresh_seeds)} fresh, "
                  f"{len(corpus_seeds)} corpus, "
                  f"{len(stale_seeds)} stale",
                  file=sys.stderr)

        return result

    # ── LLM integration call ───────────────────────────────────────────

    def _format_chunks_for_prompt(self, cluster: list[dict],
                                   budget_chars: int = 6000) -> str:
        """Format cluster chunks for the integration prompt.

        Allocates character budget evenly across chunks. Small chunks
        use their full content. Large chunks get LLM-summarized via
        Ollama (fast, local) to preserve meaning without blind truncation.
        """
        n = len(cluster)
        per_chunk = budget_chars // max(n, 1)

        lines = []
        for meta in cluster:
            uid = meta.get("uid", "")
            thread = meta.get("thread", "?")
            ts = meta.get("ts", "")[:10]
            role = meta.get("role", "?")
            content = meta.get("content", "")
            heading = meta.get("heading", "")

            if len(content) <= per_chunk:
                display = content
            else:
                display = self._summarize_chunk(content, heading, per_chunk)

            prefix = f"[{uid} {thread} {ts} {role}]"
            if heading:
                prefix += f" {heading}"
            emo_class = meta.get("emotion_class", "")
            if emo_class and emo_class != "neutral":
                emo_v = meta.get("emotion_valence", 0.0)
                prefix += f" (emotion:{emo_class} v:{emo_v:+.1f})"
            lines.append(f"{prefix} {display}")
        return "\n".join(lines)

    def _summarize_chunk(self, content: str, heading: str, target_chars: int) -> str:
        """Summarize a large chunk via Ollama for integration context.

        Falls back to head+tail if the LLM call fails.
        """
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "Summarize the following content concisely. "
                     "Preserve key facts, decisions, errors, and outcomes. "
                     f"Keep under {target_chars} characters."},
                    {"role": "user", "content": content[:8000]},  # cap input to model context
                ],
                "stream": False,
                "options": {"temperature": 0.1},
            }
            result = subprocess.run(
                ["curl", "-s", "http://localhost:11434/api/chat",
                 "-d", json.dumps(payload)],
                capture_output=True, text=True, timeout=30,
            )
            response = json.loads(result.stdout)
            summary = response.get("message", {}).get("content", "")
            if summary:
                self.tokens_used += count_tokens(content[:8000]) + count_tokens(summary)
                return f"[summarized] {summary[:target_chars]}"
        except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError):
            pass

        # Fallback: head + tail
        half = target_chars // 2
        return content[:half] + "\n  [...]\n" + content[-half:]

    def _retrieve_cluster_context(self, cluster: list[dict]) -> str:
        """Retrieve additional corpus context for a cluster.

        Finds related chunks beyond the semantic neighbors already in
        the cluster, giving the integration LLM richer evidence.
        """
        try:
            from .retrieval import ContextRetriever
            from .index import load_index as _load_idx

            # Build query from cluster content (first 500 chars of each)
            query_parts = [m.get("content", "")[:200] for m in cluster[:4]]
            query = " ".join(query_parts)

            if not hasattr(self, '_retriever'):
                idx = _load_idx()
                self._retriever = ContextRetriever(sources=[], index=idx)

            result = self._retriever.retrieve(
                query=query,
                token_budget=2000,
                conversation_tail="",
                cull=False,
                exclude_roles=["chain", "kernel"],  # only real corpus evidence
            )

            if not result or not result.strip():
                return ""

            # Filter out chunks already in the cluster
            cluster_uids = {m.get("uid", "") for m in cluster}
            lines = []
            for line in result.strip().split("\n"):
                # Check if any cluster UID appears in this line
                skip = False
                for uid in cluster_uids:
                    if uid and uid in line:
                        skip = True
                        break
                if not skip:
                    lines.append(line)

            return "\n".join(lines[:15])  # cap at 15 lines
        except Exception as e:
            if self.verbose:
                print(f"  [warn] Cluster context retrieval failed: {e}",
                      file=sys.stderr)
            return ""

    def _call_llm(self, cluster: list[dict]) -> list[dict]:
        """Send a cluster to the dream model with enriched context."""
        chunks_text = self._format_chunks_for_prompt(cluster)

        # Enrich with retrieved context
        extra_context = self._retrieve_cluster_context(cluster)

        prompt_parts = [
            INTEGRATION_USER_TEMPLATE.format(
                count=len(cluster),
                chunks=chunks_text,
            )
        ]
        if extra_context:
            prompt_parts.append(
                f"\nRELATED CONTEXT (from elsewhere in the corpus):\n{extra_context}"
            )

        user_prompt = "\n".join(prompt_parts)

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

    # ── Synthesis pass (human-meaningful kernels) ────────────────────────

    def run_synthesis(self, all_chains: list[dict] | None = None) -> dict | None:
        """Synthesize chains into human-meaningful kernels.

        Model cascade: Ollama compresses chains into proto-kernels (free,
        many small batches), then Sonnet curates proto-kernels into final
        kernels (paid, small input). 90%+ reduction in paid API tokens.

        Pipeline:
        1. Local pre-synthesis (Ollama) — batch chains into groups of 15,
           extract proto-kernels from each batch. Every chain gets processed.
        2. Dedup proto-kernels
        3. Final synthesis (Sonnet) — curate proto-kernels into polished kernels
        4. Gap analysis — planned items with no chain activity
        """
        if all_chains is None:
            all_chains = self._load_all_chains()

        if not all_chains:
            print(f"[dream] No chains to synthesize", file=sys.stderr)
            return None

        # Step 1: Local pre-synthesis (free, unlimited)
        proto_kernels = self._run_local_presynthesis(all_chains)

        # Step 2: Final synthesis with Sonnet (paid, small input)
        if self.focus_project:
            focus = self.focus_project
            focus_protos = [p for p in proto_kernels
                           if any(focus == pr or pr.startswith(focus + "/")
                                  for pr in p.get("projects", []))]
            if focus_protos:
                result_focus = self._run_synthesis_pass(focus_protos, "focus",
                                                        from_protos=True)
            else:
                result_focus = None
            result_all = self._run_synthesis_pass(proto_kernels, "global",
                                                   from_protos=True)
            gap_kernels = self._run_gap_analysis(all_chains)

            merged = self._merge_synthesis(result_focus, result_all)
            if gap_kernels and merged:
                merged["kernels"] = merged.get("kernels", []) + gap_kernels
            elif gap_kernels:
                merged = {"kernels": gap_kernels, "top_insights": [], "data_story": ""}
            return merged

        result = self._run_synthesis_pass(proto_kernels, "global", from_protos=True)
        gap_kernels = self._run_gap_analysis(all_chains)
        if gap_kernels and result:
            result["kernels"] = result.get("kernels", []) + gap_kernels
        return result

    # ── Local pre-synthesis (Ollama, free) ─────────────────────────────

    LOCAL_PRESYNTHESIS_PROMPT = """You are compressing chain connections into proto-kernels — rough insights for further curation.

Given these chains, extract the most significant insights. For each:
- type: correction | orphan | pattern | stress | growth | question
- content: 1-2 sentences, specific
- importance: 1-10
- projects: which projects are involved

Rules:
- Extract 3-8 proto-kernels per batch
- Skip trivial chains ("files were edited", "code was read")
- Focus on corrections, unfinished work, recurring patterns, pain points
- Be specific — names, projects, dates when available

Output valid JSON: {"proto_kernels": [{"type": "...", "content": "...", "importance": N, "projects": [...]}]}"""

    def _run_local_presynthesis(self, all_chains: list[dict]) -> list[dict]:
        """Batch chains through Ollama to produce proto-kernels.

        Processes every chain in small batches (free, local).
        Returns deduplicated proto-kernels for Sonnet to curate.
        """
        import random

        BATCH_SIZE = 15
        random.shuffle(all_chains)

        # Build batches
        batches = []
        for i in range(0, len(all_chains), BATCH_SIZE):
            batch = all_chains[i:i + BATCH_SIZE]
            batches.append(batch)

        print(f"[dream] Local pre-synthesis: {len(all_chains)} chains → "
              f"{len(batches)} batches of ~{BATCH_SIZE} (Ollama, free)",
              file=sys.stderr)

        all_protos = []
        presynth_start = time.time()

        for bi, batch in enumerate(batches):
            # Check termination
            stop, reason = self._should_stop()
            if stop:
                print(f"[dream] Pre-synthesis stopped: {reason}", file=sys.stderr)
                break

            # Format batch
            lines = []
            for chain in batch:
                ctype = chain.get("chain_type", "?")
                content = chain.get("content", "")[:200]
                projects = ", ".join(chain.get("member_projects", []))
                xp = " [cross-project]" if chain.get("cross_project") else ""
                lines.append(f"({ctype}) {content} [projects: {projects}]{xp}")

            user_prompt = f"Compress these {len(batch)} chains into proto-kernels:\n\n" + "\n".join(lines)

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.LOCAL_PRESYNTHESIS_PROMPT},
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
                    capture_output=True, text=True, timeout=60,
                )
                response = json.loads(result.stdout)
                content = response.get("message", {}).get("content", "")
                parsed = json.loads(content)
                protos = parsed.get("proto_kernels", [])
                if isinstance(protos, list):
                    all_protos.extend(protos)
            except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError):
                pass

            # Progress every 10 batches
            if (bi + 1) % 10 == 0 or bi == len(batches) - 1:
                elapsed = time.time() - presynth_start
                print(f"[dream] Pre-synthesis: {bi+1}/{len(batches)} batches, "
                      f"{len(all_protos)} proto-kernels, {elapsed:.0f}s",
                      file=sys.stderr)

        # Dedup proto-kernels by content similarity
        seen = set()
        unique = []
        for p in all_protos:
            key = p.get("content", "").strip().lower()[:60]
            if key not in seen and len(key) > 10:
                seen.add(key)
                unique.append(p)

        # Sort by importance
        unique.sort(key=lambda p: -(p.get("importance", 5)))

        elapsed = time.time() - presynth_start
        print(f"[dream] Pre-synthesis complete: {len(all_protos)} raw → "
              f"{len(unique)} unique proto-kernels ({elapsed:.0f}s)",
              file=sys.stderr)

        return unique

    def _merge_synthesis(self, focus: dict | None, broad: dict | None) -> dict | None:
        """Merge focus and broad synthesis results, deduplicating kernels."""
        if not focus and not broad:
            return None
        if not focus:
            return broad
        if not broad:
            return focus

        # Use broad as base, add unique focus kernels
        merged = dict(broad)
        focus_kernels = focus.get("kernels", [])
        broad_kernels = broad.get("kernels", [])

        # Dedup by content similarity (exact substring match)
        broad_contents = {k.get("content", "").lower()[:80] for k in broad_kernels}
        for fk in focus_kernels:
            fc = fk.get("content", "").lower()[:80]
            if fc not in broad_contents:
                broad_kernels.append(fk)

        merged["kernels"] = broad_kernels
        # Prefer focus data_story if it exists
        if focus.get("data_story"):
            merged["data_story"] = focus["data_story"]
        # Merge top_insights
        focus_insights = focus.get("top_insights", [])
        broad_insights = broad.get("top_insights", [])
        seen = set()
        combined = []
        for i in focus_insights + broad_insights:
            key = i[:50].lower()
            if key not in seen:
                combined.append(i)
                seen.add(key)
        merged["top_insights"] = combined[:5]

        return merged

    def _run_synthesis_pass(self, items: list[dict], label: str,
                            from_protos: bool = False) -> dict | None:
        """Run final synthesis via Sonnet on proto-kernels or raw chains."""
        if not items:
            return None

        # Format items for prompt
        item_lines = []
        projects = set()

        if from_protos:
            # Proto-kernels — already compressed, much smaller
            for p in items[:100]:  # cap at 100 proto-kernels
                ptype = p.get("type", "?")
                content = p.get("content", "")
                importance = p.get("importance", 5)
                pprojects = p.get("projects", [])
                if isinstance(pprojects, list):
                    projects.update(pprojects)
                item_lines.append(
                    f"({ptype}, importance={importance}) {content} "
                    f"[projects: {', '.join(pprojects) if isinstance(pprojects, list) else str(pprojects)}]"
                )
        else:
            # Raw chains (fallback)
            MAX_SYNTHESIS_CHARS = 150_000
            total_chars = sum(len(c.get("content", "")) for c in items)
            if total_chars > MAX_SYNTHESIS_CHARS:
                type_priority = {"correction": 0, "orphan": 1, "causal": 2,
                                 "thematic": 3, "temporal_link": 4}
                items = sorted(items,
                               key=lambda c: (type_priority.get(c.get("chain_type"), 5),
                                              c.get("ts", "")))
                trimmed = []
                char_count = 0
                for c in items:
                    char_count += len(c.get("content", "")) + 100
                    if char_count > MAX_SYNTHESIS_CHARS:
                        break
                    trimmed.append(c)
                items = trimmed

            for chain in items:
                uid = chain.get("uid", "?")
                ctype = chain.get("chain_type", "?")
                content = chain.get("content", "")
                member_count = len(chain.get("member_uids", []))
                chain_projects = chain.get("member_projects", [])
                projects.update(chain_projects)
                xp = " [cross-project]" if chain.get("cross_project") else ""
                item_lines.append(
                    f"[{uid}] ({ctype}) {content} "
                    f"[{member_count} members, projects: {', '.join(chain_projects)}]{xp}"
                )

        project_state = self._gather_project_state()

        # For focus pass: inject the full CLAUDE.md so synthesis can
        # compare chains against planned/documented work and flag
        # items that were planned but never generated session activity
        focus_context = ""
        if label == "focus" and self.focus_project:
            focus_context = self._load_focus_context()

        input_type = "proto-kernels" if from_protos else "chains"
        user_prompt = SYNTHESIS_USER_TEMPLATE.format(
            count=len(items),
            corpus_size=len(self.all_metadata),
            project_count=len(projects),
            project_state=project_state,
            chains="\n".join(item_lines),
        )

        if focus_context:
            user_prompt += (
                f"\n\nFOCUS PROJECT CONTEXT ({self.focus_project}):\n"
                f"Below are the project's CLAUDE.md files and saved plans. "
                f"These represent documented intentions and detailed designs. "
                f"Compare the {input_type} above against this context:\n"
                f"- Flag planned work with NO corresponding {input_type} as orphans\n"
                f"- Plans marked [PLAN] represent invested design effort — "
                f"unstarted plans are higher-priority orphans than passing mentions\n"
                f"- Check Direction/Pending sections for active priorities vs completed work\n\n"
                f"{focus_context}"
            )

        synth_start = time.time()
        prompt_chars = len("\n".join(item_lines))
        print(f"[dream] Running synthesis ({label}) on {len(items)} {input_type} "
              f"across {len(projects)} projects "
              f"({prompt_chars:,} chars prompt)...", file=sys.stderr)

        # Use claude --print for synthesis (like hackathon)
        synthesis_model = get_model("compress", self.config)
        full_prompt = SYNTHESIS_SYSTEM_PROMPT + "\n\n" + user_prompt + \
            "\n\nRespond with ONLY valid JSON, no markdown fences."

        import os
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

        try:
            result = subprocess.run(
                ["claude", "--print", "--model", synthesis_model],
                input=full_prompt, env=env,
                capture_output=True, text=True, timeout=180,
            )
            content = result.stdout.strip()
            if result.returncode != 0:
                print(f"[dream] Synthesis error (exit {result.returncode}): "
                      f"{result.stderr[:200]}", file=sys.stderr)
                return None
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"[dream] Synthesis failed: {e}", file=sys.stderr)
            return None

        # Strip markdown fences if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        try:
            synthesis = json.loads(content)
            kernels = synthesis.get("kernels", [])

            # Clean hallucinated chain_refs — LLM invents plausible UIDs
            known_uids = {m.get("uid", "") for m in self.all_metadata}
            for kernel in kernels:
                refs = kernel.get("chain_refs", [])
                cleaned = [r for r in refs if r in known_uids
                           or r.replace("«", "").replace("»", "") in
                           {u.replace("«", "").replace("»", "") for u in known_uids}]
                if len(refs) != len(cleaned):
                    if self.verbose:
                        print(f"[dream] Cleaned {len(refs) - len(cleaned)} "
                              f"hallucinated refs from kernel", file=sys.stderr)
                kernel["chain_refs"] = cleaned

            synth_elapsed = time.time() - synth_start
            print(f"[dream] Synthesized {len(kernels)} kernels in {synth_elapsed:.0f}s "
                  f"(model: {synthesis_model})", file=sys.stderr)

            # Write kernels as first-class corpus entries
            if kernels and not self.dry_run:
                self._write_kernel_chunks(kernels)

            return synthesis
        except json.JSONDecodeError as e:
            print(f"[dream] Synthesis JSON parse failed: {e}", file=sys.stderr)
            print(f"[dream] Content: {content[:300]}", file=sys.stderr)
            return None

    def _gather_project_state(self) -> str:
        """Read current State sections from all project CLAUDE.md files.

        Gives the synthesis model ground truth about what's implemented
        vs planned, preventing stale claims about unfinished work.
        """
        import glob
        projects_dir = Path.home() / "projects"
        state_lines = []

        for claude_md in sorted(glob.glob(str(projects_dir / "**/CLAUDE.md"), recursive=True)):
            path = Path(claude_md)
            # Derive project name from path
            rel = path.parent.relative_to(projects_dir)
            project = str(rel) if str(rel) != "." else "home"

            try:
                text = path.read_text()
            except OSError:
                continue

            # Extract ## State section (up to next ## heading)
            import re
            match = re.search(r'^## State\s*\n(.*?)(?=^## |\Z)',
                              text, re.MULTILINE | re.DOTALL)
            if match:
                state = match.group(1).strip()
                # Limit per project to keep prompt manageable
                if len(state) > 500:
                    state = state[:500] + "..."
                state_lines.append(f"[{project}] {state}")

        return "\n\n".join(state_lines) if state_lines else "(no project state available)"

    def _run_gap_analysis(self, all_chains: list[dict]) -> list[dict]:
        """Find planned items with zero chain activity.

        Scans CLAUDE.md Direction/Pending sections and plans/*.md for
        named items (bullet points, headings). Checks each against
        all chain content. Items with no matching chains become orphan
        kernels — the absence of activity is the signal.

        No LLM needed — pure text matching.
        """
        import glob
        import re
        projects_dir = Path.home() / "projects"
        focus = self.focus_project

        # Scope scan to focus project if set
        if focus:
            scan_dir = projects_dir / focus
            if not scan_dir.exists():
                scan_dir = projects_dir  # fallback to all
        else:
            scan_dir = projects_dir

        # Collect all chain content for matching
        chain_text = " ".join(c.get("content", "") for c in all_chains).lower()

        # Also collect all corpus content for broader matching
        corpus_text = " ".join(
            m.get("content", "") for m in self.all_metadata
            if m.get("role") not in ("chain", "kernel")
        ).lower()

        # Extract items from ALL markdown: CLAUDE.md (all sections) + plans
        planned_items: list[tuple[str, str, str]] = []  # (item, source, project)

        # Noise patterns to skip
        noise_patterns = re.compile(
            r'^\|.*\|$'           # table rows
            r'|^```'              # code fences
            r'|^---'              # horizontal rules
            r'|^>\s'              # blockquotes (usually examples)
            r'|^\*\*Date\*\*'     # metadata lines
            r'|^\*\*Status\*\*'
            r'|^\*\*Phase\*\*'
            r'|^\*\*Horizon\*\*'
            r'|^Co-Authored'      # commit signatures
        )

        for claude_md in sorted(glob.glob(str(scan_dir / "**/CLAUDE.md"), recursive=True)):
            path = Path(claude_md)
            try:
                text = path.read_text()
            except OSError:
                continue

            rel = path.parent.relative_to(projects_dir)
            project = str(rel) if str(rel) != "." else "home"

            # Extract bullet points from ALL sections
            current_section = ""
            for line in text.split("\n"):
                if line.startswith("## "):
                    current_section = line.lstrip("# ").strip()
                    continue
                line = line.strip().lstrip("- [x] ").lstrip("- [ ] ").lstrip("- ").strip()
                if len(line) < 15 or line.startswith("#"):
                    continue
                if noise_patterns.search(line):
                    continue
                planned_items.append((line[:200], f"{project}/CLAUDE.md#{current_section}", project))

        # Extract from plans — titles and all substantive lines
        for plan_file in sorted(glob.glob(str(scan_dir / "**/plans/*.md"), recursive=True)):
            path = Path(plan_file)
            try:
                text = path.read_text()
            except OSError:
                continue

            rel = path.relative_to(projects_dir)
            project = str(rel).split("/")[0]

            for line in text.split("\n"):
                if line.startswith("# "):
                    planned_items.append((line.lstrip("# ").strip(), str(rel), project))
                    continue
                line = line.strip().lstrip("- ").strip()
                if len(line) < 20 or line.startswith("#") or line.startswith("```"):
                    continue
                if noise_patterns.search(line):
                    continue
                planned_items.append((line[:200], str(rel), project))

        # Also scan auto-memory files
        memory_dir = Path.home() / ".claude" / "projects" / "-home-symbolic-projects" / "memory"
        if memory_dir.is_dir():
            for mem_file in sorted(memory_dir.glob("*.md")):
                try:
                    text = mem_file.read_text()
                except OSError:
                    continue
                for line in text.split("\n"):
                    line = line.strip().lstrip("- ").strip()
                    if len(line) < 20 or line.startswith("#") or line.startswith("---"):
                        continue
                    if noise_patterns.search(line):
                        continue
                    planned_items.append((line[:200], f"memory/{mem_file.name}", "memory"))

        if not planned_items:
            return []

        # Check each planned item against chain content
        gaps = []
        seen = set()
        for item, source, project in planned_items:
            # Extract key terms (3+ char words, skip common words)
            words = set(re.findall(r'\b[a-z_]{4,}\b', item.lower()))
            stop = {"this", "that", "with", "from", "have", "been", "will",
                    "should", "could", "each", "into", "when", "what", "which",
                    "their", "there", "than", "then", "also", "some", "more",
                    "other", "about", "first", "next"}
            key_terms = words - stop

            if len(key_terms) < 2:
                continue

            # Check key terms against chain content AND corpus content
            chain_matches = sum(1 for t in key_terms if t in chain_text)
            corpus_matches = sum(1 for t in key_terms if t in corpus_text)
            chain_coverage = chain_matches / len(key_terms) if key_terms else 0
            corpus_coverage = corpus_matches / len(key_terms) if key_terms else 0

            # Gap: documented but low chain coverage
            # - Pure gap: low chain AND low corpus coverage (never discussed)
            # - Discussed gap: low chain but high corpus (discussed, never acted on)
            if chain_coverage < 0.3:
                dedup_key = item[:50].lower()
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                if corpus_coverage >= 0.5:
                    gap_type = "discussed but no integration chains"
                    importance = 8  # higher — was discussed, still no action
                else:
                    gap_type = "no corresponding session activity or chains"
                    importance = 6

                gaps.append({
                    "type": "orphan",
                    "content": f"[GAP] {item} — documented in {source} but {gap_type}.",
                    "importance": importance,
                    "chain_refs": [],
                    "cross_project": False,
                    "evidence_verdict": "gap_analysis",
                    "evidence_reason": (
                        f"Source: {source}, "
                        f"chain coverage: {chain_matches}/{len(key_terms)} ({chain_coverage:.0%}), "
                        f"corpus coverage: {corpus_matches}/{len(key_terms)} ({corpus_coverage:.0%})"
                    ),
                    "evidence_confidence": 1.0 - chain_coverage,
                })

        # Cap gaps: prioritize "discussed but no chains" (importance 8) over pure gaps (6)
        # Tiebreak by specificity: more key terms + lower coverage = more specific gap
        MAX_GAPS = 25
        gaps.sort(key=lambda g: (-g["importance"], -g["evidence_confidence"]))
        if len(gaps) > MAX_GAPS:
            gaps = gaps[:MAX_GAPS]

        if gaps:
            discussed = sum(1 for g in gaps if g["importance"] >= 8)
            print(f"[dream] Gap analysis: {len(gaps)} gaps "
                  f"({discussed} discussed-but-unacted, {len(gaps) - discussed} pure gaps)",
                  file=sys.stderr)

            # Write gap kernels to corpus
            if not self.dry_run:
                self._write_kernel_chunks(gaps)

        return gaps

    def _load_focus_context(self) -> str:
        """Load CLAUDE.md files and plans for the focus project.

        CLAUDE.md Direction/Pending = active priorities.
        plans/*.md = detailed designs someone invested time writing.
        Both are stronger orphan signals than passing session mentions.
        """
        import glob
        projects_dir = Path.home() / "projects"
        focus_dir = projects_dir / self.focus_project

        parts = []

        # CLAUDE.md files (project + sub-projects)
        for claude_md in sorted(glob.glob(str(focus_dir / "**/CLAUDE.md"), recursive=True)):
            path = Path(claude_md)
            try:
                text = path.read_text()
                rel = path.parent.relative_to(projects_dir)
                parts.append(f"--- {rel}/CLAUDE.md ---\n{text[:3000]}")
            except OSError:
                continue

        # Plans (project + sub-projects)
        for plan_file in sorted(glob.glob(str(focus_dir / "**/plans/*.md"), recursive=True)):
            path = Path(plan_file)
            try:
                text = path.read_text()
                rel = path.relative_to(projects_dir)
                parts.append(f"--- {rel} [PLAN] ---\n{text[:2000]}")
            except OSError:
                continue

        # Also check for auto-memory files (Claude Code's accumulated memory)
        memory_dirs = [
            Path.home() / ".claude" / "projects" / f"-home-symbolic-projects-{self.focus_project.replace('/', '-')}" / "memory",
            Path.home() / ".claude" / "projects" / f"-home-symbolic-projects" / "memory",
        ]
        for mem_dir in memory_dirs:
            if mem_dir.is_dir():
                for mem_file in sorted(mem_dir.glob("*.md")):
                    try:
                        text = mem_file.read_text()
                        parts.append(f"--- memory/{mem_file.name} ---\n{text[:2000]}")
                    except OSError:
                        continue

        result = "\n\n".join(parts)
        if result and self.verbose:
            print(f"[dream] Loaded {len(parts)} context files for focus project "
                  f"({self.focus_project})", file=sys.stderr)
        return result

    def _write_kernel_chunks(self, kernels: list[dict]):
        """Write synthesized kernels as first-class corpus entries.

        Kernels are higher-value than raw chains — they carry a 'kernel'
        role and get embedded for retrieval with provenance back to chains.
        """
        CHAINS_DIR.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc)
        out_path = CHAINS_DIR / f"kernels_{now.strftime('%Y%m%d_%H%M%S')}.jsonl"

        entries = []
        for kernel in kernels:
            uid = mint_uid()
            content = kernel.get("content", "")
            entry = {
                "uid": uid,
                "role": "kernel",
                "content": content,
                "turn": 0,
                "ts": now.isoformat(),
                "thread": "dream",
                "source_file": "dream:synthesis",
                "heading": f"kernel: {content[:60]}",
                "chunk_type": "kernel",
                "kernel_type": kernel.get("type", "pattern"),
                "importance": kernel.get("importance", 5),
                "chain_refs": kernel.get("chain_refs", []),
                "cross_project": kernel.get("cross_project", False),
                "dream_run": now.isoformat(),
            }
            embedding = embed_text(content)
            if embedding:
                entry["embedding"] = embedding
            entries.append(entry)

        with open(out_path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        print(f"[dream] Wrote {len(entries)} kernel chunks to {out_path}",
              file=sys.stderr)

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

    # ── Evidence validation ────────────────────────────────────────────

    def validate_kernels(self, synthesis: dict) -> dict:
        """Validate each kernel by retrieving evidence and LLM-judging it.

        For each kernel:
        1. Retrieve corpus evidence matching the claim
        2. Ask Haiku: does the evidence support, contradict, or leave
           the claim unresolved?
        3. Annotate kernel with verdict + confidence + reason
        """
        kernels = synthesis.get("kernels", [])
        if not kernels:
            return synthesis

        validate_start = time.time()
        print(f"[dream] Validating {len(kernels)} kernels "
              f"(retrieve + Haiku judge each)...", file=sys.stderr)

        try:
            from .retrieval import ContextRetriever
            from .index import load_index as _load_idx

            if not hasattr(self, '_retriever'):
                idx = _load_idx()
                self._retriever = ContextRetriever(sources=[], index=idx)
        except Exception as e:
            print(f"[dream] Could not load retriever for validation: {e}",
                  file=sys.stderr)
            return synthesis

        judge_model = get_model("cull", self.config)  # Haiku
        import os
        import re
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

        for i, kernel in enumerate(kernels):
            content = kernel.get("content", "")
            if not content:
                continue

            # 1. Retrieve evidence (exclude dream-generated content to prevent
            #    circular validation — dream output validating itself)
            try:
                evidence = self._retriever.retrieve(
                    query=content,
                    token_budget=3000,
                    conversation_tail="",
                    cull=False,
                    exclude_roles=["chain", "kernel"],
                )
            except Exception:
                evidence = ""

            if not evidence or not evidence.strip():
                kernel["evidence_verdict"] = "insufficient"
                kernel["evidence_reason"] = "no matching corpus evidence (excluding dream output)"
                kernel["evidence_confidence"] = 0.0
                kernel["evidence_uids"] = []
                continue

            evidence_uids = re.findall(r'[«]([a-f0-9]{6})[»]', evidence)

            # 2. LLM judge
            judge_prompt = (
                "You are an evidence validator. Given a CLAIM and EVIDENCE from a corpus, "
                "determine whether the evidence supports, contradicts, or is insufficient "
                "to judge the claim.\n\n"
                f"CLAIM: {content}\n\n"
                f"EVIDENCE:\n{evidence[:4000]}\n\n"
                "Respond with ONLY valid JSON:\n"
                '{"verdict": "supported"|"contradicted"|"insufficient", '
                '"reason": "one sentence explanation", '
                '"confidence": 0.0-1.0}'
            )

            try:
                result = subprocess.run(
                    ["claude", "--print", "--model", judge_model],
                    input=judge_prompt, env=env,
                    capture_output=True, text=True, timeout=30,
                )
                judge_content = result.stdout.strip()
                if "```json" in judge_content:
                    judge_content = judge_content.split("```json")[1].split("```")[0]
                elif "```" in judge_content:
                    judge_content = judge_content.split("```")[1].split("```")[0]

                verdict = json.loads(judge_content)
                kernel["evidence_verdict"] = verdict.get("verdict", "insufficient")
                kernel["evidence_reason"] = verdict.get("reason", "")
                kernel["evidence_confidence"] = verdict.get("confidence", 0.5)
                kernel["evidence_uids"] = evidence_uids

                v = kernel["evidence_verdict"]
                c = kernel["evidence_confidence"]
                elapsed = time.time() - validate_start
                print(f"[dream] Validating {i+1}/{len(kernels)} "
                      f"({elapsed:.0f}s) {v} ({c:.1f}) "
                      f"-- {content[:50]}", file=sys.stderr)

            except (subprocess.TimeoutExpired, json.JSONDecodeError,
                    FileNotFoundError) as e:
                kernel["evidence_verdict"] = "insufficient"
                kernel["evidence_reason"] = f"judge failed: {e}"
                kernel["evidence_confidence"] = 0.0
                kernel["evidence_uids"] = evidence_uids

        validate_elapsed = time.time() - validate_start
        verdicts = [k.get("evidence_verdict", "?") for k in kernels]
        from collections import Counter
        vc = Counter(verdicts)
        print(f"[dream] Validation: {vc.get('supported', 0)} supported, "
              f"{vc.get('insufficient', 0)} insufficient, "
              f"{vc.get('contradicted', 0)} contradicted "
              f"({validate_elapsed:.0f}s, {validate_elapsed/max(len(kernels),1):.1f}s/kernel)",
              file=sys.stderr)

        return synthesis

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

    def generate_report(self, stats: dict, temporal_links: list[dict],
                        synthesis: dict | None = None) -> dict:
        """Generate report data JSON from ALL chains + synthesis."""
        all_chains = self._load_all_chains()

        # Categorize chains
        by_type = defaultdict(list)
        for chain in all_chains:
            by_type[chain.get("chain_type", "thematic")].append(chain)

        # Build report
        report = {
            "profile": {
                "name": "Symbolic",
                "generated": datetime.now().astimezone().isoformat(),
                "stats": {
                    "corpus_entries": len(self.all_metadata),
                    "embedded_entries": len(self.index) if self.index else 0,
                    "chains_created": len(all_chains),
                    "passes": stats.get("passes", 0),
                    "tokens_used": stats.get("tokens_used", 0),
                },
            },
            # Synthesis results (human-meaningful — primary content)
            "synthesis": synthesis or {},
            "kernels": (synthesis or {}).get("kernels", []),
            "data_story": (synthesis or {}).get("data_story", ""),
            "top_insights": (synthesis or {}).get("top_insights", []),
            # Raw chains (drill-down backing data)
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

        # Write current report (full content for live drill-down)
        with open(DREAM_REPORT_PATH, "w") as f:
            json.dump(report, f, indent=2)

        # Preserve versioned copy — slim (no member content, just metadata)
        # Full content is in the corpus; versioned reports are for history/diffs
        reports_dir = DREAM_REPORT_PATH.parent / "dream_reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().astimezone().strftime("%Y-%m-%dT%H%M%S")
        versioned = reports_dir / f"dream_report_{ts}.json"

        slim = dict(report)
        slim_chains = {}
        for ctype, arr in report.get("chains", {}).items():
            slim_chains[ctype] = []
            for chain in arr:
                slim_chain = dict(chain)
                # Keep member UIDs and metadata, drop full content
                slim_chain["members"] = [
                    {"uid": m.get("uid"), "thread": m.get("thread"),
                     "ts": m.get("ts"), "role": m.get("role")}
                    for m in chain.get("members", [])
                ]
                slim_chains[ctype].append(slim_chain)
        slim["chains"] = slim_chains
        # Also slim cross_project and unfinished
        for key in ("cross_project", "unfinished"):
            if key in slim:
                slim[key] = [
                    {**c, "members": [
                        {"uid": m.get("uid"), "thread": m.get("thread"),
                         "ts": m.get("ts"), "role": m.get("role")}
                        for m in c.get("members", [])
                    ]} for c in slim[key]
                ]

        with open(versioned, "w") as f:
            json.dump(slim, f)

        slim_size = versioned.stat().st_size / 1024
        full_size = DREAM_REPORT_PATH.stat().st_size / 1024 / 1024
        print(f"[dream] Report saved: {full_size:.1f}MB live, "
              f"{slim_size:.0f}KB versioned ({versioned.name})", file=sys.stderr)
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
                "content": meta.get("content", ""),
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
