#!/usr/bin/env python3
"""Mine ground truth queries from real frustration moments in session history.

Scans user messages for signals that the LLM forgot context, got something wrong,
or the user had to repeat information from a previous session. Each hit becomes
a candidate ground truth entry for the autoresearch benchmark.

Two mining modes:
1. Frustration mining — find moments where the human corrected the AI
2. Technical recall — find unique discoveries that should be retrievable

Usage:
    python mine_ground_truth.py                    # mine and print candidates
    python mine_ground_truth.py --write            # append to ground_truth.json
    python mine_ground_truth.py --emotion          # also use emotion metadata
    python mine_ground_truth.py --profanity        # include profanity signals
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path

_CONTINUUM_DIR = Path(__file__).resolve().parent
CORPUS_DIR = Path.home() / ".continuum" / "corpus"
GROUND_TRUTH_PATH = _CONTINUUM_DIR / "ground_truth.json"


# ── Frustration patterns ──────────────────────────────────────────────

FRUSTRATION_PATTERNS = [
    # Forgetting / missing context
    (r"i already (told|explained|said|showed|mentioned)", "forgot_context"),
    (r"(you|we) (already|just) (discussed|covered|fixed|did|went over|talked about)", "forgot_context"),
    (r"you (forgot|don't remember|lost|dropped)", "forgot_context"),
    (r"you don't (have|know|remember)", "missing_context"),
    (r"you('re| are) missing (the |)context", "missing_context"),
    (r"in the (other|previous|last|earlier) session", "cross_session"),
    (r"we (covered|did|fixed|discussed|established) (this|that) (before|earlier|already|last time)", "cross_session"),
    (r"i (already |)explained this", "forgot_context"),
    (r"remember (when|that|how) (we|i|you)", "cross_session"),
    (r"you should (already |)(know|have|remember)", "missing_context"),

    # Corrections — user had to fix the AI
    (r"that's (wrong|not right|incorrect|backwards|not what i)", "correction"),
    (r"no[\.,!] (not |)(that|the|it)", "correction"),
    (r"i (asked|told) you (not to|never to|to never|to always|to stop)", "correction"),
    (r"i said (not to|never|don't|stop|always)", "correction"),
    (r"(stop|quit|don't) doing (that|this|it)", "correction"),
    (r"how many times (do i|have i)", "repeated_correction"),
    (r"i keep (telling|saying|asking|having to)", "repeated_correction"),

    # Frustration / exasperation
    (r"this is (frustrating|annoying|exhausting|painful)", "frustration"),
    (r"(why|how) (did|does|is|are) (it|this|that) (still|again|keep)", "frustration"),
    (r"i (just|literally) (told|said|showed|explained)", "frustration"),
    (r"you (just|literally) (did|made|broke|ignored)", "frustration"),
    (r"(again|still)\?{0,1}$", "frustration"),
]

PROFANITY_PATTERNS = [
    (r"\b(fuck|shit|damn|hell|crap|wtf|ffs|dammit)\b", "profanity"),
    (r"\b(fucking|shitty|goddamn)\b", "profanity"),
]

# Positive profanity — excitement, not frustration. Exclude from ground truth.
POSITIVE_PROFANITY_RE = re.compile(
    r"(fucking (beautiful|dope|amazing|awesome|great|perfect|incredible|good|love|nice)|"
    r"(hell|damn) (yes|yeah)|"
    r"holy (shit|crap|fuck).*(!|nailed|amazing|awesome))",
    re.IGNORECASE,
)

EXCLAMATION_PATTERNS = [
    # Multiple exclamation marks with surrounding words (not bare !!)
    (r"\w+[!]{2,}", "emphasis"),
    # All caps frustration with exclamation
    (r"[A-Z]{4,}[!]", "emphasis"),
    # Frustrated question marks
    (r"(why|what|how)\?{2,}", "emphasis"),
]


def _compile_patterns(include_profanity: bool = False) -> list[tuple[re.Pattern, str]]:
    patterns = [(re.compile(p, re.IGNORECASE), cat) for p, cat in FRUSTRATION_PATTERNS]
    patterns += [(re.compile(p, re.IGNORECASE), cat) for p, cat in EXCLAMATION_PATTERNS]
    if include_profanity:
        patterns += [(re.compile(p, re.IGNORECASE), cat) for p, cat in PROFANITY_PATTERNS]
    return patterns


# ── Corpus scanning ───────────────────────────────────────────────────

def scan_corpus(
    include_profanity: bool = False,
    include_emotion: bool = False,
    min_valence: float = -0.4,  # emotion threshold for negative entries
) -> list[dict]:
    """Scan all corpus entries for frustration signals.

    Returns list of hits with context:
    {
        "user_content": str,          # the frustrating user message
        "pattern_category": str,      # what kind of signal
        "pattern_match": str,         # the matched text
        "prev_assistant": str,        # what the AI said before (context)
        "next_assistant": str,        # what the AI said after (the correction/re-explanation)
        "session": str,               # source session ID
        "project": str,               # project/thread
        "timestamp": str,
        "emotion_class": str,         # if emotion metadata present
        "emotion_valence": float,
    }
    """
    patterns = _compile_patterns(include_profanity)
    hits = []

    corpus_files = sorted(CORPUS_DIR.rglob("*.jsonl"))
    # Skip chains/kernels — only scan real sessions
    corpus_files = [f for f in corpus_files if "_chains" not in str(f)]

    for cf in corpus_files:
        entries = []
        try:
            with open(cf) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
        except (json.JSONDecodeError, OSError):
            continue

        for i, entry in enumerate(entries):
            role = entry.get("role", "")
            if role != "user":
                continue

            content = entry.get("content", "")
            if len(content) < 10:
                continue

            # Skip tool results (start with [{ JSON)
            if content.lstrip().startswith("[{") or content.lstrip().startswith("{'tool"):
                continue
            # Skip identity block injections
            if "<identity>" in content or "pre-computed echo" in content:
                continue

            # Check patterns
            for pattern, category in patterns:
                m = pattern.search(content)
                if m:
                    # Skip positive profanity ("fucking beautiful" = excitement, not frustration)
                    if category == "profanity" and POSITIVE_PROFANITY_RE.search(content):
                        continue
                    # Get surrounding context
                    prev_asst = ""
                    next_asst = ""
                    for j in range(i - 1, max(i - 5, -1), -1):
                        if entries[j].get("role") == "assistant":
                            prev_asst = entries[j].get("content", "")[:500]
                            break
                    for j in range(i + 1, min(i + 5, len(entries))):
                        if entries[j].get("role") == "assistant":
                            next_asst = entries[j].get("content", "")[:500]
                            break

                    hit = {
                        "user_content": content[:500],
                        "pattern_category": category,
                        "pattern_match": m.group()[:80],
                        "prev_assistant": prev_asst,
                        "next_assistant": next_asst,
                        "session": entry.get("source_session", cf.stem),
                        "project": entry.get("thread", ""),
                        "timestamp": entry.get("ts", ""),
                        "emotion_class": entry.get("emotion_class", ""),
                        "emotion_valence": entry.get("emotion_valence", 0.0),
                    }
                    hits.append(hit)
                    break  # one hit per entry

            # Emotion-based mining (non-neutral negative entries)
            if include_emotion:
                valence = entry.get("emotion_valence", 0.0)
                emo_class = entry.get("emotion_class", "neutral")
                if isinstance(valence, (int, float)) and valence < min_valence and emo_class != "neutral":
                    # Check if already caught by patterns
                    already_hit = any(h["user_content"][:100] == content[:100] for h in hits[-20:])
                    if not already_hit:
                        prev_asst = ""
                        next_asst = ""
                        for j in range(i - 1, max(i - 5, -1), -1):
                            if entries[j].get("role") == "assistant":
                                prev_asst = entries[j].get("content", "")[:500]
                                break
                        for j in range(i + 1, min(i + 5, len(entries))):
                            if entries[j].get("role") == "assistant":
                                next_asst = entries[j].get("content", "")[:500]
                                break

                        hits.append({
                            "user_content": content[:500],
                            "pattern_category": f"emotion:{emo_class}",
                            "pattern_match": f"valence={valence:.2f}",
                            "prev_assistant": prev_asst,
                            "next_assistant": next_asst,
                            "session": entry.get("source_session", cf.stem),
                            "project": entry.get("thread", ""),
                            "timestamp": entry.get("ts", ""),
                            "emotion_class": emo_class,
                            "emotion_valence": valence,
                        })

    return hits


# ── Ground truth generation ───────────────────────────────────────────

def hits_to_ground_truth(hits: list[dict]) -> list[dict]:
    """Convert mined hits into ground truth query entries.

    For each frustration moment:
    - query = what the user was asking about (extracted from their message)
    - expected_keywords = key terms from the correction/re-explanation
    - description = the frustration pattern and context
    """
    ground_truth = []
    seen_queries = set()

    for hit in hits:
        user_msg = hit["user_content"]
        next_asst = hit["next_assistant"]
        category = hit["pattern_category"]

        # Extract the topic (strip the frustration and get the substance)
        # Use the first sentence or up to 100 chars as query seed
        query_seed = user_msg.split(".")[0].split("!")[0].split("?")[0][:100].strip()
        if len(query_seed) < 15:
            query_seed = user_msg[:100].strip()

        # Deduplicate by similar query
        query_key = query_seed[:50].lower()
        if query_key in seen_queries:
            continue
        seen_queries.add(query_key)

        # Extract keywords from the assistant's correction/re-explanation
        keywords = []
        if next_asst:
            # Extract identifiers, file paths, technical terms
            keywords += re.findall(r'[\w_]{4,}\.py\b', next_asst)
            keywords += re.findall(r'[\w_]{6,}\(\)', next_asst)
            keywords += re.findall(r'`([^`]{3,})`', next_asst)
            # Unique, max 8
            keywords = list(dict.fromkeys(keywords))[:8]

        if not keywords:
            # Fall back to extracting from the user message itself
            keywords = re.findall(r'`([^`]{3,})`', user_msg)
            keywords += [w for w in re.split(r'\W+', user_msg)
                        if len(w) >= 6 and w.lower() not in
                        {"already", "should", "before", "because", "remember",
                         "explained", "discussed", "session", "forgot"}]
            keywords = list(dict.fromkeys(keywords))[:6]

        gt_entry = {
            "query": query_seed,
            "expected_keywords": keywords,
            "description": f"[{category}] {hit['pattern_match']} — {hit['project']} {hit['timestamp'][:10]}",
            "source_category": category,
            "source_session": hit["session"][:12],
            "source_project": hit["project"],
        }

        ground_truth.append(gt_entry)

    return ground_truth


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Mine ground truth from frustration moments")
    parser.add_argument("--write", action="store_true", help="Append candidates to ground_truth.json")
    parser.add_argument("--emotion", action="store_true", help="Include emotion-based mining")
    parser.add_argument("--profanity", action="store_true", help="Include profanity signals")
    parser.add_argument("--raw", action="store_true", help="Show raw hits instead of ground truth")
    parser.add_argument("--limit", type=int, default=50, help="Max entries to show/write")
    args = parser.parse_args()

    print("Mining corpus for frustration signals...", file=sys.stderr)
    hits = scan_corpus(
        include_profanity=args.profanity,
        include_emotion=args.emotion,
    )

    print(f"Found {len(hits)} hits", file=sys.stderr)

    # Category breakdown
    from collections import Counter
    cats = Counter(h["pattern_category"] for h in hits)
    for cat, count in cats.most_common():
        print(f"  {cat}: {count}", file=sys.stderr)

    if args.raw:
        for hit in hits[:args.limit]:
            print(f"\n[{hit['pattern_category']}] {hit['pattern_match']}")
            print(f"  project: {hit['project']}  session: {hit['session'][:12]}")
            print(f"  user: {hit['user_content'][:120]}")
            if hit['next_assistant']:
                print(f"  asst: {hit['next_assistant'][:120]}")
        return

    # Generate ground truth
    gt = hits_to_ground_truth(hits)
    gt = gt[:args.limit]

    print(f"\nGenerated {len(gt)} ground truth candidates", file=sys.stderr)

    if args.write:
        # Load existing
        existing = []
        if GROUND_TRUTH_PATH.exists():
            with open(GROUND_TRUTH_PATH) as f:
                existing = json.load(f)

        # Append new
        combined = existing + gt
        with open(GROUND_TRUTH_PATH, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"Wrote {len(combined)} entries to {GROUND_TRUTH_PATH}", file=sys.stderr)
    else:
        # Print candidates for review
        for entry in gt:
            print(f"\n  query: {entry['query'][:80]}")
            print(f"  keywords: {entry['expected_keywords']}")
            print(f"  source: {entry['description']}")


if __name__ == "__main__":
    main()
