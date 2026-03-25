#!/usr/bin/env python3
"""Benchmark harness for retrieval quality evaluation.

Runs ground truth queries against the retrieval system and scores
keyword recall, MRR, precision@k, token count, and latency.

Usage:
    python eval.py                          # run with current params
    python eval.py --config /path/to.yaml   # run with custom config
    python eval.py --params '{"keyword_weight": 0.5}'  # override specific params
"""

import argparse
import json
import sys
import time
from pathlib import Path

_CONTINUUM_DIR = Path(__file__).resolve().parent
if str(_CONTINUUM_DIR) not in sys.path:
    sys.path.insert(0, str(_CONTINUUM_DIR))

from core.config import load_config
from core.index import load_index
from core.retrieval import ContextRetriever
from core.tokens import count_tokens


DEFAULT_GROUND_TRUTH = _CONTINUUM_DIR / "ground_truth.json"


def load_ground_truth(path: Path = DEFAULT_GROUND_TRUTH) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def score_result(result_text: str, test_case: dict) -> dict:
    """Score a single retrieval result against a test case."""
    result_lower = result_text.lower()
    lines = [l for l in result_text.split("\n") if l.strip()]

    expected_keywords = test_case.get("expected_keywords", [])
    expected_uids = test_case.get("expected_uids", [])

    # Keyword recall: fraction of expected keywords found
    if expected_keywords:
        kw_hits = sum(1 for kw in expected_keywords if kw.lower() in result_lower)
        keyword_recall = kw_hits / len(expected_keywords)
    else:
        keyword_recall = 0.0

    # MRR: 1/rank of first line containing any expected keyword or UID
    mrr = 0.0
    for i, line in enumerate(lines):
        line_lower = line.lower()
        hit = False
        if expected_keywords and any(kw.lower() in line_lower for kw in expected_keywords):
            hit = True
        if expected_uids and any(uid in line for uid in expected_uids):
            hit = True
        if hit:
            mrr = 1.0 / (i + 1)
            break

    # Precision@k: fraction of top-k lines that contain any expected keyword
    k = min(20, len(lines))
    if k > 0 and expected_keywords:
        relevant_lines = sum(
            1 for line in lines[:k]
            if any(kw.lower() in line.lower() for kw in expected_keywords)
        )
        precision_at_k = relevant_lines / k
    else:
        precision_at_k = 0.0

    # Token count
    token_count = count_tokens(result_text)

    return {
        "keyword_recall": keyword_recall,
        "mrr": mrr,
        "precision_at_k": precision_at_k,
        "token_count": token_count,
    }


def run_eval(
    config: dict,
    ground_truth: list[dict],
    param_overrides: dict | None = None,
    budget: int = 30000,
) -> dict:
    """Run full evaluation. Returns aggregate scores."""
    # Apply param overrides to config
    if param_overrides:
        if "retrieval" not in config:
            config["retrieval"] = {}
        config["retrieval"].update(param_overrides)

    idx = load_index()
    sources = config.get("context_sources", [])
    retriever = ContextRetriever(sources=sources, index=idx, config=config)

    all_scores = []
    total_latency = 0.0

    for tc in ground_truth:
        query = tc["query"]

        t0 = time.monotonic()
        result = retriever.retrieve(
            query=query,
            token_budget=budget,
            conversation_tail="",
            cull=False,  # no cull during eval — measure raw retrieval quality
        )
        latency_ms = (time.monotonic() - t0) * 1000
        total_latency += latency_ms

        scores = score_result(result, tc)
        scores["latency_ms"] = latency_ms
        scores["query"] = query
        all_scores.append(scores)

    # Aggregate
    n = len(all_scores)
    if n == 0:
        return {"keyword_recall": 0, "mrr": 0, "precision_at_k": 0, "token_count": 0, "latency_ms": 0}

    agg = {
        "keyword_recall": sum(s["keyword_recall"] for s in all_scores) / n,
        "mrr": sum(s["mrr"] for s in all_scores) / n,
        "precision_at_k": sum(s["precision_at_k"] for s in all_scores) / n,
        "token_count": sum(s["token_count"] for s in all_scores) / n,
        "latency_ms": total_latency / n,
    }

    return agg, all_scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality")
    parser.add_argument("--config", default=None, help="Path to continuum.yaml")
    parser.add_argument("--params", default=None, help="JSON string of param overrides")
    parser.add_argument("--ground-truth", default=None, help="Path to ground_truth.json")
    parser.add_argument("--budget", type=int, default=30000, help="Token budget per query")
    parser.add_argument("--verbose", action="store_true", help="Show per-query scores")
    args = parser.parse_args()

    config = load_config(args.config or _CONTINUUM_DIR / "continuum.yaml")
    gt_path = Path(args.ground_truth) if args.ground_truth else DEFAULT_GROUND_TRUTH
    ground_truth = load_ground_truth(gt_path)
    param_overrides = json.loads(args.params) if args.params else None

    print(f"Running {len(ground_truth)} queries...", file=sys.stderr)

    agg, per_query = run_eval(config, ground_truth, param_overrides, args.budget)

    if args.verbose:
        for s in per_query:
            print(f"  {s['query'][:50]:50s}  kw={s['keyword_recall']:.2f}  mrr={s['mrr']:.2f}  p@k={s['precision_at_k']:.2f}  tok={s['token_count']}  {s['latency_ms']:.0f}ms")
        print()

    print(f"keyword_recall: {agg['keyword_recall']:.3f}")
    print(f"mrr:            {agg['mrr']:.3f}")
    print(f"precision_at_k: {agg['precision_at_k']:.3f}")
    print(f"token_count:    {agg['token_count']:.0f}")
    print(f"latency_ms:     {agg['latency_ms']:.0f}")

    # Also output as JSON for scripting
    print(json.dumps(agg))


if __name__ == "__main__":
    main()
