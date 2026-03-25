#!/usr/bin/env python3
"""LLM-guided autoresearch loop for retrieval parameter optimization.

An LLM reads its own experiment history and proposes parameter changes.
Multi-objective scoring: keyword recall (primary), MRR, precision@k,
token efficiency, latency.

Usage:
    python autoresearch.py                    # 100 iterations
    python autoresearch.py --iterations 500   # overnight
    python autoresearch.py --max-time 28800   # 8 hour cap
    python autoresearch.py --report           # show best params
"""

import argparse
import copy
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_CONTINUUM_DIR = Path(__file__).resolve().parent
if str(_CONTINUUM_DIR) not in sys.path:
    sys.path.insert(0, str(_CONTINUUM_DIR))

from core.config import load_config, get_model
from eval import run_eval, load_ground_truth

LOG_PATH = Path.home() / ".continuum" / "autoresearch_log.jsonl"

PARAM_BOUNDS = {
    "semantic_k": (10, 100, int),
    "keyword_k": (10, 100, int),
    "keyword_weight": (0.0, 1.0, float),
    "hybrid_boost": (0.0, 1.0, float),
    "identifier_weight": (0.0, 1.0, float),
    "decay_half_life_days": (7, 120, float),
    "decay_floor": (0.0, 0.8, float),
    "correction_boost_max": (0.0, 2.0, float),
    "context_boost": (0.5, 5.0, float),
    "kernel_boost": (0.5, 5.0, float),
    "chain_boost": (0.5, 3.0, float),
}

RESEARCH_PROMPT = """\
You are a retrieval systems researcher running parameter optimization experiments.
Your goal is to maximize cross-session recall (keyword_recall) while keeping other metrics stable.

Current parameters:
{params_json}

Parameter bounds (min, max, type):
{bounds_json}

Experiment history (last {n_history} runs):
{log_entries}

Primary metric: keyword_recall (cross-session memory recall — higher is better)
Secondary metrics:
- mrr (mean reciprocal rank — higher is better)
- precision_at_k (fraction of top-k results that are relevant — higher is better)
- token_count (tokens in output — lower is better for efficiency)
- latency_ms (retrieval time — lower is better)

Propose the next experiment. You may change 1-3 parameters at once.
Reason briefly about what the history suggests — which changes helped, which hurt,
what interactions you notice — then output ONLY this JSON:
{{"changes": {{"param_name": new_value, ...}}, "reasoning": "your reasoning"}}"""


def load_log() -> list[dict]:
    """Load experiment history."""
    if not LOG_PATH.exists():
        return []
    entries = []
    with open(LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


def append_log(entry: dict):
    """Append an experiment result."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def clamp_params(changes: dict) -> dict:
    """Clamp proposed values to bounds and correct types."""
    clamped = {}
    for param, value in changes.items():
        if param not in PARAM_BOUNDS:
            continue
        lo, hi, typ = PARAM_BOUNDS[param]
        value = typ(value)
        value = max(lo, min(hi, value))
        clamped[param] = value
    return clamped


def propose_changes(current_params: dict, log: list[dict], model: str) -> dict | None:
    """Ask the LLM to propose parameter changes based on experiment history."""
    # Format log entries (last 20)
    recent = log[-20:] if len(log) > 20 else log
    log_text = ""
    for entry in recent:
        accepted = "ACCEPTED" if entry.get("accepted") else "REJECTED"
        log_text += f"\n  iter {entry.get('iteration', '?')}: {json.dumps(entry.get('changes', {}))} → "
        log_text += f"kw={entry['scores'].get('keyword_recall', 0):.3f} "
        log_text += f"mrr={entry['scores'].get('mrr', 0):.3f} "
        log_text += f"p@k={entry['scores'].get('precision_at_k', 0):.3f} "
        log_text += f"tok={entry['scores'].get('token_count', 0):.0f} "
        log_text += f"[{accepted}] {entry.get('reasoning', '')[:80]}"

    if not log_text:
        log_text = "(no previous experiments — this is the first run)"

    prompt = RESEARCH_PROMPT.format(
        params_json=json.dumps(current_params, indent=2),
        bounds_json=json.dumps({k: (lo, hi) for k, (lo, hi, _) in PARAM_BOUNDS.items()}, indent=2),
        n_history=len(recent),
        log_entries=log_text,
    )

    try:
        result = subprocess.run(
            ["claude", "--print", "--model", model],
            input=prompt,
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            print(f"  LLM error: {(result.stderr or '').strip()[:200]}", file=sys.stderr)
            return None

        output = result.stdout.strip()
        # Extract JSON from response
        import re
        match = re.search(r'\{[\s\S]*"changes"[\s\S]*\}', output)
        if match:
            parsed = json.loads(match.group())
            changes = parsed.get("changes", {})
            reasoning = parsed.get("reasoning", "")
            clamped = clamp_params(changes)
            if clamped:
                return {"changes": clamped, "reasoning": reasoning}

        print(f"  Could not parse LLM output", file=sys.stderr)
        return None

    except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        print(f"  LLM call failed: {e}", file=sys.stderr)
        return None


def is_improvement(new_scores: dict, baseline: dict) -> bool:
    """Multi-objective acceptance: primary improves, secondaries don't regress >10%."""
    # Primary must improve
    if new_scores["keyword_recall"] <= baseline["keyword_recall"]:
        return False

    # Secondaries can't regress more than 10%
    for metric in ["mrr", "precision_at_k"]:
        base_val = baseline[metric]
        if base_val > 0 and new_scores[metric] < base_val * 0.9:
            return False

    return True


def run_autoresearch(args):
    """Main autoresearch loop."""
    config = load_config(_CONTINUUM_DIR / "continuum.yaml")
    ground_truth = load_ground_truth()
    model = args.model or get_model("compress")
    log = load_log()

    # Establish baseline with current params
    print("── Baseline ──", file=sys.stderr)
    current_params = dict(config.get("retrieval", {}))
    # Fill in defaults for any missing params
    from core.retrieval import ContextRetriever
    for k, v in ContextRetriever.DEFAULT_PARAMS.items():
        if k not in current_params:
            current_params[k] = v

    baseline_agg, _ = run_eval(config, ground_truth)
    print(f"  kw={baseline_agg['keyword_recall']:.3f}  mrr={baseline_agg['mrr']:.3f}  "
          f"p@k={baseline_agg['precision_at_k']:.3f}  tok={baseline_agg['token_count']:.0f}  "
          f"lat={baseline_agg['latency_ms']:.0f}ms", file=sys.stderr)

    start_time = time.monotonic()
    iteration = len(log)
    accepted_count = 0

    for i in range(args.iterations):
        # Time cap
        if args.max_time and (time.monotonic() - start_time) > args.max_time:
            print(f"\n── Time cap reached ({args.max_time}s) ──", file=sys.stderr)
            break

        iteration += 1
        print(f"\n── Iteration {iteration} ──", file=sys.stderr)

        # Ask LLM for proposal
        proposal = propose_changes(current_params, log, model)
        if not proposal:
            print("  No valid proposal, retrying...", file=sys.stderr)
            continue

        changes = proposal["changes"]
        reasoning = proposal["reasoning"]
        print(f"  Proposed: {json.dumps(changes)}", file=sys.stderr)
        print(f"  Reasoning: {reasoning[:100]}", file=sys.stderr)

        # Apply changes to a copy of config
        test_params = dict(current_params)
        test_params.update(changes)
        test_config = copy.deepcopy(config)
        test_config["retrieval"] = test_params

        # Evaluate
        new_agg, _ = run_eval(test_config, ground_truth)
        print(f"  Result:   kw={new_agg['keyword_recall']:.3f}  mrr={new_agg['mrr']:.3f}  "
              f"p@k={new_agg['precision_at_k']:.3f}  tok={new_agg['token_count']:.0f}", file=sys.stderr)

        # Accept or reject
        accepted = is_improvement(new_agg, baseline_agg)

        entry = {
            "iteration": iteration,
            "changes": changes,
            "reasoning": reasoning,
            "scores": new_agg,
            "baseline": baseline_agg,
            "params_after": test_params if accepted else current_params,
            "accepted": accepted,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        append_log(entry)
        log.append(entry)

        if accepted:
            current_params = test_params
            baseline_agg = new_agg
            config["retrieval"] = current_params
            accepted_count += 1
            print(f"  ✓ ACCEPTED (total: {accepted_count})", file=sys.stderr)
        else:
            print(f"  ✗ rejected", file=sys.stderr)

    # Summary
    elapsed = time.monotonic() - start_time
    print(f"\n── Done: {iteration - len(load_log()) + len(log)} iterations, "
          f"{accepted_count} accepted, {elapsed:.0f}s ──", file=sys.stderr)
    print(f"Best params: {json.dumps(current_params, indent=2)}", file=sys.stderr)
    print(f"Best scores: kw={baseline_agg['keyword_recall']:.3f}  mrr={baseline_agg['mrr']:.3f}  "
          f"p@k={baseline_agg['precision_at_k']:.3f}", file=sys.stderr)


def show_report():
    """Print summary of autoresearch history."""
    log = load_log()
    if not log:
        print("No experiments yet. Run: python autoresearch.py")
        return

    accepted = [e for e in log if e.get("accepted")]
    print(f"Total experiments: {len(log)}")
    print(f"Accepted: {len(accepted)}")
    print()

    if accepted:
        best = accepted[-1]
        print("Current best params:")
        print(json.dumps(best.get("params_after", {}), indent=2))
        print()
        print(f"Best scores:")
        for k, v in best["scores"].items():
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    # Show improvement trajectory
    print("\nAccepted improvements:")
    for e in accepted[-10:]:
        changes = e.get("changes", {})
        kw = e["scores"].get("keyword_recall", 0)
        print(f"  iter {e['iteration']}: {json.dumps(changes)} → kw={kw:.3f}")


def main():
    parser = argparse.ArgumentParser(description="LLM-guided autoresearch for retrieval optimization")
    parser.add_argument("--iterations", type=int, default=100, help="Max iterations (default: 100)")
    parser.add_argument("--max-time", type=int, default=0, help="Wall time cap in seconds (0=unlimited)")
    parser.add_argument("--model", default="", help="Model for research proposals (default: compress model)")
    parser.add_argument("--report", action="store_true", help="Show experiment history and best params")
    args = parser.parse_args()

    if args.report:
        show_report()
    else:
        run_autoresearch(args)


if __name__ == "__main__":
    main()
