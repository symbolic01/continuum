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

# Hard bounds: physically meaningful limits that can never be crossed
# Soft bounds: starting search range, auto-expanded when the LLM hits them
PARAM_HARD_BOUNDS = {
    "semantic_k": (1, None, int),        # at least 1, no upper limit
    "keyword_k": (1, None, int),
    "keyword_weight": (0.0, None, float),  # non-negative, no ceiling
    "hybrid_boost": (0.0, None, float),
    "identifier_weight": (0.0, None, float),
    "decay_half_life_days": (1, None, float),  # at least 1 day
    "decay_floor": (0.0, 1.0, float),    # 0 = fully decay, 1 = no decay
    "correction_boost_max": (0.0, None, float),
    "context_boost": (0.0, None, float),
    "kernel_boost": (0.0, None, float),
    "chain_boost": (0.0, None, float),
    # Reranking params — these directly affect result ORDER
    "rerank_query_overlap": (0.0, None, float),
    "rerank_identifier_hit": (0.0, None, float),
    "rerank_specificity": (0.0, None, float),
    "rerank_recency": (0.0, None, float),
    "rerank_semantic": (0.0, None, float),
    "rerank_role_weight": (0.0, None, float),
}

# Soft bounds — starting search range (auto-expand on wall hits)
PARAM_SOFT_BOUNDS = {
    "semantic_k": (10, 100),
    "keyword_k": (10, 100),
    "keyword_weight": (0.0, 1.0),
    "hybrid_boost": (0.0, 1.0),
    "identifier_weight": (0.0, 1.0),
    "decay_half_life_days": (7, 120),
    "decay_floor": (0.0, 0.8),
    "correction_boost_max": (0.0, 2.0),
    "context_boost": (0.5, 5.0),
    "kernel_boost": (0.5, 5.0),
    "chain_boost": (0.5, 3.0),
    "rerank_query_overlap": (0.0, 5.0),
    "rerank_identifier_hit": (0.0, 5.0),
    "rerank_specificity": (0.0, 3.0),
    "rerank_recency": (0.0, 3.0),
    "rerank_semantic": (0.0, 5.0),
    "rerank_role_weight": (0.0, 3.0),
}

WALL_HIT_THRESHOLD = 3  # auto-expand after this many iterations at a bound

RESEARCH_PROMPT = """\
You are a retrieval systems researcher running parameter optimization experiments.
Your goal is to maximize the composite score = 0.25*keyword_recall + 0.40*mrr + 0.35*precision_at_k.

Current parameters:
{params_json}

Parameter bounds (min, max, type):
{bounds_json}

Experiment history (last {n_history} runs):
{log_entries}

Composite score = 0.25*keyword_recall + 0.40*mrr + 0.35*precision_at_k (higher is better)
- keyword_recall: fraction of expected keywords found (0-1)
- mrr: mean reciprocal rank of first relevant result (0-1)
- precision_at_k: fraction of top-20 results that are relevant (0-1)

{phase_guidance}

Reason about what the history suggests — which changes helped, which hurt,
what interactions you notice — then output ONLY this JSON:
{{"changes": {{"param_name": new_value, ...}}, "reasoning": "your reasoning"}}"""

PHASE_GUIDANCE = {
    1: "You are in EXPLORATION phase. Propose 3-5 bold parameter changes across diverse params. Try extreme values. The goal is to map which parameters matter at all.",
    2: "You are in REFINEMENT phase. Propose 2-4 parameter changes. Focus on the params that showed the most impact in exploration. Combine promising directions.",
    3: "You are in FINE-TUNING phase. Propose 1-2 small parameter changes. Narrow in on the best region found so far.",
}


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


def _get_effective_bounds() -> dict:
    """Get current soft bounds, auto-expanded by wall hits in the log."""
    bounds = {k: list(v) for k, v in PARAM_SOFT_BOUNDS.items()}
    log = load_log()

    # Count consecutive wall hits per param per bound direction
    wall_hits: dict[str, dict[str, int]] = {}  # param → {"lo": n, "hi": n}
    for entry in log[-50:]:  # look at recent history
        changes = entry.get("changes", {})
        params_after = entry.get("params_after", {})
        for param, value in changes.items():
            if param not in bounds:
                continue
            lo, hi = bounds[param]
            if wall_hits.get(param) is None:
                wall_hits[param] = {"lo": 0, "hi": 0}
            if value <= lo * 1.01:  # within 1% of lower bound
                wall_hits[param]["lo"] += 1
            else:
                wall_hits[param]["lo"] = 0
            if hi is not None and value >= hi * 0.99:  # within 1% of upper bound
                wall_hits[param]["hi"] += 1
            else:
                wall_hits[param]["hi"] = 0

    # Auto-expand soft bounds that hit the wall
    for param, hits in wall_hits.items():
        hard = PARAM_HARD_BOUNDS.get(param, (None, None, float))
        hard_lo, hard_hi, _ = hard
        if hits["lo"] >= WALL_HIT_THRESHOLD:
            new_lo = bounds[param][0] * 0.5  # halve the lower bound
            if hard_lo is not None:
                new_lo = max(hard_lo, new_lo)
            if new_lo != bounds[param][0]:
                bounds[param][0] = new_lo
                print(f"  ⚡ auto-expanded {param} lower bound → {new_lo}", file=sys.stderr)
        if hits["hi"] >= WALL_HIT_THRESHOLD:
            new_hi = bounds[param][1] * 1.5  # 50% expansion
            if hard_hi is not None:
                new_hi = min(hard_hi, new_hi)
            if new_hi != bounds[param][1]:
                bounds[param][1] = new_hi
                print(f"  ⚡ auto-expanded {param} upper bound → {new_hi}", file=sys.stderr)

    return bounds


def clamp_params(changes: dict) -> dict:
    """Clamp proposed values to effective bounds and correct types."""
    bounds = _get_effective_bounds()
    clamped = {}
    for param, value in changes.items():
        if param not in PARAM_HARD_BOUNDS:
            continue
        _, _, typ = PARAM_HARD_BOUNDS[param]
        value = typ(value)
        # Clamp to hard bounds (soft bounds are just guidance for the LLM)
        hard_lo, hard_hi, _ = PARAM_HARD_BOUNDS[param]
        if hard_lo is not None:
            value = max(hard_lo, value)
        if hard_hi is not None:
            value = min(hard_hi, value)
        clamped[param] = value
    return clamped


def random_proposal(current_params: dict) -> dict:
    """Phase 1: random multi-param exploration to map the landscape."""
    import random
    bounds = _get_effective_bounds()
    n_params = random.randint(3, 5)
    chosen = random.sample(list(bounds.keys()), n_params)
    changes = {}
    for p in chosen:
        lo, hi = bounds[p]
        _, _, typ = PARAM_HARD_BOUNDS[p]
        if typ == int:
            changes[p] = random.randint(int(lo), int(hi))
        else:
            changes[p] = round(random.uniform(lo, hi), 3)
    return {"changes": clamp_params(changes), "reasoning": f"Phase 1 random exploration: {', '.join(chosen)}"}


def get_phase(iteration_in_run: int, forced_phase: int | None = None) -> int:
    """Determine exploration phase from iteration count."""
    if forced_phase is not None:
        return forced_phase
    if iteration_in_run <= 15:
        return 1
    elif iteration_in_run <= 35:
        return 2
    else:
        return 3


def propose_changes(current_params: dict, log: list[dict], model: str) -> dict | None:
    """Ask the LLM to propose parameter changes based on experiment history."""
    # Format log entries (last 20)
    recent = log[-20:] if len(log) > 20 else log
    log_text = ""
    for entry in recent:
        accepted = "ACCEPTED" if entry.get("accepted") else "REJECTED"
        cs = composite_score(entry.get("scores", {}))
        log_text += f"\n  iter {entry.get('iteration', '?')}: {json.dumps(entry.get('changes', {}))} → "
        log_text += f"composite={cs:.3f} kw={entry['scores'].get('keyword_recall', 0):.3f} "
        log_text += f"mrr={entry['scores'].get('mrr', 0):.3f} "
        log_text += f"p@k={entry['scores'].get('precision_at_k', 0):.3f} "
        log_text += f"[{accepted}] {entry.get('reasoning', '')[:200]}"

    if not log_text:
        log_text = "(no previous experiments — this is the first run)"

    effective_bounds = _get_effective_bounds()
    # Phase guidance — caller can set this attribute
    phase = getattr(propose_changes, '_phase', 2)
    prompt = RESEARCH_PROMPT.format(
        params_json=json.dumps(current_params, indent=2),
        bounds_json=json.dumps(effective_bounds, indent=2),
        n_history=len(recent),
        log_entries=log_text,
        phase_guidance=PHASE_GUIDANCE.get(phase, PHASE_GUIDANCE[3]),
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


def composite_score(scores: dict) -> float:
    """Weighted composite: MRR and precision@k matter most."""
    return (
        0.25 * scores.get("keyword_recall", 0)
        + 0.40 * scores.get("mrr", 0)
        + 0.35 * scores.get("precision_at_k", 0)
    )


def is_improvement(new_scores: dict, baseline: dict) -> bool:
    """Accept if composite score improves."""
    return composite_score(new_scores) > composite_score(baseline)


def run_autoresearch(args):
    """Main autoresearch loop."""
    config = load_config(_CONTINUUM_DIR / "continuum.yaml")
    gt_path = Path(args.ground_truth) if args.ground_truth else None
    ground_truth = load_ground_truth(gt_path) if gt_path else load_ground_truth()
    model = args.model or get_model("compress")
    log = load_log()

    # Resume from last accepted params if log exists, otherwise use config defaults
    from core.retrieval import ContextRetriever
    accepted_log = [e for e in log if e.get("accepted")]
    if accepted_log:
        current_params = dict(accepted_log[-1]["params_after"])
        print(f"── Resuming from iteration {len(log)} (best composite={composite_score(accepted_log[-1]['scores']):.3f}) ──", file=sys.stderr)
    else:
        current_params = dict(config.get("retrieval", {}))
        print("── Baseline ──", file=sys.stderr)

    # Fill in defaults for any missing params
    for k, v in ContextRetriever.DEFAULT_PARAMS.items():
        if k not in current_params:
            current_params[k] = v

    config["retrieval"] = current_params
    baseline_agg, _ = run_eval(config, ground_truth)
    print(f"  kw={baseline_agg['keyword_recall']:.3f}  mrr={baseline_agg['mrr']:.3f}  "
          f"p@k={baseline_agg['precision_at_k']:.3f}  tok={baseline_agg['token_count']:.0f}  "
          f"lat={baseline_agg['latency_ms']:.0f}ms  composite={composite_score(baseline_agg):.3f}", file=sys.stderr)

    start_time = time.monotonic()
    iteration = len(log)
    accepted_count = 0

    for i in range(args.iterations):
        # Time cap
        if args.max_time and (time.monotonic() - start_time) > args.max_time:
            print(f"\n── Time cap reached ({args.max_time}s) ──", file=sys.stderr)
            break

        iteration += 1
        iteration_in_run = i + 1
        phase = get_phase(iteration_in_run, getattr(args, 'phase', None))
        print(f"\n── Iteration {iteration} (phase {phase}) ──", file=sys.stderr)

        # Phase 1: random exploration. Phase 2-3: LLM-guided.
        if phase == 1:
            proposal = random_proposal(current_params)
        else:
            propose_changes._phase = phase
            proposal = propose_changes(current_params, log, model)
        if not proposal:
            print("  No valid proposal, retrying...", file=sys.stderr)
            continue

        changes = proposal["changes"]
        reasoning = proposal["reasoning"]
        print(f"  Proposed: {json.dumps(changes)}", file=sys.stderr)
        print(f"  Reasoning: {reasoning[:300]}", file=sys.stderr)

        # Apply changes to a copy of config
        test_params = dict(current_params)
        test_params.update(changes)
        test_config = copy.deepcopy(config)
        test_config["retrieval"] = test_params

        # Evaluate
        new_agg, _ = run_eval(test_config, ground_truth)
        accepted = is_improvement(new_agg, baseline_agg)
        cs_new = composite_score(new_agg)
        cs_base = composite_score(baseline_agg)
        verdict = "ACCEPTED" if accepted else "REJECTED"
        print(f"  Result:   composite={cs_new:.3f} (baseline={cs_base:.3f})  kw={new_agg['keyword_recall']:.3f}  "
              f"mrr={new_agg['mrr']:.3f}  p@k={new_agg['precision_at_k']:.3f}  [{verdict}]", file=sys.stderr)

        entry = {
            "iteration": iteration,
            "phase": phase,
            "changes": changes,
            "reasoning": reasoning,
            "scores": new_agg,
            "composite": cs_new,
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
    parser.add_argument("--phase", type=int, default=None, choices=[1, 2, 3], help="Force exploration phase (1=random, 2=guided, 3=fine-tune)")
    parser.add_argument("--ground-truth", default=None, help="Path to ground_truth.json (default: built-in)")
    parser.add_argument("--report", action="store_true", help="Show experiment history and best params")
    args = parser.parse_args()

    if args.report:
        show_report()
    else:
        run_autoresearch(args)


if __name__ == "__main__":
    main()
