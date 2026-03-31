#!/usr/bin/env python3
"""Dream daemon — batched integration with wakeup/timed synthesis.

Night mode: runs fast integration-only cycles while idle. Accumulates
chains cheaply (Ollama only, no API calls). Triggers full synthesis
when the user wakes up (session activity detected), or after max_hours,
or at morning_hour — whichever comes first.

Usage:
    python dream_daemon.py                    # defaults: 15min idle, 30min dream
    python dream_daemon.py --idle 10 --dream 30
    python dream_daemon.py --once             # check once and exit (for cron)
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

CC_SESSIONS_DIR = Path.home() / ".claude" / "projects"
DREAM_STATE = Path.home() / ".continuum" / "dream_state.json"
DAEMON_LOG = Path.home() / ".continuum" / "dream_daemon.log"
CONTINUUM_DIR = Path(__file__).resolve().parent


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, file=sys.stderr)
    try:
        with open(DAEMON_LOG, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def newest_session_mtime() -> float:
    pattern = str(CC_SESSIONS_DIR / "*" / "*.jsonl")
    files = glob.glob(pattern)
    if not files:
        return 0.0
    return max(os.path.getmtime(f) for f in files)


def load_dream_state() -> dict:
    if DREAM_STATE.exists():
        try:
            return json.loads(DREAM_STATE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_dream_state(state: dict):
    DREAM_STATE.parent.mkdir(parents=True, exist_ok=True)
    with open(DREAM_STATE, "w") as f:
        json.dump(state, f, indent=2)


def is_user_active() -> bool:
    """Check if user has been active recently (session write within 2 min)."""
    last_write = newest_session_mtime()
    if last_write == 0:
        return False
    return (time.time() - last_write) < 120


def should_integrate(idle_minutes: float, min_gap_minutes: float) -> tuple[bool, str]:
    """Check if we should run an integration-only cycle."""
    now = time.time()

    last_write = newest_session_mtime()
    if last_write == 0:
        return False, "no session files found"

    idle_seconds = now - last_write
    idle_mins = idle_seconds / 60

    if idle_mins < idle_minutes:
        return False, f"sessions active {idle_mins:.0f}m ago (need {idle_minutes:.0f}m)"

    # Check gap since last dream
    state = load_dream_state()
    last_run = state.get("last_run", "")
    if last_run:
        try:
            dt = datetime.fromisoformat(last_run)
            gap_mins = (now - dt.timestamp()) / 60
            if gap_mins < min_gap_minutes:
                return False, f"dreamed {gap_mins:.0f}m ago (need {min_gap_minutes:.0f}m gap)"
        except (ValueError, TypeError):
            pass

    return True, f"idle {idle_mins:.0f}m"


def should_synthesize(max_hours: float, morning_hour: float) -> tuple[bool, str]:
    """Check if accumulated chains should be synthesized now.

    Triggers on:
    1. User wakeup (session activity after idle period)
    2. max_hours since synthesis started accumulating
    3. Morning hour reached (e.g., 6:30 AM)
    """
    state = load_dream_state()
    pending_since = state.get("pending_synthesis_since", "")

    if not pending_since:
        return False, "no pending synthesis"

    now = datetime.now()

    # 1. User wakeup
    if is_user_active():
        return True, "user woke up"

    # 2. Max hours elapsed
    try:
        dt = datetime.fromisoformat(pending_since)
        hours_elapsed = (now.timestamp() - dt.timestamp()) / 3600
        if hours_elapsed >= max_hours:
            return True, f"max time ({hours_elapsed:.1f}h >= {max_hours}h)"
    except (ValueError, TypeError):
        pass

    # 3. Morning hour
    current_hour = now.hour + now.minute / 60
    if current_hour >= morning_hour and current_hour < morning_hour + 0.5:
        return True, f"morning ({now.strftime('%H:%M')})"

    return False, "waiting"


def run_autoresearch(verbose: bool = False):
    """Run one autoresearch iteration to tune retrieval params.

    Uses a random sample of GT entries each time (prevents overfitting).
    Generates new GT entries periodically to track corpus growth.
    Resumes from best params found so far.
    """
    import random

    gt_full_path = CONTINUUM_DIR / "ground_truth_uid.json"
    gt_sample_path = Path.home() / ".continuum" / "gt_sample.json"

    # Load full GT pool
    try:
        with open(gt_full_path) as f:
            gt_full = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        log("Autoresearch: no ground truth file, skipping")
        return

    if len(gt_full) < 10:
        log(f"Autoresearch: only {len(gt_full)} GT entries, skipping")
        return

    # Sample random subset (100 or all if fewer)
    sample_size = min(100, len(gt_full))
    sample = random.sample(gt_full, sample_size)
    with open(gt_sample_path, "w") as f:
        json.dump(sample, f)

    log(f"Autoresearch: sampled {sample_size}/{len(gt_full)} GT entries")

    cmd = [
        sys.executable, str(CONTINUUM_DIR / "autoresearch.py"),
        "--iterations", "1",
        "--ground-truth", str(gt_sample_path),
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=1200,  # 20 min max
        )
        for line in result.stderr.strip().split("\n"):
            if line.strip() and "identifiers resolved" not in line:
                log(f"  {line.strip()}")
    except subprocess.TimeoutExpired:
        log("Autoresearch iteration timed out")


def run_gt_refresh(verbose: bool = False):
    """Generate new GT entries from current corpus. Runs every ~10 cycles."""
    state = load_dream_state()
    cycles_since_refresh = state.get("cycles_since_gt_refresh", 0)

    if cycles_since_refresh < 10:
        state["cycles_since_gt_refresh"] = cycles_since_refresh + 1
        save_dream_state(state)
        return

    log("Refreshing ground truth from current corpus...")

    cmd = [
        sys.executable, "-c",
        """
import json, random, re, subprocess, glob, sys
sys.path.insert(0, '%s')
from core.index import load_index
from core.retrieval import ContextRetriever
from core.config import load_config

# Load current corpus chunks with embeddings
chunks = []
idx = load_index()
index_uids = {m.get('uid','') for m in idx.metadata} if idx.metadata else set()

for f in glob.glob('/home/symbolic/.continuum/corpus/**/*.jsonl', recursive=True):
    if '_archive' in f: continue
    for line in open(f):
        line = line.strip()
        if not line: continue
        try:
            e = json.loads(line)
            if e.get('embedding') and e.get('uid','') in index_uids:
                content = e.get('content','')
                if len(content) >= 150 and e.get('role','') not in ('tool_result','tool_use'):
                    chunks.append(e)
        except: pass

# Sample 50 new chunks
sample = random.sample(chunks, min(50, len(chunks)))
contents = [c.get('content','')[:500] for c in sample]

# Generate queries via Claude
all_queries = []
for batch_start in range(0, len(contents), 20):
    batch = contents[batch_start:batch_start+20]
    chunk_text = ''
    for i, content in enumerate(batch):
        chunk_text += f'\\n[{i}]\\n{content[:400]}\\n'
    prompt = f'For each numbered chunk, write ONE search query (10-15 words). Return JSON: {{"0":"query",...}}\\n{chunk_text}\\nJSON:'
    try:
        result = subprocess.run(['claude','--print','--model','claude-haiku-4-5-20251001'],
            input=prompt, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            match = re.search(r'\\{[\\s\\S]*\\}', result.stdout)
            if match:
                parsed = json.loads(match.group())
                for key, query in parsed.items():
                    idx_num = int(key)
                    if 0 <= idx_num < len(batch) and isinstance(query,str) and 10<len(query)<150:
                        all_queries.append((batch_start+idx_num, query.strip()))
    except: pass

# Append new entries to GT
gt_path = '%s/ground_truth_uid.json'
try:
    with open(gt_path) as f: gt = json.load(f)
except: gt = []
existing_uids = {e.get('target_uid','') for e in gt}

added = 0
for chunk_idx, query in all_queries:
    uid = sample[chunk_idx].get('uid','')
    if uid in existing_uids: continue
    gt.append({'query':query,'target_uid':uid,'expected_keywords':[],'description':'[auto_refresh]','source_category':'reverse_gt','source_project':sample[chunk_idx].get('thread','')})
    existing_uids.add(uid)
    added += 1

with open(gt_path,'w') as f: json.dump(gt, f, indent=2)
print(f'Added {added} new GT entries (total: {len(gt)})')
""" % (str(CONTINUUM_DIR), str(CONTINUUM_DIR))
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.stdout.strip():
            log(f"  {result.stdout.strip()}")
        for line in result.stderr.strip().split("\n"):
            if line.strip():
                log(f"  {line.strip()}")
    except subprocess.TimeoutExpired:
        log("GT refresh timed out")

    state["cycles_since_gt_refresh"] = 0
    save_dream_state(state)


def run_integration(dream_minutes: int, verbose: bool = False):
    """Run integration-only cycle (no synthesis, no validation — fast and free)."""
    dream_seconds = dream_minutes * 60
    log(f"Integration cycle ({dream_minutes}m)")

    cmd = [
        sys.executable, str(CONTINUUM_DIR / "dream_tool.py"),
        "--max-time", str(dream_seconds),
        "--force",
        "--wake-on-activity",
        "--no-synthesis",
        "--no-temporal",
    ]
    if verbose:
        cmd.append("-v")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=dream_seconds + 60,
        )
        for line in result.stderr.strip().split("\n"):
            if line.strip():
                log(f"  {line.strip()}")

        # Mark pending synthesis + increment cycle count
        state = load_dream_state()
        if not state.get("pending_synthesis_since"):
            state["pending_synthesis_since"] = datetime.now().astimezone().isoformat()
        state["integration_cycle_count"] = state.get("integration_cycle_count", 0) + 1
        save_dream_state(state)

        log(f"Integration cycle complete (cycle {state['integration_cycle_count']})")
    except subprocess.TimeoutExpired:
        log(f"Integration timed out")


def run_synthesis(verbose: bool = False):
    """Run full synthesis + validation + gap analysis on accumulated chains."""
    log("Running synthesis on accumulated chains...")

    cmd = [
        sys.executable, str(CONTINUUM_DIR / "dream_tool.py"),
        "--force",
        "--no-ingest",
        "--max-time", "300",  # 5 min for integration + pre-synthesis batching
        "--report-file", str(Path.home() / ".continuum" / "last_dream_report.md"),
    ]
    if verbose:
        cmd.append("-v")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=900,  # 15 min for pre-synthesis + synthesis + validation
        )
        for line in result.stderr.strip().split("\n"):
            if line.strip():
                log(f"  {line.strip()}")

        # Archive chains if 3+ cycles accumulated
        state = load_dream_state()
        cycle_count = state.get("integration_cycle_count", 0)
        if cycle_count >= 3:
            log(f"Archiving chains ({cycle_count} cycles accumulated)")
            try:
                archive_cmd = [
                    sys.executable, "-c",
                    "import sys; sys.path.insert(0, '" + str(CONTINUUM_DIR) + "'); "
                    "from core.dream import DreamEngine; "
                    "e = DreamEngine(verbose=True); "
                    "e.load_corpus(); "
                    "e.archive_chains()"
                ]
                archive_result = subprocess.run(
                    archive_cmd, capture_output=True, text=True, timeout=120)
                for line in archive_result.stderr.strip().split("\n"):
                    if line.strip():
                        log(f"  {line.strip()}")
            except (subprocess.TimeoutExpired, Exception) as e:
                log(f"  Archive failed: {e}")

        # Clear pending synthesis flag + reset cycle count
        state.pop("pending_synthesis_since", None)
        state["integration_cycle_count"] = 0
        save_dream_state(state)

        # Copy report to served directory
        import shutil
        src_report = Path.home() / ".continuum" / "dream_report.json"
        dst_report = CONTINUUM_DIR / "dream_report.json"
        if src_report.exists():
            shutil.copy2(str(src_report), str(dst_report))
        # Copy manifest
        src_manifest = Path.home() / ".continuum" / "dream_reports" / "manifest.json"
        dst_manifest = CONTINUUM_DIR / "dream_reports" / "manifest.json"
        if src_manifest.exists():
            dst_manifest.parent.mkdir(exist_ok=True)
            shutil.copy2(str(src_manifest), str(dst_manifest))
        # Copy new versioned reports
        src_dir = Path.home() / ".continuum" / "dream_reports"
        dst_dir = CONTINUUM_DIR / "dream_reports"
        for f in src_dir.glob("dream_report_*.json"):
            dst = dst_dir / f.name
            if not dst.exists():
                shutil.copy2(str(f), str(dst))

        log("Synthesis complete")
    except subprocess.TimeoutExpired:
        log("Synthesis timed out (15m)")


def main():
    parser = argparse.ArgumentParser(description="Dream daemon — batched integration + wakeup synthesis")
    parser.add_argument("--idle", type=float, default=15,
                        help="Minutes of inactivity before dreaming (default: 15)")
    parser.add_argument("--dream", type=int, default=30,
                        help="Minutes per integration cycle (default: 30)")
    parser.add_argument("--check", type=int, default=60,
                        help="Seconds between checks (default: 60)")
    parser.add_argument("--gap", type=float, default=10,
                        help="Minutes between integration cycles (default: 10)")
    parser.add_argument("--max-hours", type=float, default=7,
                        help="Max hours before forced synthesis (default: 7)")
    parser.add_argument("--morning", type=float, default=6.5,
                        help="Morning synthesis hour in 24h (default: 6.5 = 6:30 AM)")
    parser.add_argument("--once", action="store_true",
                        help="Check once and exit (for cron/systemd)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.once:
        # Check synthesis trigger first
        should_synth, synth_reason = should_synthesize(args.max_hours, args.morning)
        if should_synth:
            log(f"Triggering synthesis: {synth_reason}")
            run_synthesis(args.verbose)
            return

        # Otherwise check integration
        should, reason = should_integrate(args.idle, args.gap)
        if should:
            # Pre-dream: refresh GT + tune retrieval
            run_gt_refresh(args.verbose)
            run_autoresearch(args.verbose)
            # Then dream
            log(f"Triggering integration: {reason}")
            run_integration(args.dream, args.verbose)
        else:
            # quiet — don't log every skip
            pass
        return

    # Daemon loop
    log(f"Dream daemon started (idle={args.idle}m, dream={args.dream}m, "
        f"gap={args.gap}m, max_hours={args.max_hours}h, "
        f"morning={args.morning})")

    while True:
        # Check synthesis trigger first (higher priority)
        should_synth, synth_reason = should_synthesize(args.max_hours, args.morning)
        if should_synth:
            log(f"Triggering synthesis: {synth_reason}")
            run_synthesis(args.verbose)
        else:
            # Check integration
            should, reason = should_integrate(args.idle, args.gap)
            if should:
                # Pre-dream: refresh GT (every ~10 cycles) + tune retrieval
                run_gt_refresh(args.verbose)
                run_autoresearch(args.verbose)
                # Then dream
                log(f"Triggering integration: {reason}")
                run_integration(args.dream, args.verbose)

        time.sleep(args.check)


if __name__ == "__main__":
    main()
