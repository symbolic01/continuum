#!/usr/bin/env python3
"""Dream daemon — triggers cx dream when all sessions are idle.

Watches Claude Code session files for activity. When no session has
been written to for `idle_minutes`, runs `cx dream` for `dream_minutes`.

Usage:
    python dream_daemon.py                    # defaults: 15min idle, 30min dream
    python dream_daemon.py --idle 10 --dream 60
    python dream_daemon.py --once             # check once and exit (for cron)

Designed to run as a background process or via cron/systemd timer.
"""

import argparse
import glob
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
    """Log with timestamp."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, file=sys.stderr)
    try:
        with open(DAEMON_LOG, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def newest_session_mtime() -> float:
    """Find the most recent write to any CC session JSONL."""
    pattern = str(CC_SESSIONS_DIR / "*" / "*.jsonl")
    files = glob.glob(pattern)
    if not files:
        return 0.0
    return max(os.path.getmtime(f) for f in files)


def is_claude_running() -> bool:
    """Check if any claude process is currently running."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "claude"],
            capture_output=True, timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def last_dream_time() -> float:
    """Get timestamp of last dream run."""
    if not DREAM_STATE.exists():
        return 0.0
    try:
        import json
        state = json.loads(DREAM_STATE.read_text())
        ts = state.get("last_run", "")
        if ts:
            dt = datetime.fromisoformat(ts)
            return dt.timestamp()
    except (json.JSONDecodeError, ValueError):
        pass
    return 0.0


def should_dream(idle_minutes: float, min_gap_minutes: float = 60) -> tuple[bool, str]:
    """Check if conditions are met for dreaming.

    Returns (should_dream, reason).
    """
    now = time.time()

    # Check if sessions are idle
    last_write = newest_session_mtime()
    if last_write == 0:
        return False, "no session files found"

    idle_seconds = now - last_write
    idle_mins = idle_seconds / 60

    if idle_mins < idle_minutes:
        return False, f"sessions active {idle_mins:.0f}m ago (need {idle_minutes:.0f}m)"

    # Check if claude is currently running (don't dream during active use)
    if is_claude_running():
        return False, "claude process is running"

    # Check minimum gap since last dream
    last_dream = last_dream_time()
    if last_dream > 0:
        gap_mins = (now - last_dream) / 60
        if gap_mins < min_gap_minutes:
            return False, f"dreamed {gap_mins:.0f}m ago (need {min_gap_minutes:.0f}m gap)"

    # Check if corpus has new content since last dream
    # (dream_tool handles this too, but we can skip the startup cost)
    if last_dream > 0 and last_write < last_dream:
        return False, "no new sessions since last dream"

    return True, f"idle {idle_mins:.0f}m, last dream {(now - last_dream) / 60:.0f}m ago"


def run_dream(dream_minutes: int, verbose: bool = False):
    """Run cx dream with the specified time limit."""
    dream_seconds = dream_minutes * 60
    log(f"Starting dream ({dream_minutes}m / {dream_seconds}s)")

    cmd = [
        sys.executable, str(CONTINUUM_DIR / "dream_tool.py"),
        "--max-time", str(dream_seconds),
        "--force",
        "--wake-on-activity",
        "--report-file", str(Path.home() / ".continuum" / "last_dream_report.md"),
    ]
    if verbose:
        cmd.append("-v")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True, text=True,
            timeout=dream_seconds + 120,  # extra buffer for synthesis
        )
        # Log stderr (where dream progress goes)
        for line in result.stderr.strip().split("\n"):
            if line.strip():
                log(f"  {line.strip()}")
        if result.returncode == 0:
            log("Dream completed successfully")
        else:
            log(f"Dream exited with code {result.returncode}")
    except subprocess.TimeoutExpired:
        log(f"Dream timed out after {dream_seconds + 120}s")


def daemon_loop(idle_minutes: float, dream_minutes: int, check_interval: int = 60,
                min_gap: float = 60, verbose: bool = False):
    """Main daemon loop — check periodically, dream when idle."""
    log(f"Dream daemon started (idle={idle_minutes}m, dream={dream_minutes}m, "
        f"check={check_interval}s, gap={min_gap}m)")

    while True:
        should, reason = should_dream(idle_minutes, min_gap)
        if should:
            log(f"Triggering dream: {reason}")
            run_dream(dream_minutes, verbose)
        else:
            # Only log status every 10 checks to avoid noise
            pass

        time.sleep(check_interval)


def main():
    parser = argparse.ArgumentParser(description="Dream daemon — auto-trigger on idle")
    parser.add_argument("--idle", type=float, default=15,
                        help="Minutes of session inactivity before dreaming (default: 15)")
    parser.add_argument("--dream", type=int, default=30,
                        help="Minutes to run dream for (default: 30)")
    parser.add_argument("--check", type=int, default=60,
                        help="Seconds between idle checks (default: 60)")
    parser.add_argument("--gap", type=float, default=60,
                        help="Minimum minutes between dream runs (default: 60)")
    parser.add_argument("--once", action="store_true",
                        help="Check once and exit (for cron)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.once:
        should, reason = should_dream(args.idle, args.gap)
        if should:
            log(f"Triggering dream: {reason}")
            run_dream(args.dream, args.verbose)
        else:
            log(f"Skipping: {reason}")
        return

    daemon_loop(args.idle, args.dream, args.check, args.gap, args.verbose)


if __name__ == "__main__":
    main()
