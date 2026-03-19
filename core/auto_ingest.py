"""Auto-ingest: run ingest + index rebuild only if new sessions exist since last build.

Checks mtime of the newest CC session log against the index file.
If sessions are newer, runs a fast incremental ingest (new files only, with embeddings).
Designed to be called before retrieve/spoof operations.
"""

import glob
import os
import subprocess
import sys
from pathlib import Path

_CONTINUUM_DIR = Path(__file__).resolve().parent
_INDEX_PATH = Path.home() / ".continuum" / "index" / "corpus.meta.json"
_CC_PROJECTS_DIR = Path.home() / ".claude" / "projects"
_PROJECTS_DIR = Path.home() / "projects"


def _newest_mtime(pattern: str) -> float:
    """Find the newest mtime among files matching a glob pattern."""
    files = glob.glob(pattern, recursive=True)
    if not files:
        return 0.0
    return max(os.path.getmtime(f) for f in files)


def needs_ingest() -> bool:
    """Check if new session logs or markdown files exist since last index build."""
    if not _INDEX_PATH.exists():
        return True

    index_mtime = os.path.getmtime(_INDEX_PATH)

    # Check CC session logs
    cc_mtime = _newest_mtime(str(_CC_PROJECTS_DIR / "*" / "*.jsonl"))
    if cc_mtime > index_mtime:
        return True

    # Check CLAUDE.md files
    md_mtime = _newest_mtime(str(_PROJECTS_DIR / "**/CLAUDE.md"))
    if md_mtime > index_mtime:
        return True

    return False


def auto_ingest(quiet: bool = True) -> None:
    """Run incremental ingest + index rebuild if needed."""
    if not needs_ingest():
        return

    stderr = subprocess.DEVNULL if quiet else None

    # Ingest new CC sessions (incremental, with embeddings)
    subprocess.run(
        [sys.executable, str(_CONTINUUM_DIR / "ingest.py"), "claude-code", "--embed"],
        cwd=str(_CONTINUUM_DIR),
        stdout=subprocess.DEVNULL,
        stderr=stderr,
    )

    # Re-ingest markdown (always force — files change in place)
    subprocess.run(
        [sys.executable, str(_CONTINUUM_DIR / "ingest.py"), "markdown", "--embed", "--force"],
        cwd=str(_CONTINUUM_DIR),
        stdout=subprocess.DEVNULL,
        stderr=stderr,
    )

    # Rebuild index
    subprocess.run(
        [sys.executable, "-c", "from index import build_index; build_index(force=True)"],
        cwd=str(_CONTINUUM_DIR),
        stdout=subprocess.DEVNULL,
        stderr=stderr,
    )

    if not quiet:
        print("[continuum] auto-ingest complete", file=sys.stderr)
