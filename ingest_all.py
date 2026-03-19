#!/usr/bin/env python3
"""One-shot ingest: CC sessions + markdown → corpus → index rebuild.

Usage:
    python ~/+/continuum/ingest_all.py              # ingest new, rebuild index
    python ~/+/continuum/ingest_all.py --force       # re-ingest everything
    python ~/+/continuum/ingest_all.py --no-embed    # skip embeddings (fast, no semantic search)
"""

import argparse
import subprocess
import sys
from pathlib import Path

_CONTINUUM_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Ingest all sources and rebuild index")
    parser.add_argument("--force", action="store_true", help="Re-ingest already-converted files")
    parser.add_argument("--no-embed", action="store_true", help="Skip embedding generation")
    parser.add_argument("--sources", nargs="+", help="Additional markdown source directories")
    args = parser.parse_args()

    embed_flag = [] if args.no_embed else ["--embed"]
    force_flag = ["--force"] if args.force else []
    sources_flag = []
    if args.sources:
        sources_flag = ["--sources"] + args.sources

    # 1. Ingest Claude Code sessions
    print("── CC sessions ──", file=sys.stderr)
    subprocess.run(
        [sys.executable, str(_CONTINUUM_DIR / "ingest.py"), "claude-code"] + embed_flag + force_flag,
        cwd=str(_CONTINUUM_DIR),
    )

    # 2. Ingest markdown (always --force since files change in place)
    print("\n── Markdown ──", file=sys.stderr)
    subprocess.run(
        [sys.executable, str(_CONTINUUM_DIR / "ingest.py"), "markdown"] + embed_flag + ["--force"] + sources_flag,
        cwd=str(_CONTINUUM_DIR),
    )

    # 3. Rebuild index
    print("\n── Index rebuild ──", file=sys.stderr)
    subprocess.run(
        [sys.executable, "-c", "from index import build_index; build_index(force=True)"],
        cwd=str(_CONTINUUM_DIR),
    )


if __name__ == "__main__":
    main()
