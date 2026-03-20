#!/usr/bin/env python3
"""Standalone retrieval tool — retrieve context from continuum's corpus index.

Usage:
    python ~/+/continuum/retrieve_tool.py "token refresh auth flow"
    python ~/+/continuum/retrieve_tool.py --budget 10000 "query"
"""

import argparse
import sys
from pathlib import Path

# Ensure continuum modules are importable
_CONTINUUM_DIR = Path(__file__).resolve().parent
if str(_CONTINUUM_DIR) not in sys.path:
    sys.path.insert(0, str(_CONTINUUM_DIR))

from core.auto_ingest import auto_ingest
from core.index import load_index
from core.retrieval import ContextRetriever
from core.config import load_config
from core.tokens import count_tokens


def main():
    parser = argparse.ArgumentParser(description="Retrieve context from continuum corpus")
    parser.add_argument("query", help="Query string for retrieval")
    parser.add_argument("--budget", type=int, default=30000, help="Token budget (default: 30000)")
    parser.add_argument("--no-cull", action="store_true", help="Skip LLM precision filtering (faster, noisier)")
    parser.add_argument("--cull-factor", type=int, default=5, help="Over-retrieval multiplier (default: 5)")
    parser.add_argument("--code", action="store_true", help="Only return code chunks (skip sessions/markdown)")
    parser.add_argument("--project", default="", help="Filter to a specific project/codebase")
    parser.add_argument("--no-ingest", action="store_true", help="Skip auto-ingest check")
    args = parser.parse_args()

    if not args.no_ingest:
        auto_ingest()

    config = load_config(_CONTINUUM_DIR / "continuum.yaml")
    idx = load_index()

    if len(idx) == 0:
        # No corpus — try static sources only
        sources = config.get("context_sources", [])
        if not sources:
            print("[continuum:retrieve] empty index, no static sources", file=sys.stderr)
            sys.exit(0)

    sources = config.get("context_sources", [])
    retriever = ContextRetriever(sources=sources, index=idx)
    result = retriever.retrieve(
        query=args.query,
        token_budget=args.budget,
        conversation_tail="",
        cull=not args.no_cull,
        cull_factor=args.cull_factor,
        role_filter="code" if args.code else "",
        project_filter=args.project,
    )

    if not result.strip():
        print("[continuum:retrieve] no results", file=sys.stderr)
        sys.exit(0)

    entry_count = result.count("\n") + 1 if result.strip() else 0
    token_est = count_tokens(result)
    print(f"[continuum:retrieve] {entry_count} entries | {token_est / 1000:.1f}K tokens", file=sys.stderr)
    print(result)


if __name__ == "__main__":
    main()
