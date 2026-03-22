#!/usr/bin/env python3
"""Standalone dream tool — offline integration pipeline for the continuum corpus.

Usage:
    python ~/+/continuum/dream_tool.py              # full pipeline
    python ~/+/continuum/dream_tool.py --dry-run     # show what would happen
    python ~/+/continuum/dream_tool.py --report       # emit markdown report
    python ~/+/continuum/dream_tool.py --max-passes 3 # cap passes
"""

import argparse
import sys
from pathlib import Path

# Ensure continuum modules are importable
_CONTINUUM_DIR = Path(__file__).resolve().parent
if str(_CONTINUUM_DIR) not in sys.path:
    sys.path.insert(0, str(_CONTINUUM_DIR))

from core.auto_ingest import auto_ingest, needs_ingest
from core.config import load_config
from core.dream import DreamEngine
from core.index import build_index


def main():
    parser = argparse.ArgumentParser(description="Dream — offline corpus integration")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen, no writes")
    parser.add_argument("--max-passes", type=int, default=50,
                        help="Max integration passes (default: 50)")
    parser.add_argument("--max-time", type=int, default=1800,
                        help="Max wall time in seconds (default: 1800)")
    parser.add_argument("--max-tokens", type=int, default=500_000,
                        help="Max LLM tokens (default: 500000)")
    parser.add_argument("--max-chains", type=int, default=500,
                        help="Max chains to create (default: 500)")
    parser.add_argument("--model", default="",
                        help="Override dream model (default: from config)")
    parser.add_argument("--no-ingest", action="store_true",
                        help="Skip ingestion housekeeping")
    parser.add_argument("--report", action="store_true",
                        help="Print markdown report to stdout")
    parser.add_argument("--report-file", type=str, default="",
                        help="Write markdown report to file")
    parser.add_argument("--no-temporal", action="store_true",
                        help="Skip temporal reconnection phase")
    parser.add_argument("--no-synthesis", action="store_true",
                        help="Skip synthesis pass (Claude)")
    parser.add_argument("--stop-on-convergence", action="store_true",
                        help="Stop when convergence detected (default: use time/token caps)")
    parser.add_argument("--force", action="store_true",
                        help="Run even if corpus hasn't changed since last dream")
    parser.add_argument("--wake-on-activity", action="store_true",
                        help="Stop integration if claude process detected (used by daemon)")
    parser.add_argument("--focus-project", type=str, default="",
                        help="Bias seeding toward this project (auto-selects if empty)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    args = parser.parse_args()

    config = load_config(_CONTINUUM_DIR / "continuum.yaml")

    # Apply config defaults for dream section
    dream_config = config.get("dream", {})
    max_passes = args.max_passes if args.max_passes != 50 else dream_config.get("max_passes", 50)
    max_time = args.max_time if args.max_time != 1800 else dream_config.get("max_wall_time_seconds", 1800)
    max_tokens = args.max_tokens if args.max_tokens != 500_000 else dream_config.get("max_llm_tokens", 500_000)
    max_chains = args.max_chains if args.max_chains != 500 else dream_config.get("max_chains_per_run", 500)

    engine = DreamEngine(
        config=config,
        dry_run=args.dry_run,
        max_passes=max_passes,
        max_wall_time=max_time,
        max_llm_tokens=max_tokens,
        max_chains=max_chains,
        cluster_size_min=dream_config.get("cluster_size_min", 3),
        cluster_size_max=dream_config.get("cluster_size_max", 8),
        model=args.model,
        verbose=args.verbose,
        focus_project=args.focus_project,
    )
    engine._check_wake_up = args.wake_on_activity

    # Phase 0: Check if dreaming is needed
    if not args.force and not args.dry_run and not engine.should_dream():
        print("[dream] Corpus unchanged since last dream — skipping", file=sys.stderr)
        return

    # Phase 1: Ingestion housekeeping
    if not args.no_ingest:
        if needs_ingest():
            print("[dream] Running ingestion housekeeping...", file=sys.stderr)
            auto_ingest(quiet=not args.verbose)
        elif args.verbose:
            print("[dream] Corpus is current — skipping ingest", file=sys.stderr)

    # Load corpus
    engine.load_corpus()

    if not engine.all_metadata:
        print("[dream] Empty corpus — nothing to dream about", file=sys.stderr)
        return

    # Phase 2: Continuous integration
    stats = engine.run_integration_passes()

    # Phase 3: Temporal reconnection
    temporal_links = []
    if not args.no_temporal and not args.dry_run:
        temporal_links = engine.run_temporal_reconnection(
            min_distance_days=dream_config.get("temporal_distance_days", 14),
            similarity_threshold=dream_config.get("temporal_similarity_threshold", 0.7),
        )

    # Phase 4: Post-processing
    if not args.dry_run:
        all_new = engine.new_chains + temporal_links
        if all_new:
            engine.write_chains(all_new)
            engine.build_xrefs()

            # Rebuild index to incorporate chain chunks
            print("[dream] Rebuilding index...", file=sys.stderr)
            build_index(force=True)

        engine.save_state(stats)

        # Git commit the corpus changes
        engine.git_commit()

        # Synthesis pass — compress chains into human-meaningful kernels
        synthesis = None
        if not args.no_synthesis:
            synthesis = engine.run_synthesis()

        # Generate report (includes ALL chains + synthesis)
        report = engine.generate_report(stats, temporal_links, synthesis)

        # Copy report JSON next to HTML for serving
        import shutil
        html_dir = _CONTINUUM_DIR
        report_copy = html_dir / "dream_report.json"
        from core.dream import DREAM_REPORT_PATH
        shutil.copy2(str(DREAM_REPORT_PATH), str(report_copy))

        if args.report:
            engine.print_report_markdown(report)
            print(f"\n[dream] HTML report: file://{html_dir}/dream_report.html",
                  file=sys.stderr)

        if args.report_file:
            import io
            buf = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = buf
            engine.print_report_markdown(report)
            sys.stdout = old_stdout
            Path(args.report_file).write_text(buf.getvalue())
            print(f"[dream] Report written to {args.report_file}", file=sys.stderr)
    else:
        # Dry run summary
        print(f"\n[dream] Dry run complete. Would process "
              f"{len(engine.all_metadata)} entries.", file=sys.stderr)

    # Summary
    print(f"\n[dream] Done: {stats.get('passes', 0)} clusters, "
          f"{stats.get('chains_created', 0)} integration chains"
          f"{f', {len(temporal_links)} temporal links' if temporal_links else ''}, "
          f"{stats.get('tokens_used', 0)} tokens, "
          f"{stats.get('elapsed_seconds', 0):.0f}s",
          file=sys.stderr)


if __name__ == "__main__":
    main()
