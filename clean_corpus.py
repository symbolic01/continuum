#!/usr/bin/env python3
"""Clean low-signal entries from the continuum corpus.

Removes assistant entries that are pure tool-use bracket summaries
([Read: path], [TaskUpdate], etc.) or known filler ("No response requested.").
These waste embeddings and pollute semantic search.

Usage:
    python clean_corpus.py              # dry run — show what would be removed
    python clean_corpus.py --apply      # actually rewrite corpus files
    python clean_corpus.py --apply -v   # verbose: show each removed entry
"""

import argparse
import json
import sys
from pathlib import Path

CORPUS_DIR = Path.home() / ".continuum" / "corpus"

FILLER = {
    "No response requested.",
    "I have this context. Ready to continue.",
}


def is_low_signal(entry: dict) -> bool:
    """Check if a corpus entry is low-signal noise."""
    role = entry.get("role", "")
    content = entry.get("content", "").strip()
    if not content:
        return True
    # Never filter kernels or chains
    if role in ("kernel", "chain"):
        return False
    if role == "assistant":
        # Pure bracket summaries: [Read: path], [TaskUpdate], [Agent], etc.
        if content.startswith("[") and content.endswith("]") and len(content) < 200:
            return True
        if content in FILLER:
            return True
    if role == "user":
        # Tool result rejection boilerplate
        if "The user doesn't want to proceed with this tool use" in content and len(content) < 400:
            return True
        # Empty tool results
        if content.startswith("[{'tool_use_id':") and len(content) < 80:
            return True
    return False


def clean_file(path: Path, apply: bool, verbose: bool) -> tuple[int, int]:
    """Clean a single JSONL file. Returns (kept, removed) counts."""
    kept = []
    removed = 0

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                kept.append(line)
                continue

            if is_low_signal(entry):
                removed += 1
                if verbose:
                    content = entry.get("content", "")[:60]
                    print(f"  - [{entry.get('thread','?')}] {content}")
            else:
                kept.append(json.dumps(entry))

    if apply and removed > 0:
        with open(path, "w") as f:
            for line in kept:
                f.write(line + "\n")

    return len(kept), removed


def main():
    parser = argparse.ArgumentParser(description="Clean low-signal corpus entries")
    parser.add_argument("--apply", action="store_true",
                        help="Actually rewrite files (default: dry run)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show each removed entry")
    args = parser.parse_args()

    if not CORPUS_DIR.exists():
        print(f"Corpus not found at {CORPUS_DIR}", file=sys.stderr)
        sys.exit(1)

    total_kept = 0
    total_removed = 0
    files_modified = 0

    # Process all JSONL files (skip _chains — those are dream output)
    corpus_files = sorted(CORPUS_DIR.rglob("*.jsonl"))

    for path in corpus_files:
        kept, removed = clean_file(path, args.apply, args.verbose)
        if removed > 0:
            files_modified += 1
            if not args.verbose:
                rel = path.relative_to(CORPUS_DIR)
                print(f"  {rel}: {removed} removed, {kept} kept")
        total_kept += kept
        total_removed += removed

    mode = "Cleaned" if args.apply else "Would clean"
    print(f"\n{mode}: {total_removed:,} entries removed from {files_modified} files")
    print(f"Remaining: {total_kept:,} entries")

    if args.apply and total_removed > 0:
        print(f"\nRebuilding index...")
        # Add continuum to path and rebuild
        continuum_dir = Path(__file__).resolve().parent
        if str(continuum_dir) not in sys.path:
            sys.path.insert(0, str(continuum_dir))
        from core.index import build_index
        build_index(force=True)
        print("Done.")
    elif not args.apply and total_removed > 0:
        print(f"\nRun with --apply to execute. No files were modified.")


if __name__ == "__main__":
    main()
