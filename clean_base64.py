#!/usr/bin/env python3
"""Remove base64-encoded data entries from the continuum corpus.

Finds and removes entries containing base64 image/binary data that
got ingested before the ingest filter was added.

Usage:
    python clean_base64.py              # dry run
    python clean_base64.py --apply      # actually remove + rebuild index
    python clean_base64.py --apply -v   # verbose: show each removed entry
"""

import argparse
import json
import sys
from pathlib import Path

CORPUS_DIR = Path.home() / ".continuum" / "corpus"

# Patterns that indicate base64 or binary data
BASE64_MARKERS = [
    "'base64'",
    '"base64"',
    "iVBORw0KGgo",     # PNG header in base64
    "/9j/4AAQ",         # JPEG header in base64
    "data:image/",
    "R0lGODlh",         # GIF header in base64
]

MIN_SIZE = 2000  # only flag entries over this size (small base64 refs are fine)


def is_base64_entry(entry: dict) -> bool:
    """Check if an entry contains base64 encoded data."""
    content = entry.get("content", "")
    if len(content) < MIN_SIZE:
        return False
    for marker in BASE64_MARKERS:
        if marker in content:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Remove base64 data from corpus")
    parser.add_argument("--apply", action="store_true",
                        help="Actually rewrite files (default: dry run)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show each removed entry")
    args = parser.parse_args()

    if not CORPUS_DIR.exists():
        print(f"Corpus not found at {CORPUS_DIR}", file=sys.stderr)
        sys.exit(1)

    total_removed = 0
    total_kept = 0
    total_bytes_freed = 0
    files_modified = 0

    for path in sorted(CORPUS_DIR.rglob("*.jsonl")):
        kept = []
        removed = 0
        bytes_freed = 0

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

                if is_base64_entry(entry):
                    removed += 1
                    bytes_freed += len(line)
                    if args.verbose:
                        uid = entry.get("uid", "?")
                        thread = entry.get("thread", "?")
                        role = entry.get("role", "?")
                        size = len(entry.get("content", ""))
                        print(f"  - [{uid}] {thread} ({role}) {size:,} chars")
                else:
                    kept.append(json.dumps(entry))

        if removed > 0:
            files_modified += 1
            rel = path.relative_to(CORPUS_DIR)
            print(f"  {rel}: {removed} removed ({bytes_freed:,} bytes), {len(kept)} kept")

            if args.apply:
                with open(path, "w") as f:
                    for line in kept:
                        f.write(line + "\n")

        total_removed += removed
        total_kept += len(kept)
        total_bytes_freed += bytes_freed

    mode = "Removed" if args.apply else "Would remove"
    print(f"\n{mode}: {total_removed} entries from {files_modified} files "
          f"({total_bytes_freed / 1024 / 1024:.1f} MB freed)")
    print(f"Remaining: {total_kept:,} entries")

    if args.apply and total_removed > 0:
        print(f"\nRebuilding index...")
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
