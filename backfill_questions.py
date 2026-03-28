#!/usr/bin/env python3
"""One-time backfill: generate question embeddings for existing corpus entries.

Reads corpus JSONL files, generates hypothetical questions for entries
that have content embeddings but no question embeddings, embeds the
questions, and rewrites the JSONL files in place.

Resumable — skips entries that already have question_embeddings.

Usage:
    python backfill_questions.py              # backfill all
    python backfill_questions.py --limit 100  # process first 100 entries
    python backfill_questions.py --dry-run    # count entries to process
"""

import argparse
import json
import sys
import time
from pathlib import Path

_CONTINUUM_DIR = Path(__file__).resolve().parent
if str(_CONTINUUM_DIR) not in sys.path:
    sys.path.insert(0, str(_CONTINUUM_DIR))

from core.questions import generate_questions
from core.embeddings import embed_batch

CORPUS_DIR = Path.home() / ".continuum" / "corpus"


def backfill(limit: int = 0, dry_run: bool = False):
    """Backfill question embeddings on existing corpus."""
    corpus_files = sorted(
        f for f in CORPUS_DIR.rglob("*.jsonl")
        if "_archive" not in str(f)
    )

    # First pass: count what needs processing
    to_process = 0
    total = 0
    for cf in corpus_files:
        with open(cf) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total += 1
                # Only process entries that have content embedding but no questions
                if entry.get("embedding") and not entry.get("question_embeddings"):
                    content = entry.get("content", "")
                    if len(content.strip()) >= 30:
                        to_process += 1

    print(f"Corpus: {total} entries, {to_process} need question embeddings")
    if dry_run:
        est_hours = to_process / 3600  # ~1s per entry
        print(f"Estimated time: {est_hours:.1f} hours")
        return

    # Second pass: generate and write
    processed = 0
    questions_generated = 0
    t_start = time.monotonic()

    for cf in corpus_files:
        lines = []
        modified = False

        with open(cf) as f:
            raw_lines = f.readlines()

        for raw_line in raw_lines:
            raw_line = raw_line.rstrip("\n")
            if not raw_line.strip():
                lines.append(raw_line)
                continue

            try:
                entry = json.loads(raw_line)
            except json.JSONDecodeError:
                lines.append(raw_line)
                continue

            # Skip if already has questions or no content embedding
            if entry.get("question_embeddings") or not entry.get("embedding"):
                lines.append(raw_line)
                continue

            content = entry.get("content", "")
            if len(content.strip()) < 30:
                lines.append(raw_line)
                continue

            if limit and processed >= limit:
                lines.append(raw_line)
                continue

            # Generate questions
            questions = generate_questions(content)
            if questions:
                # Embed the questions
                q_vecs = embed_batch(questions)
                q_embeddings = []
                for q_text, q_vec in zip(questions, q_vecs):
                    if q_vec is not None:
                        q_embeddings.append({
                            "text": q_text,
                            "embedding": q_vec,
                        })

                if q_embeddings:
                    entry["questions"] = questions
                    entry["question_embeddings"] = q_embeddings
                    modified = True
                    questions_generated += len(q_embeddings)

            processed += 1

            if processed % 50 == 0:
                elapsed = time.monotonic() - t_start
                rate = processed / elapsed
                remaining = (to_process - processed) / rate if rate > 0 else 0
                print(f"  {processed}/{to_process} ({questions_generated} questions, "
                      f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)",
                      file=sys.stderr)

            lines.append(json.dumps(entry, ensure_ascii=False))

        # Write back if modified
        if modified:
            with open(cf, "w") as f:
                f.write("\n".join(lines) + "\n")

    elapsed = time.monotonic() - t_start
    print(f"\nDone: {processed} entries processed, {questions_generated} question embeddings generated in {elapsed:.0f}s")

    # Rebuild index
    if processed > 0:
        print("\nRebuilding index...")
        from core.index import build_index
        build_index(force=True)


def main():
    parser = argparse.ArgumentParser(description="Backfill question embeddings")
    parser.add_argument("--limit", type=int, default=0, help="Max entries to process (0=all)")
    parser.add_argument("--dry-run", action="store_true", help="Count only, don't process")
    args = parser.parse_args()

    backfill(limit=args.limit, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
