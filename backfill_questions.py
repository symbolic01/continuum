#!/usr/bin/env python3
"""One-time backfill: generate question embeddings for existing corpus entries.

Reads corpus JSONL files, generates hypothetical questions for entries
that have content embeddings but no question embeddings, embeds the
questions, and rewrites the JSONL files in place.

Resumable — skips entries that already have question_embeddings.
Parallel — 4 concurrent Qwen calls for ~3-4x throughput.
Filtered — skips low-value content (short, tool_result, filler).

Usage:
    python backfill_questions.py              # backfill all
    python backfill_questions.py --limit 100  # process first 100 entries
    python backfill_questions.py --dry-run    # count entries to process
    python backfill_questions.py --workers 2  # control parallelism
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_CONTINUUM_DIR = Path(__file__).resolve().parent
if str(_CONTINUUM_DIR) not in sys.path:
    sys.path.insert(0, str(_CONTINUUM_DIR))

from core.questions import generate_questions
from core.embeddings import embed_batch

CORPUS_DIR = Path.home() / ".continuum" / "corpus"

# Roles worth generating questions for
VALUABLE_ROLES = {"user", "assistant", "context", "kernel", "code", "plan"}

# Skip content matching these patterns
SKIP_PATTERNS = [
    "tool_use_id",       # raw tool result JSON
    "task-notification",  # CC task notifications
    "[{",                # JSON arrays (tool results)
    "Temporal link",     # dream temporal links
]


def _is_worth_processing(entry: dict) -> bool:
    """Filter: only generate questions for high-value chunks."""
    if not entry.get("embedding"):
        return False
    if entry.get("question_embeddings"):
        return False

    content = entry.get("content", "")
    if len(content.strip()) < 80:  # stricter than before — skip very short
        return False

    role = entry.get("role", "")
    if role and role not in VALUABLE_ROLES:
        return False

    # Skip content that starts with noise patterns
    for pat in SKIP_PATTERNS:
        if content.strip().startswith(pat):
            return False

    return True


def _process_one(content: str) -> tuple[list[str], list[dict]]:
    """Generate questions + embed for a single chunk. Thread-safe."""
    questions = generate_questions(content)
    if not questions:
        return [], []

    q_vecs = embed_batch(questions)
    q_embeddings = []
    for q_text, q_vec in zip(questions, q_vecs):
        if q_vec is not None:
            q_embeddings.append({
                "text": q_text,
                "embedding": q_vec,
            })
    return questions, q_embeddings


def backfill(limit: int = 0, dry_run: bool = False, workers: int = 4):
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
                if _is_worth_processing(entry):
                    to_process += 1

    print(f"Corpus: {total} entries, {to_process} need question embeddings (after filtering)")
    if dry_run:
        est_hours = to_process * 1.5 / 3600 / workers
        print(f"Estimated time: {est_hours:.1f} hours ({workers} workers)")
        return

    # Second pass: generate and write
    processed = 0
    skipped = 0
    questions_generated = 0
    t_start = time.monotonic()

    for cf in corpus_files:
        raw_lines = []
        with open(cf) as f:
            raw_lines = f.readlines()

        # Identify entries that need processing in this file
        entries_to_process = []  # (line_idx, entry, content)
        parsed_entries = []  # (line_idx, entry_or_None)

        for line_idx, raw_line in enumerate(raw_lines):
            stripped = raw_line.strip()
            if not stripped:
                parsed_entries.append((line_idx, None))
                continue
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError:
                parsed_entries.append((line_idx, None))
                continue

            parsed_entries.append((line_idx, entry))

            if not _is_worth_processing(entry):
                continue

            if limit and (processed + len(entries_to_process)) >= limit:
                continue

            entries_to_process.append((line_idx, entry, entry.get("content", "")))

        if not entries_to_process:
            continue

        # Process in parallel
        results = {}  # line_idx → (questions, q_embeddings)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for line_idx, entry, content in entries_to_process:
                fut = pool.submit(_process_one, content)
                futures[fut] = line_idx

            for fut in as_completed(futures):
                line_idx = futures[fut]
                try:
                    questions, q_embeddings = fut.result()
                    if q_embeddings:
                        results[line_idx] = (questions, q_embeddings)
                except Exception:
                    pass

        # Apply results and rewrite file
        modified = False
        output_lines = []

        for line_idx, raw_line in enumerate(raw_lines):
            stripped = raw_line.strip()
            if not stripped:
                output_lines.append(raw_line.rstrip("\n"))
                continue

            if line_idx in results:
                try:
                    entry = json.loads(stripped)
                    questions, q_embeddings = results[line_idx]
                    entry["questions"] = questions
                    entry["question_embeddings"] = q_embeddings
                    output_lines.append(json.dumps(entry, ensure_ascii=False))
                    modified = True
                    questions_generated += len(q_embeddings)
                    processed += 1
                except json.JSONDecodeError:
                    output_lines.append(raw_line.rstrip("\n"))
            else:
                output_lines.append(raw_line.rstrip("\n"))
                # Count processed if it was in our batch but got no results
                for li, _, _ in entries_to_process:
                    if li == line_idx and line_idx not in results:
                        processed += 1
                        break

        if modified:
            with open(cf, "w") as f:
                f.write("\n".join(output_lines) + "\n")

        elapsed = time.monotonic() - t_start
        rate = processed / elapsed if elapsed > 0 else 0
        remaining = (to_process - processed) / rate if rate > 0 else 0
        print(f"  {processed}/{to_process} ({questions_generated} questions, "
              f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)",
              file=sys.stderr)

    elapsed = time.monotonic() - t_start
    print(f"\nDone: {processed} entries processed, {questions_generated} question embeddings in {elapsed:.0f}s")

    # Rebuild index
    if questions_generated > 0:
        print("\nRebuilding index...")
        from core.index import build_index
        build_index(force=True)


def main():
    parser = argparse.ArgumentParser(description="Backfill question embeddings")
    parser.add_argument("--limit", type=int, default=0, help="Max entries to process (0=all)")
    parser.add_argument("--dry-run", action="store_true", help="Count only, don't process")
    parser.add_argument("--workers", type=int, default=4, help="Parallel Qwen workers (default: 4)")
    args = parser.parse_args()

    backfill(limit=args.limit, dry_run=args.dry_run, workers=args.workers)


if __name__ == "__main__":
    main()
