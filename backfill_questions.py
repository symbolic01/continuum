#!/usr/bin/env python3
"""Backfill question embeddings for existing corpus entries.

Uses Claude API (batches 20 chunks per call) for fast question generation,
then local Ollama for embedding. Resumable — skips entries with existing
question_embeddings.

Usage:
    python backfill_questions.py              # backfill all via Claude
    python backfill_questions.py --local      # use local Qwen instead
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

from core.questions import generate_questions, generate_questions_batch_claude
from core.embeddings import embed_batch

CORPUS_DIR = Path.home() / ".continuum" / "corpus"

VALUABLE_ROLES = {"user", "assistant", "context", "kernel", "code", "plan"}
SKIP_PATTERNS = ["tool_use_id", "task-notification", "[{", "Temporal link"]


def _is_worth_processing(entry: dict) -> bool:
    if not entry.get("embedding"):
        return False
    if entry.get("question_embeddings"):
        return False
    content = entry.get("content", "")
    if len(content.strip()) < 80:
        return False
    role = entry.get("role", "")
    if role and role not in VALUABLE_ROLES:
        return False
    for pat in SKIP_PATTERNS:
        if content.strip().startswith(pat):
            return False
    return True


def backfill(limit: int = 0, dry_run: bool = False, local: bool = False):
    corpus_files = sorted(
        f for f in CORPUS_DIR.rglob("*.jsonl")
        if "_archive" not in str(f)
    )

    # Scan: collect all entries needing processing
    # Store as (file_path, line_number, content) for later writeback
    work_items = []  # (corpus_file, line_idx, content)
    total = 0

    for cf in corpus_files:
        with open(cf) as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total += 1
                if _is_worth_processing(entry):
                    work_items.append((cf, line_idx, entry.get("content", "")))

    to_process = len(work_items)
    if limit:
        work_items = work_items[:limit]
        to_process = len(work_items)

    print(f"Corpus: {total} entries, {to_process} need question embeddings")
    if dry_run:
        if local:
            est = to_process * 4 / 3600
            print(f"Estimated: {est:.1f} hours (local Qwen, serial)")
        else:
            est = (to_process / 20) * 4 / 3600  # ~4s per batch of 20
            print(f"Estimated: {est:.1f} hours (Claude, batches of 20)")
        return

    # Generate questions
    contents = [c for _, _, c in work_items]
    t_start = time.monotonic()

    if local:
        print("Using local Qwen (serial)...")
        all_questions = []
        for i, content in enumerate(contents):
            qs = generate_questions(content)
            all_questions.append(qs)
            if (i + 1) % 50 == 0:
                elapsed = time.monotonic() - t_start
                rate = (i + 1) / elapsed
                remaining = (len(contents) - i - 1) / rate
                total_qs = sum(len(q) for q in all_questions)
                print(f"  {i+1}/{len(contents)} ({total_qs} questions, ~{remaining:.0f}s remaining)",
                      file=sys.stderr)
    else:
        print(f"Using Claude API (batches of 20)...")
        all_questions = generate_questions_batch_claude(contents, batch_size=20)

    t_questions = time.monotonic() - t_start
    total_qs = sum(len(q) for q in all_questions)
    print(f"Generated {total_qs} questions in {t_questions:.0f}s")

    # Embed all questions
    print("Embedding questions...")
    all_q_texts = []
    q_map = []  # (work_idx, q_idx_in_entry)
    for i, questions in enumerate(all_questions):
        for j, q in enumerate(questions):
            all_q_texts.append(q)
            q_map.append((i, j))

    all_q_vecs = embed_batch(all_q_texts) if all_q_texts else []
    t_embed = time.monotonic() - t_start - t_questions
    print(f"Embedded {len(all_q_texts)} questions in {t_embed:.0f}s")

    # Build per-entry question_embeddings
    entry_qe = [[] for _ in work_items]  # list of {text, embedding} per entry
    for (work_idx, q_idx), vec in zip(q_map, all_q_vecs):
        if vec is not None:
            entry_qe[work_idx].append({
                "text": all_questions[work_idx][q_idx],
                "embedding": vec,
            })

    # Write back: group by file, rewrite in place
    print("Writing to corpus files...")
    files_to_update = {}  # file_path → {line_idx: (questions, q_embeddings)}
    for i, (cf, line_idx, _) in enumerate(work_items):
        if entry_qe[i]:
            if cf not in files_to_update:
                files_to_update[cf] = {}
            files_to_update[cf][line_idx] = (all_questions[i], entry_qe[i])

    files_modified = 0
    entries_written = 0
    for cf, updates in files_to_update.items():
        with open(cf) as f:
            raw_lines = f.readlines()

        modified = False
        output_lines = []
        for line_idx, raw_line in enumerate(raw_lines):
            if line_idx in updates:
                try:
                    entry = json.loads(raw_line.strip())
                    questions, q_embeddings = updates[line_idx]
                    entry["questions"] = questions
                    entry["question_embeddings"] = q_embeddings
                    output_lines.append(json.dumps(entry, ensure_ascii=False))
                    modified = True
                    entries_written += 1
                except json.JSONDecodeError:
                    output_lines.append(raw_line.rstrip("\n"))
            else:
                output_lines.append(raw_line.rstrip("\n"))

        if modified:
            with open(cf, "w") as f:
                f.write("\n".join(output_lines) + "\n")
            files_modified += 1

    elapsed = time.monotonic() - t_start
    print(f"\nDone: {entries_written} entries updated across {files_modified} files in {elapsed:.0f}s")
    print(f"  {total_qs} questions generated, {sum(len(qe) for qe in entry_qe)} embedded")

    if entries_written > 0:
        print("\nRebuilding index...")
        from core.index import build_index
        build_index(force=True)


def main():
    parser = argparse.ArgumentParser(description="Backfill question embeddings")
    parser.add_argument("--limit", type=int, default=0, help="Max entries to process (0=all)")
    parser.add_argument("--dry-run", action="store_true", help="Count only, don't process")
    parser.add_argument("--local", action="store_true", help="Use local Qwen instead of Claude API")
    args = parser.parse_args()

    backfill(limit=args.limit, dry_run=args.dry_run, local=args.local)


if __name__ == "__main__":
    main()
