"""Generate hypothetical questions that corpus chunks answer.

Two backends:
- Local Qwen via Ollama (free, ~3s/chunk)
- Claude via `claude --print` (fast, batches 20 chunks per call)
"""

import json
import re
import subprocess
import sys
import urllib.request
import urllib.error


QUESTION_PROMPT_SINGLE = """Given the following text, generate 3-5 questions that this text directly answers. The questions should be natural questions a developer might search for. Return ONLY a JSON array of question strings.

Text:
{content}

Questions:"""


QUESTION_PROMPT_BATCH = """For each numbered text chunk below, generate 3-5 questions that the chunk directly answers. Questions should be natural queries a developer might search for.

Return a JSON object mapping chunk numbers to question arrays:
{{"0": ["question1", "question2", ...], "1": ["q1", "q2", ...], ...}}

{chunks}

JSON:"""


def generate_questions(
    content: str,
    model: str = "qwen2.5:7b",
    url: str = "http://localhost:11434/api/generate",
) -> list[str]:
    """Generate hypothetical questions for a single chunk via local Ollama."""
    if len(content.strip()) < 30:
        return []

    prompt = QUESTION_PROMPT_SINGLE.format(content=content[:4000])
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 512},
    }).encode("utf-8")

    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())

        text = result.get("response", "").strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        questions = json.loads(text)
        if isinstance(questions, list):
            return [
                q.strip() for q in questions
                if isinstance(q, str) and 10 < len(q.strip()) < 300
            ][:5]
    except (urllib.error.URLError, json.JSONDecodeError,
            TimeoutError, OSError, ValueError, KeyError):
        pass
    return []


def generate_questions_batch_claude(
    contents: list[str],
    batch_size: int = 20,
    model: str = "claude-haiku-4-5-20251001",
) -> list[list[str]]:
    """Generate questions for multiple chunks using Claude API via claude --print.

    Batches chunks into groups of batch_size, sends one API call per batch.
    ~20x faster than serial Ollama calls.
    """
    results: list[list[str]] = [[] for _ in contents]

    for batch_start in range(0, len(contents), batch_size):
        batch = contents[batch_start:batch_start + batch_size]

        # Build numbered chunk list
        chunk_text = ""
        for i, content in enumerate(batch):
            preview = content[:500]
            chunk_text += f"\n[{i}]\n{preview}\n"

        prompt = QUESTION_PROMPT_BATCH.format(chunks=chunk_text)

        try:
            result = subprocess.run(
                ["claude", "--print", "--model", model],
                input=prompt,
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode != 0:
                continue

            output = result.stdout.strip()
            # Extract JSON object from response
            # Find the outermost { ... }
            match = re.search(r'\{[\s\S]*\}', output)
            if not match:
                continue

            parsed = json.loads(match.group())

            for key, questions in parsed.items():
                try:
                    idx = int(key)
                except (ValueError, TypeError):
                    continue
                if not isinstance(questions, list):
                    continue
                if 0 <= idx < len(batch):
                    filtered = [
                        q.strip() for q in questions
                        if isinstance(q, str) and 10 < len(q.strip()) < 300
                    ][:5]
                    results[batch_start + idx] = filtered

        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
            print(f"  batch error at {batch_start}: {e}", file=sys.stderr)
            continue

        total_qs = sum(len(r) for r in results)
        print(f"  questions: {min(batch_start + batch_size, len(contents))}/{len(contents)} "
              f"({total_qs} generated)", file=sys.stderr)

    return results


def generate_questions_batch(
    contents: list[str],
    model: str = "qwen2.5:7b",
) -> list[list[str]]:
    """Generate questions for multiple chunks. Serial Qwen calls with progress."""
    results = []
    for i, content in enumerate(contents):
        qs = generate_questions(content, model)
        results.append(qs)
        if len(contents) > 10 and (i + 1) % 50 == 0:
            total_qs = sum(len(r) for r in results)
            print(f"  questions: {i+1}/{len(contents)} ({total_qs} generated)",
                  file=sys.stderr)
    if len(contents) > 10:
        total_qs = sum(len(r) for r in results)
        print(f"  questions: {len(contents)}/{len(contents)} ({total_qs} generated)",
              file=sys.stderr)
    return results
