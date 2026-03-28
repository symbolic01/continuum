"""Generate hypothetical questions that a chunk answers, using local Qwen."""

import json
import re
import sys
import urllib.request
import urllib.error


QUESTION_PROMPT = """Given the following text, generate 3-5 questions that this text directly answers. The questions should be natural questions a developer might search for. Return ONLY a JSON array of question strings.

Text:
{content}

Questions:"""


def generate_questions(
    content: str,
    model: str = "qwen2.5:7b",
    url: str = "http://localhost:11434/api/generate",
) -> list[str]:
    """Generate hypothetical questions for a content chunk.

    Returns list of question strings, or empty list on failure.
    """
    if len(content.strip()) < 30:
        return []

    prompt = QUESTION_PROMPT.format(content=content[:4000])
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
        # Strip markdown code fences
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
