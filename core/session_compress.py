"""Session compression — LLM-powered narrative distillation of messy sessions.

Takes raw conversation turns from a CC session and compresses them into
a clean turn-based story: what was attempted, what failed, what worked.
"""

import json
import subprocess
import sys


_SYSTEM_PROMPT = """\
Compress this Claude Code session into a clean turn-based narrative.
The session contains iterative work — debugging, trial-and-error, dead ends.
Distill it into the story of what happened.

Rules:
- Output JSON: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
- 5-20 turns total. Each is a separate message (renders as its own bullet in the UI)
- Start with what was being built/attempted
- Summarize failed approaches as brief exchanges ("Tried X" / "Failed: Y")
- End with the WORKING solution in full technical detail
- Preserve file paths, function names, key code patterns — enough to continue working
- Assistant turns should be concise but not telegraphic — natural conversation tone
- Do NOT merge everything into one big block. Keep turns short and focused
- Output ONLY the JSON array, no markdown fencing or commentary"""


def _truncate_middle(turns: list[dict], max_chars: int = 120_000) -> list[dict]:
    """If turns exceed max_chars, keep first 15% + last 50%, drop the middle."""
    total = sum(len(t.get("content", "")) for t in turns)
    if total <= max_chars:
        return turns

    # Walk from start to find 15% boundary
    head_budget = int(max_chars * 0.15)
    head_chars = 0
    head_end = 0
    for i, t in enumerate(turns):
        head_chars += len(t.get("content", ""))
        if head_chars >= head_budget:
            head_end = i + 1
            break

    # Walk from end to find 50% boundary
    tail_budget = int(max_chars * 0.50)
    tail_chars = 0
    tail_start = len(turns)
    for i in range(len(turns) - 1, -1, -1):
        tail_chars += len(turns[i].get("content", ""))
        if tail_chars >= tail_budget:
            tail_start = i
            break

    if tail_start <= head_end:
        return turns  # overlap — just return all

    omitted = tail_start - head_end
    marker = {"role": "user", "content": f"[... {omitted} turns of iterative work omitted ...]"}
    return turns[:head_end] + [marker] + turns[tail_start:]


def compress_session(
    turns: list[dict],
    user_prompt: str = "",
    model: str = "",
    timeout: int = 120,
) -> list[dict]:
    """Compress raw session turns into a clean narrative via LLM.

    Args:
        turns: list of {role, content} dicts from the raw session
        user_prompt: optional steering text appended to the compression prompt
        model: model to use for compression
        timeout: subprocess timeout in seconds

    Returns:
        Compressed list of {role, content} dicts (5-20 turns)
    """
    if not model:
        from .config import get_model
        model = get_model("compress")

    if len(turns) <= 20:
        return turns  # already short enough

    # Truncate if too large
    truncated = _truncate_middle(turns)

    # Build prompt
    session_json = json.dumps(truncated, indent=None)
    prompt = _SYSTEM_PROMPT
    if user_prompt:
        prompt += f"\n\nAdditional guidance: {user_prompt}"
    prompt += f"\n\n<session>\n{session_json}\n</session>"

    try:
        result = subprocess.run(
            ["claude", "--print", "--model", model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            print(f"[continuum:compress] LLM error: {(result.stderr or '').strip()[:200]}", file=sys.stderr)
            return turns

        output = result.stdout.strip()
        compressed = _parse_json_turns(output)
        if compressed:
            return compressed

        print("[continuum:compress] failed to parse LLM output, using raw turns", file=sys.stderr)
        return turns

    except subprocess.TimeoutExpired:
        print("[continuum:compress] timeout, using raw turns", file=sys.stderr)
        return turns
    except Exception as e:
        print(f"[continuum:compress] error: {e}", file=sys.stderr)
        return turns


def _parse_json_turns(text: str) -> list[dict] | None:
    """Parse JSON array of turns from LLM output, with fallback extraction."""
    # Try direct parse
    try:
        data = json.loads(text)
        if isinstance(data, list) and all(isinstance(t, dict) and "role" in t for t in data):
            return data
    except json.JSONDecodeError:
        pass

    # Try to find JSON array in the text (LLM might have wrapped it)
    import re
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list) and all(isinstance(t, dict) and "role" in t for t in data):
                return data
        except json.JSONDecodeError:
            pass

    return None
