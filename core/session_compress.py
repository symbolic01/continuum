"""Session compression — LLM-powered narrative distillation of messy sessions.

Takes raw conversation turns from a CC session and compresses them into
a clean turn-based story: what was attempted, what failed, what worked.
"""

import json
import subprocess
import sys
import urllib.request
import urllib.error


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


_CHUNK_PROMPT = """\
Compress this CHUNK of a Claude Code session into a brief turn-based narrative.
This is chunk {chunk_num} of {total_chunks} — there may be context before/after.

Rules:
- Output JSON: [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}, ...]
- 3-10 turns for this chunk. Keep it proportional to the work done
- Preserve file paths, function names, key decisions
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


def _compress_via_ollama(prompt: str, model: str = "qwen2.5:7b", timeout: int = 180) -> str | None:
    """Compress via local Ollama HTTP API. Returns response text or None on failure."""
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 8192},
    }).encode()

    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return data.get("response", "").strip()
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        print(f"[continuum:compress] ollama error: {e}", file=sys.stderr)
        return None


def _compress_via_claude(prompt: str, model: str, timeout: int = 120) -> str | None:
    """Compress via claude --print subprocess. Returns response text or None on failure."""
    try:
        result = subprocess.run(
            ["claude", "--print", "--model", model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            print(f"[continuum:compress] LLM error (rc={result.returncode}): {(result.stderr or '').strip()[:300]}", file=sys.stderr)
            return None
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        print("[continuum:compress] timeout", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[continuum:compress] error: {e}", file=sys.stderr)
        return None


def _abbreviate_code_blocks(turns: list[dict], max_code_lines: int = 8) -> list[dict]:
    """Abbreviate long code/diff blocks in turn content to reduce token count before compression.

    Keeps the first few lines as a hint, replaces the rest with a summary line.
    """
    import re
    abbreviated = []
    for turn in turns:
        content = turn.get("content", "")
        # Match fenced code blocks: ```lang\n...\n```
        def _shorten_block(m):
            fence_open = m.group(1)  # ```lang
            body = m.group(2)
            lines = body.split("\n")
            if len(lines) <= max_code_lines + 2:
                return m.group(0)  # short enough, keep as-is
            kept = "\n".join(lines[:max_code_lines])
            omitted = len(lines) - max_code_lines
            return f"{fence_open}\n{kept}\n[... {omitted} more lines ...]\n```"

        new_content = re.sub(r"(```\w*)\n([\s\S]*?)\n```", _shorten_block, content)
        abbreviated.append({**turn, "content": new_content})
    return abbreviated


def _chunk_turns(turns: list[dict], chunk_size: int = 30) -> list[list[dict]]:
    """Split turns into chunks for batched compression.

    Splits at user-turn boundaries so each chunk starts with a user message.
    """
    chunks = []
    current = []
    for turn in turns:
        current.append(turn)
        if len(current) >= chunk_size and turn["role"] == "assistant":
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)
    return chunks


def compress_session(
    turns: list[dict],
    user_prompt: str = "",
    model: str = "",
    timeout: int = 120,
    use_local: bool = False,
    local_model: str = "qwen2.5:7b",
) -> list[dict]:
    """Compress raw session turns into a clean narrative via LLM.

    Args:
        turns: list of {role, content} dicts from the raw session
        user_prompt: optional steering text appended to the compression prompt
        model: model to use for compression (claude models)
        timeout: subprocess timeout in seconds
        use_local: if True, use Ollama instead of claude --print
        local_model: Ollama model name (default: qwen2.5:7b)

    Returns:
        Compressed list of {role, content} dicts (5-20 turns)
    """
    if not model and not use_local:
        from .config import get_model
        model = get_model("compress")

    if len(turns) <= 20:
        return turns  # already short enough

    # Abbreviate code blocks before compression to reduce input tokens
    abbreviated = _abbreviate_code_blocks(turns)

    # Chunk large sessions instead of truncating
    chunks = _chunk_turns(abbreviated, chunk_size=30)

    if len(chunks) == 1:
        # Single chunk — use the full prompt
        return _compress_single(chunks[0], user_prompt, model, timeout, use_local, local_model)

    # Multi-chunk: compress each independently, then concatenate
    all_compressed = []
    for i, chunk in enumerate(chunks):
        print(f"  compressing chunk {i+1}/{len(chunks)} ({len(chunk)} turns)...", file=sys.stderr)
        chunk_prompt = _CHUNK_PROMPT.format(chunk_num=i + 1, total_chunks=len(chunks))
        if user_prompt:
            chunk_prompt += f"\n\nAdditional guidance: {user_prompt}"
        session_json = json.dumps(chunk, indent=None)
        chunk_prompt += f"\n\n<session>\n{session_json}\n</session>"

        if use_local:
            output = _compress_via_ollama(chunk_prompt, model=local_model, timeout=timeout)
        else:
            output = _compress_via_claude(chunk_prompt, model=model, timeout=timeout)

        if output:
            parsed = _parse_json_turns(output)
            if parsed:
                all_compressed.extend(parsed)
                continue

        # Fallback: keep first and last turn of this chunk as-is
        print(f"  chunk {i+1} compression failed, keeping summary", file=sys.stderr)
        all_compressed.append({
            "role": "assistant",
            "content": f"[chunk {i+1}: {len(chunk)} turns of work, compression failed]",
        })

    return all_compressed if all_compressed else turns


def _compress_single(
    turns: list[dict],
    user_prompt: str,
    model: str,
    timeout: int,
    use_local: bool,
    local_model: str,
) -> list[dict]:
    """Compress a single batch of turns."""
    truncated = _truncate_middle(turns)
    session_json = json.dumps(truncated, indent=None)
    prompt = _SYSTEM_PROMPT
    if user_prompt:
        prompt += f"\n\nAdditional guidance: {user_prompt}"
    prompt += f"\n\n<session>\n{session_json}\n</session>"

    prompt_chars = len(prompt)
    prompt_tokens_est = prompt_chars // 4
    backend = f"ollama/{local_model}" if use_local else model
    print(f"[continuum:compress] input: {prompt_chars:,} chars (~{prompt_tokens_est:,} tokens), model={backend}", file=sys.stderr)

    if use_local:
        output = _compress_via_ollama(prompt, model=local_model, timeout=timeout)
    else:
        output = _compress_via_claude(prompt, model=model, timeout=timeout)

    if output:
        print(f"[continuum:compress] output: {len(output):,} chars", file=sys.stderr)
        compressed = _parse_json_turns(output)
        if compressed:
            return compressed

        preview = output[:500] if len(output) > 500 else output
        print(f"[continuum:compress] failed to parse LLM output, preview:\n{preview}", file=sys.stderr)
    else:
        print("[continuum:compress] no output from LLM", file=sys.stderr)

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
