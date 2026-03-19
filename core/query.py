"""Query decomposition — LLM-routed retrieval strategy selection.

Given a user query, decomposes it into structured retrieval axes
with weights. Uses Ollama for fast local decomposition.
"""

import json
import subprocess


DECOMPOSE_SYSTEM = """You decompose user queries into retrieval instructions for a memory system.

Given a query, identify which retrieval axes to use and their relative weights.

Axes:
- semantic: find content similar in meaning (always include, weight >= 0.2)
- temporal: find content by time (use when query mentions "first", "last", "recent", "yesterday", dates, etc.)
  - filter: "oldest", "newest", "before:DATE", "after:DATE", "range:DATE1-DATE2"
- project: find content from a specific project/thread
  - filter: project name
- entity: find mentions of a specific person, tool, file, or concept
  - filter: the entity name
- causal: find decision chains, reasoning, "why" something happened
- anti_pattern: find failures, errors, and their corrections
- emotional: find content with similar emotional tone
- polarity: filter by success or failure
  - filter: "success" or "failure"

Also extract:
- keywords: 5-10 expanded search terms — synonyms, related jargon, abbreviations, alternate phrasings. These augment keyword search beyond the literal query words.
- identifiers: any file paths, function names, variable names, class names, or code identifiers mentioned or implied in the query. Extract them exactly as the user wrote them (even if approximate/misspelled — fuzzy matching happens downstream).

Output ONLY valid JSON, no markdown:
{"axes": [...], "rewritten_query": "optimized search text", "keywords": ["term1", "term2", ...], "identifiers": ["file.py", "func_name", ...]}

The rewritten_query should be the semantic core of the query, stripped of meta-language.

Examples:
- "what's the first thing we ever talked about?" → {"axes": [{"axis": "temporal", "weight": 0.7, "filter": "oldest"}, {"axis": "semantic", "weight": 0.3}], "rewritten_query": "initial conversation first discussion", "keywords": ["first", "beginning", "earliest", "introduction", "started"], "identifiers": []}
- "that PTY resize bug in bridge" → {"axes": [{"axis": "semantic", "weight": 0.5}, {"axis": "project", "weight": 0.3, "filter": "bridge"}, {"axis": "entity", "weight": 0.2, "filter": "PTY resize"}], "rewritten_query": "PTY resize TIOCSWINSZ SIGWINCH bug fix", "keywords": ["PTY", "TIOCSWINSZ", "SIGWINCH", "terminal", "resize", "winsz", "ioctl"], "identifiers": ["webui_server.py", "_start_proc"]}
- "what went wrong last time we tried compression?" → {"axes": [{"axis": "anti_pattern", "weight": 0.4}, {"axis": "semantic", "weight": 0.3}, {"axis": "temporal", "weight": 0.3, "filter": "newest"}], "rewritten_query": "compression policy failure error", "keywords": ["compress", "compression", "summary", "truncate", "token_budget", "failure", "error"], "identifiers": ["compression.py", "session_compress.py"]}
- "check the webserverui file" → {"axes": [{"axis": "semantic", "weight": 0.5}, {"axis": "entity", "weight": 0.5, "filter": "webserverui"}], "rewritten_query": "web server UI file", "keywords": ["webui", "server", "HTTP", "handler", "endpoint"], "identifiers": ["webserverui"]}"""


def decompose_query(query: str, model: str = "qwen2.5:7b") -> dict:
    """Decompose a query into retrieval axes via local LLM.

    Returns:
        {"axes": [{"axis": str, "weight": float, "filter": str|None}],
         "rewritten_query": str}
    """
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": DECOMPOSE_SYSTEM},
            {"role": "user", "content": query},
        ],
        "stream": False,
        "options": {"temperature": 0.1},
        "format": "json",
    })

    try:
        result = subprocess.run(
            ["curl", "-s", "http://localhost:11434/api/chat", "-d", payload],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            return _fallback(query)

        data = json.loads(result.stdout)
        content = data.get("message", {}).get("content", "")
        parsed = json.loads(content)

        # Validate structure
        if "axes" not in parsed or not isinstance(parsed["axes"], list):
            return _fallback(query)

        return {
            "axes": parsed["axes"],
            "rewritten_query": parsed.get("rewritten_query", query),
            "keywords": parsed.get("keywords", []),
            "identifiers": parsed.get("identifiers", []),
        }
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        return _fallback(query)


def _fallback(query: str) -> dict:
    """Fallback: pure semantic retrieval."""
    return {
        "axes": [{"axis": "semantic", "weight": 1.0}],
        "rewritten_query": query,
        "keywords": [],
        "identifiers": [],
    }
