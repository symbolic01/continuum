"""Session spoofing — rewrite Claude Code session JSONL with dynamic context.

Reads Continuum's ground truth log + retrieved corpus context, and writes
a Claude Code compatible session JSONL that can be resumed with `claude --resume`.

The Claude Code JSONL is a disposable view — Continuum's log is the permanent record.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from tokens import count_tokens
from session_log import SessionLog, mint_uid
from compression import CompressionPolicy, TokenBudgetPolicy, CompressedBlock


# Claude Code session directory
CC_SESSION_DIR = Path.home() / ".claude" / "projects"


def detect_cc_project_dir() -> Path:
    """Detect the Claude Code project directory from CWD.

    Claude Code uses the CWD path mangled into a directory name:
    /home/symbolic/projects → -home-symbolic-projects
    """
    cwd = Path.cwd()
    mangled = str(cwd).replace("/", "-")
    if mangled.startswith("-"):
        pass  # already starts with -
    else:
        mangled = "-" + mangled
    return CC_SESSION_DIR / mangled


def make_cc_entry(
    entry_type: str,
    role: str,
    content,
    session_id: str,
    parent_uuid: str | None = None,
    cwd: str = "/home/symbolic/projects",
    entry_uuid: str | None = None,
    timestamp: str | None = None,
) -> dict:
    """Create a Claude Code compatible JSONL entry."""
    entry_uuid = entry_uuid or str(uuid.uuid4())
    timestamp = timestamp or datetime.now(timezone.utc).isoformat()

    entry = {
        "type": entry_type,
        "uuid": entry_uuid,
        "parentUuid": parent_uuid,
        "sessionId": session_id,
        "version": "2.1.38",
        "timestamp": timestamp,
        "isSidechain": False,
        "userType": "external",
        "cwd": cwd,
        "message": {"role": role, "content": content},
    }
    return entry


def build_spoofed_session(
    session_id: str,
    continuum_log: SessionLog,
    retrieved_context: str = "",
    compressed_blocks: list[CompressedBlock] | None = None,
    tail_budget: int = 80_000,
    identity_text: str = "",
    cwd: str = "/home/symbolic/projects",
) -> list[dict]:
    """Build a Claude Code session JSONL from Continuum state.

    Returns a list of CC-format entries ready to write.
    """
    entries = []
    prev_uuid = None

    # 1. Inject identity + retrieved context as a synthetic opening exchange
    if identity_text or retrieved_context:
        context_parts = []
        if identity_text:
            context_parts.append(f"<identity>\n{identity_text}\n</identity>")
        if retrieved_context:
            context_parts.append(f"<retrieved_context>\n{retrieved_context}\n</retrieved_context>")

        context_text = "\n\n".join(context_parts)

        # Synthetic user turn injecting context
        u = make_cc_entry("user", "user", context_text, session_id,
                          parent_uuid=prev_uuid, cwd=cwd)
        entries.append(u)
        prev_uuid = u["uuid"]

        # Synthetic assistant acknowledgment
        a = make_cc_entry("assistant", "assistant",
                          [{"type": "text", "text": "I have this context. Ready to continue."}],
                          session_id, parent_uuid=prev_uuid, cwd=cwd)
        entries.append(a)
        prev_uuid = a["uuid"]

    # 2. Inject compressed history blocks as synthetic exchanges
    if compressed_blocks:
        for block in compressed_blocks:
            summary = block.to_message()["content"]
            u = make_cc_entry("user", "user",
                              f"<compressed_history>\n{summary}\n</compressed_history>",
                              session_id, parent_uuid=prev_uuid, cwd=cwd)
            entries.append(u)
            prev_uuid = u["uuid"]

            a = make_cc_entry("assistant", "assistant",
                              [{"type": "text", "text": "Noted."}],
                              session_id, parent_uuid=prev_uuid, cwd=cwd)
            entries.append(a)
            prev_uuid = a["uuid"]

    # 3. Recent session tail at full fidelity
    log_entries = continuum_log.entries
    tail_tokens = 0
    tail_start = len(log_entries)

    # Walk backwards to find where the tail starts
    for i in range(len(log_entries) - 1, -1, -1):
        entry_tokens = count_tokens(log_entries[i].get("content", "")) + 50
        if tail_tokens + entry_tokens > tail_budget:
            break
        tail_tokens += entry_tokens
        tail_start = i

    tail = log_entries[tail_start:]

    for entry in tail:
        role = entry.get("role", "user")
        content = entry.get("content", "")
        ts = entry.get("ts", "")

        if role == "user":
            cc = make_cc_entry("user", "user", content, session_id,
                               parent_uuid=prev_uuid, cwd=cwd, timestamp=ts)
        else:
            cc = make_cc_entry("assistant", "assistant",
                               [{"type": "text", "text": content}],
                               session_id, parent_uuid=prev_uuid, cwd=cwd,
                               timestamp=ts)
        entries.append(cc)
        prev_uuid = cc["uuid"]

    return entries


def write_cc_session(session_id: str, entries: list[dict], cwd: str = "/home/symbolic/projects") -> Path:
    """Write entries to a Claude Code session JSONL file.

    Returns the path to the written file.
    """
    mangled_cwd = cwd.replace("/", "-")
    if not mangled_cwd.startswith("-"):
        mangled_cwd = "-" + mangled_cwd
    session_dir = CC_SESSION_DIR / mangled_cwd
    session_dir.mkdir(parents=True, exist_ok=True)

    session_file = session_dir / f"{session_id}.jsonl"
    with open(session_file, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    return session_file


def read_cc_new_entries(session_file: Path, known_count: int) -> list[dict]:
    """Read new entries from a Claude Code session JSONL after known_count.

    Used to capture what Claude Code added during a turn (tool use, responses, etc.)
    """
    entries = []
    with open(session_file) as f:
        for i, line in enumerate(f):
            if i >= known_count:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    return entries


def extract_text_from_cc_entry(entry: dict) -> str:
    """Extract readable text from a Claude Code entry."""
    msg = entry.get("message", {})
    content = msg.get("content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            if btype == "text":
                parts.append(block.get("text", ""))
            elif btype == "tool_use":
                name = block.get("name", "?")
                inp = block.get("input", {})
                parts.append(f"[{name}: {json.dumps(inp)[:100]}]")
            elif btype == "tool_result":
                result = block.get("content", "")
                if isinstance(result, str):
                    parts.append(f"[result: {result[:200]}]")
        return "\n".join(parts)

    return str(content)[:500]
