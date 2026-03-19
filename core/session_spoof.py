"""Session spoofing — rewrite Claude Code session JSONL with dynamic context.

Reads Continuum's ground truth log + retrieved corpus context, and writes
a Claude Code compatible session JSONL that can be resumed with `claude --resume`.

The Claude Code JSONL is a disposable view — Continuum's log is the permanent record.
"""

import copy
import glob
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .tokens import count_tokens
from .session_log import SessionLog, mint_uid
from .compression import CompressionPolicy, TokenBudgetPolicy, CompressedBlock


# Claude Code session directory
CC_SESSION_DIR = Path.home() / ".claude" / "projects"

# Template entries loaded from a real CC session (lazy-initialized)
_TEMPLATES: dict[str, dict] = {}


def _mangle_cwd(cwd: str) -> str:
    mangled = cwd.replace("/", "-")
    if not mangled.startswith("-"):
        mangled = "-" + mangled
    return mangled


def detect_cc_project_dir() -> Path:
    """Detect the Claude Code project directory from CWD.

    Claude Code uses the CWD path mangled into a directory name:
    /home/symbolic/projects → -home-symbolic-projects
    """
    cwd = Path.cwd()
    return CC_SESSION_DIR / _mangle_cwd(str(cwd))


def _load_templates(cwd: str = "/home/symbolic/projects") -> None:
    """Load real user + assistant entries from the most recent CC session as templates.

    CC validates entry structure strictly — synthetic entries built from scratch
    are silently dropped on resume. Using deepcopy from real entries preserves
    the exact field set, key order, and nested structure CC expects.
    """
    if _TEMPLATES:
        return

    cc_dir = CC_SESSION_DIR / _mangle_cwd(cwd)
    if not cc_dir.is_dir():
        return

    candidates = sorted(glob.glob(str(cc_dir / "*.jsonl")), key=os.path.getmtime, reverse=True)
    for session_file in candidates:
        real_user = None
        real_asst = None
        try:
            with open(session_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    e = json.loads(line)
                    if e.get("type") == "user" and real_user is None:
                        content = e.get("message", {}).get("content", "")
                        if isinstance(content, str) and len(content) > 5:
                            real_user = e
                    if e.get("type") == "assistant" and real_asst is None:
                        content = e.get("message", {}).get("content", [])
                        if isinstance(content, list) and any(
                            b.get("type") == "text" for b in content
                        ):
                            real_asst = e
                    if real_user and real_asst:
                        break
        except Exception:
            continue
        if real_user and real_asst:
            _TEMPLATES["user"] = real_user
            _TEMPLATES["assistant"] = real_asst
            return


def _normalize_timestamp(ts: str | None = None) -> str:
    """Convert a timestamp to CC's expected format: YYYY-MM-DDTHH:MM:SS.mmmZ"""
    if not ts:
        now = datetime.now(timezone.utc)
        return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"
    # Already in CC format
    if ts.endswith("Z") and "+" not in ts:
        return ts
    # Python isoformat: +00:00 suffix, possibly microseconds
    ts = ts.replace("+00:00", "Z")
    # Truncate microseconds to milliseconds: .123456Z → .123Z
    import re
    ts = re.sub(r"\.(\d{3})\d+Z$", r".\1Z", ts)
    return ts


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
    """Create a Claude Code compatible JSONL entry.

    Uses deepcopy from real CC entries as templates to ensure the exact
    field structure CC expects. Falls back to synthetic entries if no
    real session is available.
    """
    _load_templates(cwd)
    entry_uuid = entry_uuid or str(uuid.uuid4())
    timestamp = _normalize_timestamp(timestamp)

    template = _TEMPLATES.get(role)

    if template:
        entry = copy.deepcopy(template)
        entry["uuid"] = entry_uuid
        entry["parentUuid"] = parent_uuid
        entry["sessionId"] = session_id
        entry["timestamp"] = timestamp
        entry.pop("planContent", None)

        if role == "user":
            entry["promptId"] = str(uuid.uuid4())
            entry["message"] = {"role": "user", "content": content}
        elif role == "assistant":
            msg = entry["message"]
            msg["id"] = "msg_01" + uuid.uuid4().hex[:22]
            if isinstance(content, list):
                msg["content"] = content
            else:
                msg["content"] = [{"type": "text", "text": str(content)}]
        return entry

    # Fallback: build from scratch (may not render on resume)
    if role == "assistant":
        message = {
            "role": "assistant",
            "model": "claude-opus-4-6",
            "id": f"msg_01{uuid.uuid4().hex[:22]}",
            "type": "message",
            "content": content if isinstance(content, list) else [{"type": "text", "text": str(content)}],
            "stop_reason": None,
            "stop_sequence": None,
        }
    else:
        message = {"role": role, "content": content}

    return {
        "type": entry_type,
        "uuid": entry_uuid,
        "parentUuid": parent_uuid,
        "sessionId": session_id,
        "version": "2.1.78",
        "timestamp": timestamp,
        "isSidechain": False,
        "userType": "external",
        "entrypoint": "cli",
        "cwd": cwd,
        "message": message,
    }


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
    # Generate monotonically increasing timestamps (CC needs unique, ordered ts)
    _base_time = datetime.now(timezone.utc)
    _entry_idx = [0]

    def _next_ts() -> str:
        ts = _base_time.replace(microsecond=0) + __import__("datetime").timedelta(seconds=_entry_idx[0])
        _entry_idx[0] += 1
        return ts.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ts.microsecond // 1000:03d}Z"

    # 1. Inject identity as first-person assistant statement
    if identity_text:
        # Seed with a user prompt so the assistant identity is a response
        u = make_cc_entry("user", "user", "Who are you?", session_id,
                          parent_uuid=prev_uuid, cwd=cwd, timestamp=_next_ts())
        entries.append(u)
        prev_uuid = u["uuid"]

        a = make_cc_entry("assistant", "assistant",
                          [{"type": "text", "text": identity_text}],
                          session_id, parent_uuid=prev_uuid, cwd=cwd, timestamp=_next_ts())
        entries.append(a)
        prev_uuid = a["uuid"]

    # 2. Inject retrieved context as assistant recall blocks
    if retrieved_context:
        # Split into chunks so each gets its own ● bullet
        chunks = []
        current_chunk = []
        current_len = 0
        for line in retrieved_context.split("\n"):
            current_chunk.append(line)
            current_len += len(line)
            # Split at ~2000 chars or at entry boundaries
            if current_len >= 2000 and line.strip() == "":
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_len = 0
        if current_chunk:
            chunks.append("\n".join(current_chunk))

        u = make_cc_entry("user", "user", "What do you recall about the current work?", session_id,
                          parent_uuid=prev_uuid, cwd=cwd, timestamp=_next_ts())
        entries.append(u)
        prev_uuid = u["uuid"]

        for chunk in chunks:
            if not chunk.strip():
                continue
            a = make_cc_entry("assistant", "assistant",
                              [{"type": "text", "text": chunk.strip()}],
                              session_id, parent_uuid=prev_uuid, cwd=cwd, timestamp=_next_ts())
            entries.append(a)
            prev_uuid = a["uuid"]

    # 2. Inject compressed history blocks as synthetic exchanges
    if compressed_blocks:
        for block in compressed_blocks:
            summary = block.to_message()["content"]
            u = make_cc_entry("user", "user",
                              f"<compressed_history>\n{summary}\n</compressed_history>",
                              session_id, parent_uuid=prev_uuid, cwd=cwd, timestamp=_next_ts())
            entries.append(u)
            prev_uuid = u["uuid"]

            a = make_cc_entry("assistant", "assistant",
                              [{"type": "text", "text": "Noted."}],
                              session_id, parent_uuid=prev_uuid, cwd=cwd, timestamp=_next_ts())
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
                               parent_uuid=prev_uuid, cwd=cwd, timestamp=_next_ts())
        else:
            cc = make_cc_entry("assistant", "assistant",
                               [{"type": "text", "text": content}],
                               session_id, parent_uuid=prev_uuid, cwd=cwd,
                               timestamp=_next_ts())
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
