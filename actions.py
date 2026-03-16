"""Action execution — Symbolic takes action via Claude Code.

When Symbolic's response contains <action> blocks, the system
parses them and executes via `claude --print` in the target
project's directory. Results are captured and returned.

Action block format in Symbolic's response:
    <action project="bridge" type="command">
    Fix the PTY resize bug in webui_server.py
    </action>

Supported types:
    - command: run `claude --print` with the action text in the project dir
    - dispatch: create a dispatch card via `px dispatch` (home) or `cmux` (work)
"""

import os
import re
import shutil
import subprocess
from pathlib import Path


# Regex to extract action blocks from response text
ACTION_RE = re.compile(
    r'<action\s+project="([^"]+)"\s+type="([^"]+)">\s*\n?(.*?)\n?\s*</action>',
    re.DOTALL,
)


def parse_actions(response: str) -> list[dict]:
    """Extract action blocks from Symbolic's response.

    Returns list of {"project": str, "type": str, "content": str}.
    """
    actions = []
    for match in ACTION_RE.finditer(response):
        actions.append({
            "project": match.group(1).strip(),
            "type": match.group(2).strip(),
            "content": match.group(3).strip(),
        })
    return actions


def strip_actions(response: str) -> str:
    """Remove action blocks from response text (for display)."""
    return ACTION_RE.sub("", response).strip()


def resolve_project_dir(project: str) -> Path | None:
    """Resolve a project name to its directory via px goto."""
    px_bin = shutil.which("px") or str(Path.home() / "projects" / "px")
    try:
        result = subprocess.run(
            [px_bin, "goto", project],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return Path(result.stdout.strip().splitlines()[0])
    except Exception:
        pass

    # Fallback: try ~/projects/<project>
    fallback = Path.home() / "projects" / project
    if fallback.is_dir():
        return fallback
    return None


def execute_action(action: dict, model: str = "claude-sonnet-4-6") -> dict:
    """Execute a single action and return the result.

    Returns {"ok": bool, "result": str, "project": str, "type": str}.
    """
    project = action["project"]
    action_type = action["type"]
    content = action["content"]

    project_dir = resolve_project_dir(project)
    if not project_dir:
        return {
            "ok": False,
            "result": f"Project '{project}' not found",
            "project": project,
            "type": action_type,
        }

    if action_type == "read":
        return _execute_read(content, project_dir)
    elif action_type == "dispatch":
        return _execute_dispatch(project, content)
    elif action_type == "cmux":
        return _execute_cmux(project, content)
    elif action_type == "command":
        return _execute_command(content, project_dir, model)
    else:
        return {
            "ok": False,
            "result": f"Unknown action type: {action_type}",
            "project": project,
            "type": action_type,
        }


def _execute_read(file_path: str, cwd: Path) -> dict:
    """Read a file directly — no LLM, instant."""
    path = Path(file_path.strip()).expanduser()
    if not path.is_absolute():
        path = cwd / path
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        return {
            "ok": True,
            "result": content[:4000],
            "project": cwd.name,
            "type": "read",
            "path": str(path),
        }
    except (OSError, PermissionError) as e:
        return {"ok": False, "result": str(e), "project": cwd.name, "type": "read"}


def _execute_command(content: str, cwd: Path, model: str) -> dict:
    """Execute via claude --print in the project directory."""
    claude_bin = shutil.which("claude") or "claude"
    env = {k: v for k, v in os.environ.items()
           if k not in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")}

    try:
        result = subprocess.run(
            [claude_bin, "--print", "--model", model,
             "--no-session-persistence", "-p", content],
            capture_output=True, text=True, timeout=120,
            cwd=str(cwd), env=env, stdin=subprocess.DEVNULL,
        )
        output = (result.stdout or "").strip()
        if result.returncode != 0:
            error = (result.stderr or "").strip()[:300]
            return {
                "ok": False,
                "result": f"Exit {result.returncode}: {error}",
                "project": cwd.name,
                "type": "command",
            }
        return {
            "ok": True,
            "result": output[:2000],  # cap result size
            "project": cwd.name,
            "type": "command",
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "result": "timeout (120s)", "project": cwd.name, "type": "command"}
    except Exception as e:
        return {"ok": False, "result": str(e)[:300], "project": cwd.name, "type": "command"}


def _execute_dispatch(project: str, content: str) -> dict:
    """Create a dispatch card via px dispatch."""
    px_bin = shutil.which("px") or str(Path.home() / "projects" / "px")
    try:
        result = subprocess.run(
            [px_bin, "dispatch", project, content],
            capture_output=True, text=True, timeout=10,
        )
        output = (result.stdout or "").strip()
        if result.returncode != 0:
            return {"ok": False, "result": (result.stderr or "").strip()[:300], "project": project, "type": "dispatch"}
        return {"ok": True, "result": output, "project": project, "type": "dispatch"}
    except Exception as e:
        return {"ok": False, "result": str(e)[:300], "project": project, "type": "dispatch"}


def _execute_cmux(project: str, content: str) -> dict:
    """Execute via cmux (work laptop dispatch alternative)."""
    cmux_bin = shutil.which("cmux")
    if not cmux_bin:
        # Fallback to dispatch if cmux not available
        return _execute_dispatch(project, content)
    try:
        result = subprocess.run(
            [cmux_bin, project, content],
            capture_output=True, text=True, timeout=10,
        )
        output = (result.stdout or "").strip()
        if result.returncode != 0:
            return {"ok": False, "result": (result.stderr or "").strip()[:300], "project": project, "type": "cmux"}
        return {"ok": True, "result": output, "project": project, "type": "cmux"}
    except Exception as e:
        return {"ok": False, "result": str(e)[:300], "project": project, "type": "cmux"}
