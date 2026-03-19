#!/usr/bin/env python3
"""Continuum Web — lightweight browser UI for context-refreshed Claude Code sessions.

Single-file server: xterm.js terminal + textarea input.
Textarea submit triggers the exit/spoof/resume cycle with fresh context.
Direct xterm typing goes straight to Claude Code.

Usage:
    python web.py                              # default session, cwd=.
    python web.py --session work-auth          # named session
    python web.py --cwd /path/to/repo          # custom working dir
    python web.py --port 9000                  # custom port
"""

import argparse
import base64
import hashlib
import json
import os
import pty
import fcntl
import struct
import subprocess
import sys
import termios
import threading
import time
import signal
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Add continuum to path
sys.path.insert(0, str(Path(__file__).parent))

from config import load_config
from session_log import SessionLog
from retrieval import ContextRetriever
from index import load_index
from session_spoof import (
    build_spoofed_session,
    write_cc_session,
    read_cc_new_entries,
    extract_text_from_cc_entry,
)

WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


# ── Global state ──────────────────────────────────────────────────────

class AppState:
    def __init__(self, config, session_name, cwd, model):
        self.config = config
        self.session_name = session_name
        self.cwd = cwd
        self.model = model
        self.lock = threading.Lock()

        continuum_dir = Path(__file__).parent

        # Identity
        identity_path = Path(config.get("identity", "identity.md"))
        if not identity_path.is_absolute():
            identity_path = continuum_dir / identity_path
        self.identity_text = identity_path.read_text().strip() if identity_path.exists() else ""

        # System prompt
        sp = config.get("system_prompt", "")
        sp_path = continuum_dir / sp
        system_prompt = sp_path.read_text().strip() if sp_path.is_file() else sp
        if system_prompt:
            self.identity_text = f"{self.identity_text}\n\n{system_prompt}" if self.identity_text else system_prompt

        # Session log
        log_dir = Path(config["session"]["log_dir"]).expanduser()
        self.log = SessionLog(log_dir / f"{session_name}.jsonl")

        # Retriever
        idx = load_index()
        sources = config.get("context_sources", [])
        self.retriever = ContextRetriever(sources, index=idx)
        self.corpus_size = len(idx)

        # PTY state
        self.master_fd = -1
        self.proc = None
        self.cc_session_id = ""
        self.entry_count = 0
        self.session_file = None
        self.ws_clients = []  # list of socket objects
        self.buffer = []
        self.buffer_chars = 0

    def spoof_and_launch(self, user_message=""):
        """Kill current session, spoof fresh context, launch new claude --resume."""
        with self.lock:
            self._kill_current()
            self._capture_responses()

        # Append user message to ground truth
        if user_message:
            self.log.append("user", user_message, thread="default")

        # Retrieve context
        token_budgets = self.config.get("token_budgets", {})
        retrieved = ""
        try:
            query = user_message or (self.log.entries[-1]["content"] if self.log.entries else "continue")
            recent = self.log.get_recent(10)
            tail = "\n".join(f"[{e.get('role','?')}] {e.get('content','')[:300]}" for e in recent)
            retrieved = self.retriever.retrieve(
                query,
                token_budget=token_budgets.get("dynamic_context", 30000),
                conversation_tail=tail,
            )
        except Exception:
            pass

        # Build spoofed session
        self.cc_session_id = str(uuid.uuid4())
        cc_entries = build_spoofed_session(
            session_id=self.cc_session_id,
            continuum_log=self.log,
            retrieved_context=retrieved,
            compressed_blocks=[],
            tail_budget=token_budgets.get("recent_tail", 80000),
            identity_text=self.identity_text,
            cwd=self.cwd,
        )
        self.session_file = write_cc_session(self.cc_session_id, cc_entries, cwd=self.cwd)
        self.entry_count = len(cc_entries)

        # Launch
        with self.lock:
            self._spawn_pty()

        # Feed the user message into the PTY after Claude Code starts
        if user_message:
            def _feed_input():
                time.sleep(2)  # wait for Claude Code to initialize
                self.write_pty(user_message)
                delay = max(0.1, min(0.5, len(user_message) * 0.002))
                time.sleep(delay)
                self.write_pty("\r")
            threading.Thread(target=_feed_input, daemon=True).start()

        return len(retrieved), len(cc_entries)

    def _kill_current(self):
        if self.proc and self.proc.poll() is None:
            try:
                os.kill(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=3)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        if self.master_fd >= 0:
            try:
                os.close(self.master_fd)
            except Exception:
                pass
            self.master_fd = -1
        self.proc = None

    def _capture_responses(self):
        if not self.session_file or not self.cc_session_id:
            return
        try:
            new_entries = read_cc_new_entries(self.session_file, self.entry_count)
            captured = 0
            for entry in new_entries:
                if entry.get("type") == "assistant":
                    text = extract_text_from_cc_entry(entry)
                    if text.strip():
                        self.log.append("assistant", text, thread="default")
                        captured += 1
            if captured:
                print(f"[continuum] captured {captured} entries", flush=True)
        except Exception as e:
            print(f"[continuum] capture failed: {e}", file=sys.stderr, flush=True)

    def _spawn_pty(self):
        master, slave = pty.openpty()
        try:
            winsz = struct.pack("HHHH", 24, 80, 0, 0)
            fcntl.ioctl(master, termios.TIOCSWINSZ, winsz)
        except Exception:
            pass
        env = {k: v for k, v in os.environ.items() if k not in ("CLAUDECODE",)}
        env["TERM"] = env.get("TERM", "xterm-256color")

        cmd = ["claude", "--resume", self.cc_session_id]

        self.proc = subprocess.Popen(
            cmd, stdin=slave, stdout=slave, stderr=slave,
            cwd=self.cwd, env=env, close_fds=True,
        )
        os.close(slave)
        self.master_fd = master
        os.set_blocking(master, False)

        threading.Thread(target=self._reader_loop, daemon=True).start()

    def _reader_loop(self):
        fd = self.master_fd
        while True:
            try:
                data = os.read(fd, 4096)
                if not data:
                    break
            except OSError:
                break
            text = data.decode("utf-8", errors="replace")
            # Buffer for new WS clients
            self.buffer.append(text)
            self.buffer_chars += len(text)
            while self.buffer_chars > 100000:
                removed = self.buffer.pop(0)
                self.buffer_chars -= len(removed)
            # Broadcast to WS clients
            for ws in list(self.ws_clients):
                try:
                    _ws_send_text(ws, text)
                except Exception:
                    try:
                        self.ws_clients.remove(ws)
                    except ValueError:
                        pass

    def write_pty(self, data):
        fd = self.master_fd
        if fd >= 0:
            try:
                os.write(fd, data.encode("utf-8") if isinstance(data, str) else data)
            except Exception:
                pass

    def resize_pty(self, cols, rows):
        fd = self.master_fd
        if fd >= 0:
            try:
                winsz = struct.pack("HHHH", rows, cols, 0, 0)
                fcntl.ioctl(fd, termios.TIOCSWINSZ, winsz)
                if self.proc and self.proc.poll() is None:
                    os.kill(self.proc.pid, signal.SIGWINCH)
            except Exception:
                pass


# ── WebSocket helpers ─────────────────────────────────────────────────

def _ws_accept_key(key):
    return base64.b64encode(hashlib.sha1((key + WS_GUID).encode()).digest()).decode()

def _ws_send_text(sock, text):
    payload = text.encode("utf-8")
    header = bytearray()
    header.append(0x81)  # text frame
    length = len(payload)
    if length < 126:
        header.append(length)
    elif length < 65536:
        header.append(126)
        header.extend(length.to_bytes(2, "big"))
    else:
        header.append(127)
        header.extend(length.to_bytes(8, "big"))
    sock.sendall(bytes(header) + payload)

def _ws_read_frame(sock):
    head = sock.recv(2)
    if len(head) < 2:
        return None
    opcode = head[0] & 0x0F
    masked = head[1] & 0x80
    length = head[1] & 0x7F
    if length == 126:
        length = int.from_bytes(sock.recv(2), "big")
    elif length == 127:
        length = int.from_bytes(sock.recv(8), "big")
    mask = sock.recv(4) if masked else b""
    data = b""
    while len(data) < length:
        chunk = sock.recv(length - len(data))
        if not chunk:
            return None
        data += chunk
    if masked:
        data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
    if opcode == 0x08:
        return None  # close
    if opcode == 0x09:  # ping
        return {"type": "ping", "data": data}
    return data.decode("utf-8", errors="replace")


# ── HTTP handler ──────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # quiet

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/" or parsed.path == "":
            blob = PAGE_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)
            return

        if parsed.path == "/ws":
            self._handle_ws()
            return

        self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if parsed.path == "/turn":
            text = str(body.get("text", "")).strip()
            if not text:
                self._json({"ok": False, "error": "empty"}, 400)
                return
            app = self.server.app
            ctx_chars, entries = app.spoof_and_launch(text)
            self._json({"ok": True, "context_chars": ctx_chars, "entries": entries})
            return

        if parsed.path == "/status":
            app = self.server.app
            alive = app.proc is not None and app.proc.poll() is None
            self._json({
                "ok": True,
                "alive": alive,
                "session": app.session_name,
                "log_entries": len(app.log),
                "corpus": app.corpus_size,
            })
            return

        self.send_error(404)

    def _json(self, data, code=200):
        blob = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def _handle_ws(self):
        key = self.headers.get("Sec-WebSocket-Key", "")
        if not key:
            self.send_error(400)
            return
        accept = _ws_accept_key(key)
        self.send_response(101, "Switching Protocols")
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept)
        self.end_headers()

        sock = self.request
        app = self.server.app

        # Send buffered output
        for chunk in list(app.buffer):
            try:
                _ws_send_text(sock, chunk)
            except Exception:
                return

        app.ws_clients.append(sock)
        try:
            while True:
                frame = _ws_read_frame(sock)
                if frame is None:
                    break
                if isinstance(frame, dict) and frame.get("type") == "ping":
                    sock.sendall(b"\x8a\x00")  # pong
                    continue
                if isinstance(frame, str):
                    try:
                        msg = json.loads(frame)
                    except json.JSONDecodeError:
                        app.write_pty(frame)
                        continue
                    if msg.get("type") == "input":
                        app.write_pty(msg.get("data", ""))
                    elif msg.get("type") == "resize":
                        app.resize_pty(
                            int(msg.get("cols", 80)),
                            int(msg.get("rows", 24)),
                        )
        except Exception:
            pass
        finally:
            try:
                app.ws_clients.remove(sock)
            except ValueError:
                pass


# ── HTML page ─────────────────────────────────────────────────────────

PAGE_HTML = """<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
<title>Continuum</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/xterm/css/xterm.css">
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
html, body { height: 100%; background: #0a0e14; color: #d9e0e8; font-family: system-ui, sans-serif; }
body { display: flex; flex-direction: column; }
#terminal { flex: 1; min-height: 0; }
.bar {
  display: flex; gap: 8px; padding: 8px 12px;
  border-top: 1px solid #1a2030; background: #0d1117;
}
#input {
  flex: 1; background: #161b22; color: #d9e0e8;
  border: 1px solid #30363d; border-radius: 8px; padding: 8px 12px;
  font-size: 14px; font-family: system-ui, sans-serif; outline: none;
  resize: none; overflow-y: hidden;
}
#sendBtn {
  background: #238636; color: #fff; border: none; border-radius: 8px;
  padding: 8px 20px; font-weight: 600; cursor: pointer; font-size: 14px;
}
#sendBtn:disabled { opacity: 0.4; cursor: not-allowed; }
#status { color: #484f58; font-size: 11px; align-self: center; white-space: nowrap; }
</style>
</head><body>
<div id="terminal"></div>
<div class="bar">
  <textarea id="input" rows="1" placeholder="Context-refreshed turn (Enter to send, Shift+Enter for newline)"></textarea>
  <button id="sendBtn">Send</button>
  <span id="status">ready</span>
</div>
<script src="https://cdn.jsdelivr.net/npm/xterm/lib/xterm.js"></script>
<script src="https://cdn.jsdelivr.net/npm/xterm-addon-fit/lib/xterm-addon-fit.js"></script>
<script>
var term = new Terminal({
  convertEol: true,
  theme: { background: '#0a0e14', foreground: '#d9e0e8' },
  fontFamily: '"Cascadia Code", "Fira Code", "Consolas", monospace',
  fontSize: 14, cursorBlink: false,
});
var fit = new FitAddon.FitAddon();
term.loadAddon(fit);
term.open(document.getElementById('terminal'));
fit.fit();

var ws = null;
var input = document.getElementById('input');
var sendBtn = document.getElementById('sendBtn');
var status = document.getElementById('status');
var sending = false;

function connect() {
  var proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(proto + '//' + location.host + '/ws');
  ws.onmessage = function(e) { term.write(e.data); };
  ws.onopen = function() { status.textContent = 'connected';
    ws.send(JSON.stringify({ type: 'resize', cols: term.cols, rows: term.rows }));
  };
  ws.onclose = function() { status.textContent = 'disconnected'; };
}

term.onData(function(data) {
  if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: 'input', data: data }));
});

window.addEventListener('resize', function() {
  fit.fit();
  if (ws && ws.readyState === 1)
    ws.send(JSON.stringify({ type: 'resize', cols: term.cols, rows: term.rows }));
});

function send() {
  var text = input.value.trim();
  if (!text || sending) return;
  sending = true;
  sendBtn.disabled = true;
  status.textContent = 'refreshing context...';
  input.value = '';
  input.style.height = '';

  fetch('/turn', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: text }),
  }).then(function(r) { return r.json(); }).then(function(d) {
    sending = false;
    sendBtn.disabled = false;
    if (d.ok) {
      status.textContent = d.context_chars + ' chars context, ' + d.entries + ' entries';
      term.write('\\r\\n\\x1b[90m--- context refreshed ---\\x1b[0m\\r\\n');
      setTimeout(connect, 200);
    } else {
      status.textContent = 'error: ' + (d.error || '?');
    }
  }).catch(function(e) {
    sending = false;
    sendBtn.disabled = false;
    status.textContent = 'network error';
  });
}

sendBtn.onclick = send;
input.onkeydown = function(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
};
input.oninput = function() {
  input.style.height = '';
  input.style.height = input.scrollHeight + 'px';
};

connect();
</script>
</body></html>"""


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Continuum Web — browser UI for context-refreshed Claude Code")
    parser.add_argument("--config", "-c", help="Path to continuum.yaml")
    parser.add_argument("--session", "-s", default="default", help="Session name (default: 'default')")
    parser.add_argument("--model", "-m", help="Override model")
    parser.add_argument("--cwd", help="Working directory for Claude Code (default: cwd)")
    parser.add_argument("--port", "-p", type=int, default=8080, help="Port (default: 8080)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.model:
        config["model"] = args.model

    cwd = args.cwd or str(Path.cwd())
    model = config.get("model", "claude-sonnet-4-6")

    app = AppState(config, args.session, cwd, model)

    # Auto-launch a session on startup
    print(f"continuum web | session={args.session} | model={model}")
    print(f"  corpus: {app.corpus_size} entries | log: {len(app.log)} entries")
    print(f"  cwd: {cwd}")

    ctx_chars, entries = app.spoof_and_launch()
    print(f"  spoofed: {entries} entries, {ctx_chars} chars context")

    server = HTTPServer(("0.0.0.0", args.port), Handler)
    server.app = app
    print(f"\n  http://localhost:{args.port}\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass

    # Capture on shutdown
    with app.lock:
        app._capture_responses()
        app._kill_current()
    print(f"\nsession saved: {app.log.path} ({len(app.log)} entries)")


if __name__ == "__main__":
    main()
