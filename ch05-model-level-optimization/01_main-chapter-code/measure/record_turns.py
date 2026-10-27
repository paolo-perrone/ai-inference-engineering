#!/usr/bin/env python3
"""Record the agent's own turns: a pass-through proxy in front of the bf16 server.

Point the agent's OpenAI base URL at this proxy instead of at vLLM. Every
/v1/chat/completions call is forwarded unchanged, streamed or not, and written to a JSONL
recording in the format turns.py documents. Nothing about the agent changes, so the turns
are its real traffic: its prompts, its tools and bf16's answers.

  vllm serve Qwen/Qwen2.5-7B-Instruct --kv-cache-dtype fp8 --max-model-len 131072 \
      --hf-overrides "$YARN" --enable-auto-tool-choice --tool-call-parser hermes --port 8000
  python record_turns.py --upstream http://127.0.0.1:8000 --port 9000 \
      --out recordings/turns.jsonl --repo /path/to/the/repo/the/agent/edits
  # then run the agent with OPENAI_BASE_URL=http://127.0.0.1:9000/v1

Header X-Session-Id names the session; without it, one session per proxy run. With --repo,
each turn records the repository's HEAD commit at the time of the request, so a replay can
check out the same code.

Standard library only.
"""
import argparse
import datetime
import http.server
import json
import pathlib
import subprocess
import sys
import threading
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import parse_sse  # noqa: E402

LOCK = threading.Lock()


def head_commit(repo):
    if not repo:
        return None
    try:
        return subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def response_from_json(body):
    ch = (body.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    calls = [{"name": (c.get("function") or {}).get("name"),
              "arguments": (c.get("function") or {}).get("arguments")}
             for c in msg.get("tool_calls") or []]
    return {"text": msg.get("content") or "", "tool_calls": calls,
            "finish_reason": ch.get("finish_reason"), "usage": body.get("usage") or {}}


class Recorder:
    def __init__(self, out, repo, session):
        self.out, self.repo, self.session = pathlib.Path(out), repo, session
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.counts = {}

    def write(self, session, request, response):
        with LOCK:
            i = self.counts.get(session, 0)
            self.counts[session] = i + 1
            rec = {"turn_id": f"{session}-{i:04d}", "session_id": session, "index": i,
                   "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   "repo": self.repo, "commit": head_commit(self.repo),
                   "request": request, "response": response, "check": None}
            with open(self.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")


def make_handler(upstream, recorder):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _forward(self, method, body=None):
            req = urllib.request.Request(upstream.rstrip("/") + self.path, data=body,
                                         method=method,
                                         headers={"Content-Type": "application/json"})
            return urllib.request.urlopen(req, timeout=3600)

        def do_GET(self):
            with self._forward("GET") as r:
                data = r.read()
                self.send_response(r.status)
                self.send_header("Content-Type", r.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            record = self.path.rstrip("/").endswith("/chat/completions")
            session = self.headers.get("X-Session-Id") or recorder.session
            try:
                request = json.loads(raw) if record else None
            except ValueError:
                request, record = None, False
            with self._forward("POST", raw) as r:
                if request and request.get("stream"):
                    self.send_response(r.status)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    lines = []
                    for line in r:
                        lines.append(line)
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(line), line))
                        self.wfile.flush()
                    self.wfile.write(b"0\r\n\r\n")
                    if record:
                        p = parse_sse(lines)
                        recorder.write(session, request, {
                            "text": p["text"], "tool_calls": p["tool_calls"],
                            "finish_reason": p["finish_reason"], "usage": p["usage"]})
                    return
                data = r.read()
                self.send_response(r.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                if record:
                    recorder.write(session, request, response_from_json(json.loads(data)))

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--upstream", required=True, help="the bf16 vLLM server, e.g. http://127.0.0.1:8000")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parent
                                         / "recordings" / "turns.jsonl"))
    ap.add_argument("--repo", help="the repository the agent edits; its HEAD is recorded per turn")
    ap.add_argument("--session", default=datetime.datetime.now().strftime("s%Y%m%d%H%M%S"))
    a = ap.parse_args()
    rec = Recorder(a.out, a.repo, a.session)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(a.upstream, rec))
    print(f"recording to {a.out}; point the agent at http://127.0.0.1:{a.port}/v1", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
