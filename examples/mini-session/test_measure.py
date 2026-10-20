#!/usr/bin/env python3
"""Tests for measure.py's live modes, run against a fake engine so they need no GPU.

The fake answers like vLLM 0.30.0's OpenAI server: /health, /version, and a streamed
/v1/completions that sends one token per chunk and a usage chunk at the end. It sleeps a
fixed prefill time and a fixed step time, so the expected medians are known in advance, and
it runs slower when started with CUDA graphs off, so the pair has a known ratio.

  python3 test_measure.py
"""
import json
import os
import pathlib
import stat
import subprocess
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import measure  # noqa: E402

PREFILL_S, STEP_S, STEP_OFF_S = 0.05, 0.010, 0.012

FAKE_ENGINE = r'''
import json, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

args = sys.argv[1:]
port = int(args[args.index("--port") + 1])
graphs_off = "--compilation-config" in args and "NONE" in args[args.index("--compilation-config") + 1]
step = STEP_OFF_S if graphs_off else STEP_S
seen = open(sys.argv[0] + ".requests", "a")

class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        body = {"/health": b"", "/version": json.dumps({"version": "0.30.0"}).encode()}.get(self.path)
        self.send_response(200 if body is not None else 404)
        self.end_headers()
        self.wfile.write(body or b"")

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        seen.write(json.dumps({"args": args, "request": {k: v for k, v in req.items() if k != "prompt"}}) + "\n")
        seen.flush()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        time.sleep(PREFILL_S)
        for i in range(req["max_tokens"]):
            if i:
                time.sleep(step)
            chunk = {"choices": [{"index": 0, "text": " t%d" % i}], "usage": None}
            self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            self.wfile.flush()
        usage = {"choices": [], "usage": {"prompt_tokens": 1600, "completion_tokens": req["max_tokens"]}}
        self.wfile.write(b"data: " + json.dumps(usage).encode() + b"\n\ndata: [DONE]\n\n")

ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
'''.replace("STEP_OFF_S", repr(STEP_OFF_S)).replace("STEP_S", repr(STEP_S)).replace("PREFILL_S", repr(PREFILL_S))


class LiveModes(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.vllm = self.tmp / "vllm"
        self.vllm.write_text(f"#!{sys.executable}\n" + FAKE_ENGINE, encoding="utf-8")
        self.vllm.chmod(self.vllm.stat().st_mode | stat.S_IEXEC)
        self.requests = pathlib.Path(str(self.vllm) + ".requests")
        self.old_path, self.old_runs = os.environ["PATH"], measure.RUNS
        os.environ["PATH"] = f"{self.tmp}{os.pathsep}{self.old_path}"
        measure.RUNS = self.tmp / "runs"

    def tearDown(self):
        os.environ["PATH"], measure.RUNS = self.old_path, self.old_runs

    def seen(self):
        return [json.loads(line) for line in self.requests.read_text().splitlines()]

    def test_graphs_pair(self):
        r = measure.graphs_pair("Qwen/Qwen2.5-7B-Instruct", runs=5)
        off, on = r["settings"]["graphs_off"], r["settings"]["graphs_on"]
        self.assertEqual((off["runs"], on["runs"]), (5, 5))
        self.assertAlmostEqual(on["decode_step_ms"], STEP_S * 1000, delta=4)
        self.assertAlmostEqual(off["decode_step_ms"], STEP_OFF_S * 1000, delta=4)
        self.assertGreater(r["graphs_off_over_on"], 1.0)
        self.assertGreaterEqual(on["ttft_s"], PREFILL_S)
        self.assertEqual(on["per_run"][0]["engine_prompt_tokens"], 1600)
        # both engines ran without prefix caching, and only the first had graphs off
        self.assertIn("--no-enable-prefix-caching", off["server_command"])
        self.assertIn("--no-enable-prefix-caching", on["server_command"])
        self.assertIn('{"cudagraph_mode": "NONE"}', off["server_command"])
        self.assertNotIn("--compilation-config", on["server_command"])
        # one warm-up plus 5 timed runs per setting, every one with its own cache salt
        seen = self.seen()
        self.assertEqual(len(seen), 12)
        salts = [s["request"]["cache_salt"] for s in seen]
        self.assertEqual(len(set(salts)), len(salts))
        for s in seen:
            self.assertTrue(s["request"]["ignore_eos"])
            self.assertEqual(s["request"]["max_tokens"], 180)
            self.assertTrue(s["request"]["stream_options"]["include_usage"])
        # the run log is on disk and names the vLLM version the server reported
        log = json.loads((measure.HERE / r["run_log"]).read_text())
        self.assertEqual(log["environment"]["vllm"], "0.30.0")
        self.assertEqual(set(log["settings"]), {"graphs_off", "graphs_on"})
        # both engines were stopped
        for name in ("graphs_off", "graphs_on"):
            pid_alive = subprocess.run(["pgrep", "-f", str(self.vllm)], capture_output=True).stdout
            self.assertEqual(pid_alive.strip(), b"", name)

    def test_engine_that_dies_is_reported(self):
        self.vllm.write_text(f"#!{sys.executable}\nimport sys\nsys.exit(3)\n", encoding="utf-8")
        with self.assertRaises(SystemExit) as e:
            measure.graphs_pair("m", runs=1)
        self.assertIn("exited with code 3", str(e.exception))


class Baseline(unittest.TestCase):
    def test_merge_keeps_other_chapters_fields(self):
        path = pathlib.Path(tempfile.mkdtemp()) / "baseline.json"
        path.write_text(json.dumps({"context_tokens": 1457, "decode_step_ms": 48.1}))
        measure.write_baseline({"ttft_s": 0.21}, path)
        self.assertEqual(json.loads(path.read_text()),
                         {"context_tokens": 1457, "decode_step_ms": 48.1, "ttft_s": 0.21})
        measure.write_baseline(measure.run("naive", 4), path)
        merged = json.loads(path.read_text())
        self.assertEqual((merged["decode_step_ms"], merged["ttft_s"]), (48.1, 0.21))
        self.assertEqual(merged["cost_per_completed_task_usd"], 0.001959)


if __name__ == "__main__":
    unittest.main(verbosity=2)
