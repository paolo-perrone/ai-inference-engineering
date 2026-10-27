#!/usr/bin/env python3
"""Tests for chapter 5's measurement code. No GPU, no network, no vLLM install.

Every sample log line is rendered from the format string vLLM 0.30.0 prints it with (the
source file and line are named beside each), so a parser that passes here reads the real
line. The numbers inside the samples are parser fixtures only.

  python -m unittest test_run.py      (or: python test_run.py)
"""
import contextlib
import http.server
import io
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C          # noqa: E402
import record_turns as R    # noqa: E402
import run                  # noqa: E402
import turns as T           # noqa: E402

DRAFT = pathlib.Path(os.environ.get("CH5_DRAFT", "/Users/paoloap/Desktop/🕹️/3 - Cash 🪙/1 - Earn 💰/"
                     "1 - Syntax Studio 💎/🎯 Projects/📕 Manning Book/📚 Chapters/ch05/"
                     "ch5 - Model-level optimization.md"))
PREFIX = "(EngineCore pid=4242) INFO 10-27 12:00:00 "


def line(fileinfo, fmt, *args):
    """One vLLM log line: logger.py:22's format around the message."""
    return f"{PREFIX}[{fileinfo}] " + (fmt % args)


def kv_line(tokens, mml, conc):
    # v1/core/kv_cache_utils.py:2395-2402
    return line("kv_cache_utils.py:2395",
                "%s KV cache size: %s tokens, Maximum concurrency for %s tokens per request: %.2fx",
                "GPU", f"{tokens:,}", f"{mml:,}", conc)


def breakdown(consumed, act, graph):
    # v1/worker/gpu_worker.py:863-885, the parts the parser reads
    return (f"{PREFIX}[gpu_worker.py:888] Free memory on device (21.9/22.03 GiB) on startup. "
            f"Desired GPU memory utilization is (0.92, 20.27 GiB). Actual usage is {consumed} "
            f"GiB for consumed memory (weights + non-torch), {act} GiB for peak activation, "
            f"and {graph} GiB for CUDAGraph memory. Replace gpu_memory_utilization config with "
            f"`--kv-cache-memory=1` (0.0 GiB) to fit into requested memory.")


SAMPLE_FP8_LOG = "\n".join([
    line("__init__.py:759", "Selected %s for %s", "CutlassFP8ScaledMMLinearKernel", "Fp8OnlineLinearMethod"),
    line("gpu_model_runner.py:5417", "Model loading took %s GiB memory and %.6f seconds", "8.1", 31.5),
    line("gpu_worker.py:640", "Available KV cache memory: %s GiB", "10.9"),
    breakdown("8.5", "1.2", "0.4"),
    line("gpu_model_runner.py:6930", "Graph capturing finished in %.0f secs, took %.2f GiB", 12.0, 0.41),
    kv_line(380000, 131072, 2.9),
])
SAMPLE_AWQ_LOG = "\n".join([
    line("auto_awq.py:447", "Using %s for AutoAWQMarlinLinearMethod", "MarlinLinearKernel"),
    line("gpu_model_runner.py:5417", "Model loading took %s GiB memory and %.6f seconds", "5.2", 20.0),
    kv_line(500000, 131072, 3.81),
])


class Parsers(unittest.TestCase):
    def test_startup_fp8(self):
        s = C.parse_startup(SAMPLE_FP8_LOG)
        self.assertEqual(s["kernels"][0]["kernel"], "CutlassFP8ScaledMMLinearKernel")
        self.assertEqual(s["model_loading_gib"], 8.1)
        self.assertEqual(s["kv_cache_tokens"], 380000)
        self.assertEqual(s["max_concurrency_for_tokens"], 131072)
        self.assertEqual(s["max_concurrency"], 2.9)
        self.assertEqual((s["consumed_gib"], s["peak_activation_gib"], s["cudagraph_gib"]),
                         (8.5, 1.2, 0.4))
        self.assertEqual(s["available_kv_cache_gib"], 10.9)
        self.assertEqual(s["graph_capture_gib"], 0.41)
        self.assertTrue(s["kv_cache_line"].startswith("GPU KV cache size: 380,000 tokens"))

    def test_startup_awq_and_colour(self):
        s = C.parse_startup("\x1b[32m" + SAMPLE_AWQ_LOG + "\x1b[0m")
        self.assertEqual(s["kernels"][0], {"kernel": "MarlinLinearKernel",
                                           "for": "AutoAWQMarlinLinearMethod",
                                           "line": "Using MarlinLinearKernel for AutoAWQMarlinLinearMethod"})
        self.assertEqual(s["max_concurrency"], 3.81)

    def test_kernel_lines_deduplicated(self):
        two = SAMPLE_FP8_LOG + "\n" + SAMPLE_FP8_LOG.splitlines()[0]
        self.assertEqual(len(C.parse_startup(two)["kernels"]), 1)

    def test_async_off(self):
        # config/vllm.py:1457-1460
        l = line("vllm.py:1457", "Async scheduling not supported with %s-based speculative "
                 "decoding and will be disabled.", "suffix")
        self.assertEqual(C.parse_startup(l)["async_scheduling_disabled_by"], "suffix")

    def test_max_len_error(self):
        # config/model.py:2522-2541, as the ValueError reaches the log
        msg = ("User-specified max_model_len (131072) is greater than the derived max_model_len "
               "(max_position_embeddings=32768 or model_max_length=None in model's config.json)."
               " To allow overriding this maximum, set the env var VLLM_ALLOW_LONG_MAX_MODEL_LEN=1.")
        log = "Traceback (most recent call last):\n  File x\nValueError: " + msg
        got = C.parse_error(log)
        self.assertTrue(got.startswith("User-specified max_model_len (131072) is greater"))
        self.assertIn("max_position_embeddings=32768", got)

    def test_other_error(self):
        log = "INFO ok\ntorch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB"
        self.assertIn("OutOfMemoryError", C.parse_error(log))

    def test_spec_log(self):
        # v1/spec_decode/metrics.py:120-137
        l = line("metrics.py:137", "SpecDecoding metrics: Mean acceptance length: %.2f, "
                 "Accepted throughput: %.2f tokens/s, Drafted throughput: %.2f tokens/s, "
                 "Accepted: %d tokens, Drafted: %d tokens, Per-position acceptance rate: %s, "
                 "Avg Draft acceptance rate: %.1f%%", 2.5, 40.0, 90.0, 400, 900, "0.6, 0.4", 44.4)
        self.assertEqual(C.parse_spec_log(l), [{"mean_acceptance_length": 2.5, "accepted": 400,
                                                "drafted": 900}])

    def test_prometheus(self):
        text = ("# HELP vllm:spec_decode_num_drafts_total Number of spec decoding drafts.\n"
                "# TYPE vllm:spec_decode_num_drafts_total counter\n"
                'vllm:spec_decode_num_drafts_total{engine="0",model_name="m"} 10.0\n'
                'vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="m"} 25.0\n'
                'vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="0"} 9.0\n')
        got = C.parse_prometheus(text)
        self.assertEqual(got["vllm:spec_decode_num_drafts"], 10.0)
        self.assertEqual(got["vllm:spec_decode_num_accepted_tokens"], 25.0)
        zero = {k: 0.0 for k in C.SPEC_COUNTERS}
        self.assertEqual(C.acceptance_from_counters(zero, got), 3.5)
        self.assertIsNone(C.acceptance_from_counters(zero, zero))

    def test_bench(self):
        # benchmarks/serve.py:1286-1305, 1392
        d = {"completed": 5, "failed": 0, "median_ttft_ms": 1200.0, "median_tpot_ms": 30.0,
             "median_itl_ms": 29.0, "output_throughput": 33.0, "ttfts": [1.1, 1.2, 1.3, 1.2, 1.2],
             "input_lens": [8192] * 5, "output_lens": [64] * 5, "errors": ["", "", "", "", ""]}
        with tempfile.TemporaryDirectory() as tmp:
            p = pathlib.Path(tmp) / "b.json"
            p.write_text(json.dumps(d))
            got = C.parse_bench(p)
        self.assertEqual(got["ttfts_s"][0], 1.1)
        self.assertEqual(got["median_tpot_ms"], 30.0)
        self.assertEqual(got["errors"], [])

    def test_sse_chat_with_tools(self):
        chunks = [
            {"choices": [{"index": 0, "delta": {"role": "assistant"}, "token_ids": []}]},
            {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "c1",
                         "function": {"name": "edit_file", "arguments": ""}}]}, "token_ids": [1, 2]}]},
            {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0,
                         "function": {"arguments": "{\"path\": \"a.py\"}"}}]}, "token_ids": [3]}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
            {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 3}},
        ]
        lines = [f"data: {json.dumps(c)}\n".encode() for c in chunks] + [b"data: [DONE]\n"]
        p = C.parse_sse(lines)
        self.assertEqual(p["tool_calls"], [{"name": "edit_file", "arguments": "{\"path\": \"a.py\"}"}])
        self.assertEqual(p["token_ids"], [1, 2, 3])
        self.assertEqual(p["finish_reason"], "tool_calls")
        self.assertEqual((p["first_index"], p["last_index"]), (1, 2))

    def test_sse_completions(self):
        lines = [b'data: {"choices":[{"index":0,"text":"he"}]}', b'data: {"choices":[{"index":0,"text":"llo"}]}',
                 b"data: [DONE]"]
        self.assertEqual(C.parse_sse(lines)["text"], "hello")


class Stats(unittest.TestCase):
    def test_stat5(self):
        s = C.stat5([10, 12, 11, 9, 10])
        self.assertEqual(s["median"], 10)
        self.assertEqual(s["spread_pct"], 30.0)
        self.assertEqual(C.stat5([])["median"], None)

    def test_wilson_bounds(self):
        for k, n in ((0, 10), (3, 100), (50, 100), (100, 100)):
            lo, hi = C.wilson(k, n)
            self.assertLessEqual(lo, k / n)
            self.assertGreaterEqual(hi, k / n)
        self.assertEqual(C.wilson(0, 0), (None, None))


class Commands(unittest.TestCase):
    def test_draft_flag_refused(self):
        bad = C.serve_cmd("qwen7b", ["--tool-strict-level", "parameter"])
        with self.assertRaises(ValueError):
            C.validate_command(bad)

    def test_precisions_valid(self):
        for key, flags in C.PRECISIONS.values():
            self.assertTrue(C.validate_command(C.serve_cmd(key, flags)))
        hf = json.loads(C.PRECISIONS["fp8"][1][C.PRECISIONS["fp8"][1].index("--hf-overrides") + 1])
        self.assertEqual(hf["max_position_embeddings"], 131072)

    def test_spec_methods(self):
        for name in run.SPEC:
            C.validate_command(C.serve_cmd("qwen7b", run.spec_flags(name)))

    def test_prefill_fits_context(self):
        self.assertLessEqual(run.PREFILL_LENGTHS["128k"] + 64, C.MAX_MODEL_LEN)


class Turns(unittest.TestCase):
    REQ = {"messages": [{"role": "user", "content": "x"}],
           "tools": [{"type": "function", "function": {"name": "edit_file", "parameters": {
               "type": "object", "properties": {"path": {"type": "string"},
                                                "line": {"type": "integer"}},
               "required": ["path"], "additionalProperties": False}}}]}

    def test_action(self):
        a = {"tool_calls": [{"name": "edit_file", "arguments": "{\"path\": \"a\", \"line\": 1}"}]}
        b = {"tool_calls": [{"name": "edit_file", "arguments": "{\"line\":1,\"path\":\"a\"}"}]}
        self.assertEqual(T.action(a), T.action(b))
        self.assertNotEqual(T.action(a), T.action({"text": "done"}))
        self.assertEqual(T.action({"text": " a  b "}), T.action({"text": "a b"}))

    def test_validity(self):
        ok = {"tool_calls": [{"name": "edit_file", "arguments": "{\"path\": \"a\"}"}]}
        self.assertEqual(T.validity(ok, self.REQ)["valid"], 1)
        missing = {"tool_calls": [{"name": "edit_file", "arguments": "{\"line\": 2}"}]}
        self.assertEqual(T.validity(missing, self.REQ)["valid"], 0)
        broken = {"tool_calls": [{"name": "edit_file", "arguments": "{\"path\": "}]}
        self.assertEqual(T.validity(broken, self.REQ)["valid"], 0)
        markup = {"text": "<tool_call>\n{\"name\": \"edit_file\", \"arguments\": {\"path\"",
                  "tool_calls": []}
        v = T.validity(markup, self.REQ)
        self.assertEqual((v["attempts"], v["valid"]), (1, 0))
        self.assertEqual(T.validity({"text": "no call"}, self.REQ)["attempts"], 0)

    def test_kl_matches_section_5_2_example(self):
        # Section 5.2's worked example: bf16 0.9 / 0.1, quantized 0.8 / 0.2.
        import math
        ref = {"1": {"logprob": math.log(0.9)}, "2": {"logprob": math.log(0.1)}}
        cand = {"1": {"logprob": math.log(0.8)}, "2": {"logprob": math.log(0.2)}}
        kl = T.kl_position(T.topk_at(ref), T.topk_at(cand))
        expected = 0.9 * math.log(0.9 / 0.8) + 0.1 * math.log(0.1 / 0.2)
        self.assertAlmostEqual(kl, expected, places=12)
        self.assertEqual(T.mean_kl([ref, None], [ref, None]), (0.0, 1))

    def test_needle(self):
        m1, e1 = T.needle_prompt(300, 3)
        m2, e2 = T.needle_prompt(300, 3)
        self.assertEqual((m1, e1), (m2, e2))
        for name, code in e1.items():
            self.assertIn(f"project {name} is {code}", m1[0]["content"])
        answer = "\n".join(f"{n}: {c}" for n, c in e1.items())
        self.assertTrue(T.needle_passed(answer, e1))
        self.assertFalse(T.needle_passed(answer.replace(list(e1.values())[0], "000000"), e1))

    def test_load_and_sessions(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = pathlib.Path(tmp) / "t.jsonl"
            rows = [{"turn_id": f"s-{i}", "session_id": "s", "index": i, "request": self.REQ}
                    for i in (1, 0)]
            p.write_text("\n".join(json.dumps(r) for r in rows))
            turns = T.load_turns(p)
            self.assertEqual([t["index"] for t in T.sessions(turns)["s"]], [0, 1])
            p.write_text(json.dumps({"turn_id": "x"}))
            with self.assertRaises(ValueError):
                T.load_turns(p)

    def test_replay_body_greedy(self):
        turn = {"request": dict(self.REQ, temperature=0.7, top_p=0.8, stream=True),
                "response": {"usage": {"completion_tokens": 100}}}
        b = run.replay_body(turn, "m", greedy=True)
        self.assertEqual((b["temperature"], b["max_tokens"], b["model"]), (0, 200, "m"))
        self.assertNotIn("top_p", b)
        self.assertNotIn("stream", b)

    def test_flips(self):
        turns = [{"turn_id": "a"}, {"turn_id": "b"}]
        ref = {"a": {"text": "x"}, "b": {"text": "y"}}
        cand = {"a": {"text": "x"}, "b": {"text": "z"}}
        f = run.flips_for(turns, ref, cand)
        self.assertEqual((f["turns"], f["flips"]), (2, 1))

    def test_strict_tools(self):
        tools = run.set_strict(run.strict_tools(self.REQ["tools"], run="on0"), True)
        self.assertTrue(tools[0]["function"]["strict"])
        self.assertIn("_runon0", tools[0]["function"]["parameters"]["properties"])
        self.assertNotIn("strict", self.REQ["tools"][0]["function"])   # original untouched


class Recorder(unittest.TestCase):
    """The recording proxy against a stub upstream that speaks the chat API."""

    def test_proxy_records_plain_and_streamed(self):
        class Upstream(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if body.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for c in ({"choices": [{"index": 0, "delta": {"content": "hi"}}]},
                              {"choices": [], "usage": {"completion_tokens": 1}}):
                        self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                    return
                data = json.dumps({"choices": [{"message": {"content": "", "tool_calls": [
                    {"function": {"name": "edit_file", "arguments": "{}"}}]},
                    "finish_reason": "tool_calls"}], "usage": {"completion_tokens": 5}}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        up = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        threading.Thread(target=up.serve_forever, daemon=True).start()
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "turns.jsonl"
            rec = R.Recorder(out, None, "sess")
            px = http.server.ThreadingHTTPServer(
                ("127.0.0.1", 0), R.make_handler(f"http://127.0.0.1:{up.server_port}", rec))
            threading.Thread(target=px.serve_forever, daemon=True).start()
            url = f"http://127.0.0.1:{px.server_port}/v1/chat/completions"
            for stream in (False, True):
                req = urllib.request.Request(url, data=json.dumps(
                    {"messages": [{"role": "user", "content": "q"}], "stream": stream}).encode(),
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=10) as r:
                    r.read()
            px.shutdown()
            got = T.load_turns(out)
        up.shutdown()
        self.assertEqual([t["index"] for t in got], [0, 1])
        self.assertEqual(got[0]["response"]["tool_calls"][0]["name"], "edit_file")
        self.assertEqual(got[1]["response"]["text"], "hi")


class Cli(unittest.TestCase):
    def test_list(self):
        p = subprocess.run([sys.executable, str(HERE / "run.py"), "--list"], capture_output=True,
                           text=True, timeout=60)
        self.assertEqual(p.returncode, 0)
        ids = p.stdout.split()
        self.assertEqual(ids, list(run.MEASUREMENTS))
        self.assertEqual(ids[0], "m00")
        self.assertEqual(ids[-1], "report")
        self.assertLess(ids.index("m02"), ids.index("m03"))
        self.assertLess(ids.index("m14"), ids.index("m15"))

    def test_every_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            for mid in run.MEASUREMENTS:
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    code = run.main([mid, "--dry-run", "--turns", str(pathlib.Path(tmp) / "none.jsonl")])
                self.assertEqual(code, 0, f"{mid} dry run failed:\n{buf.getvalue()}")


class Fixtures(unittest.TestCase):
    """Every [FIXTURE: ...] in the draft has a row in fixtures.md that names a subcommand."""

    longMessage = False                      # print the placeholder, not the whole table

    @staticmethod
    def placeholders(text):
        out = []
        for m in re.finditer(r"\[FIXTURE:[^\]]*\]", text):
            out.append((text.count("\n", 0, m.start()) + 1, " ".join(m.group(0).split())))
        return out

    def test_every_placeholder_mapped(self):
        if not DRAFT.exists():
            self.skipTest(f"draft not on this machine ({DRAFT}); set CH5_DRAFT to check")
        found = self.placeholders(DRAFT.read_text(encoding="utf-8"))
        self.assertGreater(len(found), 0)
        table = (HERE / "fixtures.md").read_text(encoding="utf-8")
        rows = [r for r in table.splitlines() if r.startswith("| ") and "FIXTURE" in r]
        flat = " ".join(table.split())
        for ln, ph in found:
            self.assertIn(ph, flat, f"line {ln} unmapped: {ph}")
            row = next((r for r in rows if ph in " ".join(r.split())), None)
            self.assertIsNotNone(row, f"line {ln}: placeholder not in a table row")
            cells = [c.strip() for c in row.strip("|").split("|")]
            self.assertEqual(cells[0], str(ln), f"line number for: {ph}")
            for sub in re.findall(r"`(m\d\d|report)`", cells[2]):
                self.assertIn(sub, run.MEASUREMENTS)
            self.assertTrue(re.search(r"`(m\d\d|report)`", cells[2]), f"no subcommand for {ph}")
        self.assertEqual(len(rows), len(found), "fixtures.md has rows for placeholders the draft no longer has")


if __name__ == "__main__":
    unittest.main()
