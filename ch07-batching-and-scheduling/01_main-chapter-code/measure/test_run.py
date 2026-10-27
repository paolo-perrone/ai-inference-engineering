#!/usr/bin/env python3
"""Tests for run.py and ch7lib.py. No GPU: the parsers read text in vLLM 0.30.0's own
formats, the statistics read small hand-checkable arrays, and the subcommands run end to
end against a fake `vllm` that answers like vLLM 0.30.0's OpenAI server.

  python3 test_run.py
  CH7_DRAFT=/path/to/"ch7 - Batching and scheduling.md" python3 test_run.py   # also checks fixtures.md

The sample counts and timings below are parser fixtures, not measurements.
"""
import contextlib
import io
import json
import os
import pathlib
import re
import stat
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ch7lib as lib  # noqa: E402
import run  # noqa: E402

# Built from the format strings in vLLM 0.30.0, with the logger prefix vllm/logger.py:22
# writes and the process tag the engine core adds.
LOG = """\
INFO 10-27 12:00:01 [scheduler.py:288] Chunked prefill is enabled with max_num_batched_tokens=2048.
DEBUG 10-27 12:00:01 [arg_utils.py:2979] Defaulting max_num_batched_tokens to 2048 for OPENAI_API_SERVER usage context.
DEBUG 10-27 12:00:01 [arg_utils.py:2989] Defaulting max_num_seqs to 256 for OPENAI_API_SERVER usage context.
(EngineCore pid=4242) INFO 10-27 12:00:09 [cuda.py:538] Using FLASHINFER attention backend out of potential backends: ['FLASHINFER', 'TRITON_ATTN'].
(EngineCore pid=4242) INFO 10-27 12:01:02 [gpu_worker.py:640] Available KV cache memory: 5.25 GiB
(EngineCore pid=4242) INFO 10-27 12:01:02 [kv_cache_utils.py:2395] GPU KV cache size: 196,608 tokens, Maximum concurrency for 131,072 tokens per request: 1.50x
"""
REFUSAL = ("(EngineCore pid=7) ERROR 10-27 12:01:02 [core.py:1] ValueError: To serve at least one "
           "request with the model's max seq len (131072), (7.0 GiB KV cache is needed, which is "
           "larger than the available KV cache memory (5.25 GiB). Based on the available memory, "
           "the estimated maximum model length is 98304. Try increasing `gpu_memory_utilization`")

# prometheus_client's exposition of the metrics in vllm/v1/metrics/loggers.py
METRICS = """\
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="Qwen/Qwen2.5-7B-Instruct"} 4.0
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="Qwen/Qwen2.5-7B-Instruct"} 2.0
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="Qwen/Qwen2.5-7B-Instruct"} 0.875
# TYPE vllm:num_preemptions counter
vllm:num_preemptions_total{engine="0",model_name="Qwen/Qwen2.5-7B-Instruct"} 3.0
vllm:num_preemptions_created{engine="0",model_name="Qwen/Qwen2.5-7B-Instruct"} 1.7615e+09
# TYPE vllm:request_queue_time_seconds histogram
vllm:request_queue_time_seconds_sum{engine="0",model_name="Qwen/Qwen2.5-7B-Instruct"} 12.5
vllm:request_queue_time_seconds_bucket{engine="0",le="0.3",model_name="Qwen/Qwen2.5-7B-Instruct"} 2.0
vllm:request_queue_time_seconds_bucket{engine="0",le="0.5",model_name="Qwen/Qwen2.5-7B-Instruct"} 4.0
vllm:request_queue_time_seconds_bucket{engine="0",le="0.8",model_name="Qwen/Qwen2.5-7B-Instruct"} 4.0
vllm:request_queue_time_seconds_bucket{engine="0",le="1.0",model_name="Qwen/Qwen2.5-7B-Instruct"} 6.0
vllm:request_queue_time_seconds_bucket{engine="0",le="+Inf",model_name="Qwen/Qwen2.5-7B-Instruct"} 8.0
vllm:request_queue_time_seconds_count{engine="0",model_name="Qwen/Qwen2.5-7B-Instruct"} 8.0
"""


class Stats(unittest.TestCase):
    def test_percentile_matches_numpy_linear(self):
        self.assertEqual(lib.percentile([4, 1, 3, 2], 50), 2.5)
        self.assertAlmostEqual(lib.percentile([1, 2, 3, 4], 99), 3.97)
        self.assertEqual(lib.percentile([7], 99), 7)
        self.assertIsNone(lib.percentile([], 50))
        self.assertEqual(lib.percentile([1, None, 3], 50), 2)

    def test_across_runs(self):
        r = lib.across_runs([3.0, 1.0, 2.0, None])
        self.assertEqual((r["median"], r["spread"], r["n"]), (2.0, 2.0, 3))
        self.assertIsNone(lib.across_runs([None])["median"])

    def test_histogram_quantile_like_prometheus(self):
        p = lib.parse_prometheus(METRICS)
        b = lib.histogram(p, lib.M_QUEUE)
        self.assertEqual(b[-1], (float("inf"), 8.0))
        # rank 4 of 8 lands at the top of the 0.3 to 0.5 bucket
        self.assertAlmostEqual(lib.histogram_quantile(0.5, b), 0.5)
        # rank 2 of 8: the whole first bucket, interpolated from 0
        self.assertAlmostEqual(lib.histogram_quantile(0.25, b), 0.3)
        # rank 5 of 8: halfway through the 0.8 to 1.0 bucket (counts 4 to 6)
        self.assertAlmostEqual(lib.histogram_quantile(0.625, b), 0.9)
        # rank in +Inf returns the largest finite bound
        self.assertEqual(lib.histogram_quantile(0.99, b), 1.0)
        self.assertIsNone(lib.histogram_quantile(0.5, []))

    def test_histogram_delta(self):
        before = [(0.3, 1.0), (float("inf"), 2.0)]
        after = [(0.3, 3.0), (float("inf"), 5.0)]
        self.assertEqual(lib.histogram_delta(before, after), [(0.3, 2.0), (float("inf"), 3.0)])

    def test_dollars_per_million(self):
        # 1e6/3600 tokens a second is one million tokens an hour, so the bill is the hourly price
        self.assertAlmostEqual(lib.dollars_per_million(0.71, 1e6 / 3600), 0.71)
        self.assertIsNone(lib.dollars_per_million(0.71, 0))


class Parsers(unittest.TestCase):
    def test_prometheus_names_and_labels(self):
        p = lib.parse_prometheus(METRICS)
        self.assertEqual(lib.metric_value(p, lib.M_RUNNING), 4.0)
        self.assertEqual(lib.metric_value(p, lib.M_WAITING), 2.0)
        self.assertEqual(lib.metric_value(p, lib.M_KV_USAGE), 0.875)
        self.assertEqual(lib.metric_value(p, lib.M_PREEMPT), 3.0)
        self.assertEqual(p[lib.M_RUNNING][0][0]["model_name"], "Qwen/Qwen2.5-7B-Instruct")
        self.assertIsNone(lib.metric_value(p, "vllm:not_a_metric"))

    def test_startup_log(self):
        r = lib.parse_startup_log(LOG)
        self.assertEqual(r["kv_cache_tokens"], 196608)
        self.assertEqual(r["max_concurrency_for_tokens"], 131072)
        self.assertEqual(r["max_concurrency_x"], 1.5)
        self.assertEqual(r["available_kv_cache_gib"], 5.25)
        self.assertEqual(r["attention_backend"], "FLASHINFER")
        self.assertEqual(r["attention_candidates"], ["FLASHINFER", "TRITON_ATTN"])
        self.assertEqual(r["chunked_prefill_max_num_batched_tokens"], 2048)
        self.assertEqual(r["default_max_num_seqs"], 256)
        self.assertEqual(r["default_max_num_batched_tokens"], 2048)
        self.assertEqual(r["usage_context"], "OPENAI_API_SERVER")
        self.assertFalse(r["out_of_memory"])
        self.assertIsNone(r["kv_refusal"])

    def test_refused_launch_is_a_result(self):
        r = lib.parse_startup_log(REFUSAL + "\ntorch.OutOfMemoryError: CUDA out of memory.")
        self.assertEqual(r["kv_refusal"], {"max_model_len": 131072, "needed_gib": 7.0,
                                           "available_gib": 5.25, "estimated_max_model_len": 98304})
        self.assertTrue(r["out_of_memory"])
        self.assertIsNone(r["max_concurrency_x"])

    def test_stream_lines(self):
        self.assertEqual(lib.parse_stream_line(b"data: [DONE]\n"), "done")
        self.assertIsNone(lib.parse_stream_line(b": keep-alive\n"))
        msg = lib.parse_stream_line(b'data: {"choices":[{"index":0,"text":"x","token_ids":[5]}]}\n')
        self.assertEqual(msg["choices"][0]["token_ids"], [5])

    def test_request_numbers_like_vllm_bench(self):
        rec = {"send_t": 10.0, "chunk_times": [10.5, 10.6, 10.8], "tokens": 3}
        n = lib.request_numbers(rec)
        self.assertAlmostEqual(n["ttft"], 0.5)
        self.assertAlmostEqual(n["e2e"], 0.8)
        self.assertAlmostEqual(n["tpot"], 0.15)        # (0.8 - 0.5) / (3 - 1)
        self.assertEqual([round(x, 6) for x in n["itl"]], [0.1, 0.2])

    def test_steady_window(self):
        a = {"send_t": 0.0, "chunk_times": [1.0, 2.0, 3.0, 4.0], "chunk_tokens": [1] * 4, "tokens": 4}
        b = {"send_t": 0.0, "chunk_times": [2.0, 2.5, 3.0, 3.5, 5.0], "chunk_tokens": [1] * 5, "tokens": 5}
        r = lib.steady_window_throughput([a, b])
        # window (2.0, 4.0]: a emits at 3, 4; b at 2.5, 3, 3.5 -> 5 tokens in 2 s
        self.assertEqual(r["method"], "steady window")
        self.assertAlmostEqual(r["tokens_per_s"], 2.5)
        c = {"send_t": 0.0, "chunk_times": [1.0, 2.0], "chunk_tokens": [1, 1], "tokens": 2}
        d = {"send_t": 0.0, "chunk_times": [3.0, 4.0], "chunk_tokens": [1, 1], "tokens": 2}
        self.assertTrue(lib.steady_window_throughput([c, d])["method"].startswith("whole span"))


class Traffic(unittest.TestCase):
    def test_schedule_is_fixed_and_sorted(self):
        a = lib.arrival_schedule(run.TRAFFIC, 7)
        self.assertEqual(a, lib.arrival_schedule(run.TRAFFIC, 7))
        self.assertNotEqual(a, lib.arrival_schedule(run.TRAFFIC, 8))
        self.assertEqual([e[0] for e in a], sorted(e[0] for e in a))
        self.assertEqual(sum(1 for e in a if e[1] == "paste"), 1)
        dur = run.val("replay_duration_s")
        self.assertTrue(all(0 < e[0] <= dur for e in a))

    def test_tile_exact_length(self):
        t = lib.tile([1, 2, 3], 8, rotate=1, header=[9, 9])
        self.assertEqual(t, [9, 9, 2, 3, 1, 2, 3, 1])
        self.assertEqual(len(lib.tile([1, 2, 3], 131072)), 131072)

    def test_config_valid_and_lengths_checked(self):
        self.assertEqual(run.validate(), [])
        old = run.TRAFFIC["prefix_tokens"]["value"]
        try:
            run.TRAFFIC["prefix_tokens"]["value"] = 122880       # 120k read as 120 x 1024
            self.assertTrue(any("follow-up" in p for p in run.check_lengths()))
        finally:
            run.TRAFFIC["prefix_tokens"]["value"] = old

    def test_every_undefined_input_is_reported(self):
        p = run.provisional(run.REPLAY_KEYS)
        self.assertIn("followup_mean_gap_s", p)
        self.assertNotIn("sessions", p)


class Summary(unittest.TestCase):
    def test_operating_point(self):
        f = {"1": {"tpot_p99_ms": {"median": 60}}, "2": {"tpot_p99_ms": {"median": 80}},
             "3": {"tpot_p99_ms": {"median": 99}}, "4": {"tpot_p99_ms": {"median": 101}}}
        self.assertEqual(run.operating_point(f, 100), 3)
        self.assertIsNone(run.operating_point({"1": {"tpot_p99_ms": {"median": 120}}}, 100))

    def test_largest_spread(self):
        rs = {"x": {"fields": {"a": {"median": 10, "spread": 1, "values": []},
                               "b": {"1": {"median": 4, "spread": 2, "values": []}}}}}
        self.assertEqual(run.largest_spread(rs)["field"], "x/b/1")


class Cli(unittest.TestCase):
    def test_list_prints_every_id_in_order(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = run.main(["--list"])
        self.assertEqual(rc, 0)
        self.assertEqual(out.getvalue().split(), run.ORDER)

    def test_dry_run_everything(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = run.main(["all", "--dry-run"])
        self.assertEqual(rc, 0, out.getvalue())
        self.assertIn("--long-prefill-token-threshold 2048", out.getvalue())
        self.assertIn("--no-async-scheduling", out.getvalue())
        self.assertIn("PROVISIONAL", out.getvalue())


class FixturesTable(unittest.TestCase):
    ROW = re.compile(r"^\| (\d+) \| (\d+) \| `(\[FIXTURE[^`]*\])` \| ([^|]+) \| `([^`]+)` \| (.+) \|$")

    def rows(self):
        text = (HERE / "fixtures.md").read_text(encoding="utf-8")
        return [m.groups() for line in text.splitlines() if (m := self.ROW.match(line))]

    def test_rows_point_at_real_subcommands_and_fields(self):
        rows = self.rows()
        self.assertEqual(len(rows), 83)
        src = "".join((p).read_text(encoding="utf-8") for p in
                      (HERE / "run.py", HERE / "ch7lib.py", run.MINI / "measure.py"))
        for _, line, text, subs, key, _ in rows:
            for s in subs.split(","):
                self.assertIn(s.strip(), run.ORDER, f"line {line}: {s}")
            self.assertTrue(f'"{key}"' in src, f"line {line}: field {key} is produced nowhere")

    def test_every_placeholder_in_the_draft_is_mapped(self):
        draft = os.environ.get("CH7_DRAFT")
        if not draft or not pathlib.Path(draft).exists():
            self.skipTest("set CH7_DRAFT to the chapter 7 draft to check the mapping")
        found = []
        for i, l in enumerate(pathlib.Path(draft).read_text(encoding="utf-8").splitlines(), 1):
            found += [(str(i), m.group(0)) for m in re.finditer(r"\[FIXTURE[^\]]*\]", l)]
        mapped = [(line, text) for _, line, text, *_ in self.rows()]
        unmapped = [f for f in found if f not in mapped]
        stale = [m for m in mapped if m not in found]
        self.assertEqual(unmapped, [], "placeholders in the draft with no row in fixtures.md")
        self.assertEqual(stale, [], "rows in fixtures.md that the draft no longer has")


# A fake `vllm serve`: /health, /version, /tokenize, /metrics, and a streamed
# /v1/completions with one token per chunk and the per-request metrics chunk at the end.
FAKE = r'''#!PYTHON
import json, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
args = sys.argv[1:]
if args[:1] == ["-c"]:
    sys.exit(1)
port = int(args[args.index("--port") + 1])
lock = threading.Lock()
state = {"running": 0, "pre": 0}
print("INFO 10-27 12:00:01 [kv_cache_utils.py:2395] GPU KV cache size: 1,000 tokens, "
      "Maximum concurrency for 4,096 tokens per request: 0.24x", flush=True)

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def send(self, code, body, ctype="application/json"):
        b = body.encode()
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path == "/health": return self.send(200, "")
        if self.path == "/version": return self.send(200, json.dumps({"version": "0.30.0"}))
        if self.path == "/metrics":
            with lock: r = state["running"]
            return self.send(200, "vllm:num_requests_running{engine=\"0\"} %d\n"
                             "vllm:num_requests_waiting{engine=\"0\"} 0\n"
                             "vllm:kv_cache_usage_perc{engine=\"0\"} 0.5\n"
                             "vllm:num_preemptions_total{engine=\"0\"} 0\n"
                             "vllm:prefix_cache_queries_total{engine=\"0\"} 10\n"
                             "vllm:prefix_cache_hits_total{engine=\"0\"} 5\n" % r, "text/plain")
        self.send(404, "")
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize":
            return self.send(200, json.dumps({"count": 3, "max_model_len": 4096,
                                              "tokens": [ord(c) % 500 + 1 for c in body["prompt"]][:400] or [1]}))
        if self.path != "/v1/completions": return self.send(404, "")
        if body.get("priority") and "--scheduling-policy" not in args:
            return self.send(400, json.dumps({"error": "priority needs the priority policy"}))
        n = body["max_tokens"]
        self.send_response(200); self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close"); self.end_headers()
        with lock: state["running"] += 1
        time.sleep(0.01)
        for i in range(n):
            self.wfile.write(("data: " + json.dumps({"choices": [{"index": 0, "text": "x", "token_ids": [7]}]}) + "\n\n").encode())
            self.wfile.flush(); time.sleep(0.002)
        with lock: state["running"] -= 1
        self.wfile.write(("data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": len(body["prompt"]), "completion_tokens": n},
                          "metrics": {"queue_time_ms": 1.5, "time_to_first_token_ms": 10.0}}) + "\n\n").encode())
        self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
        self.close_connection = True

ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
'''

SMALL = {"prefix_tokens": 64, "tail_tokens": 32, "followup_new_tokens": 8, "cold_paste_tokens": 128,
         "cold_paste_max_tokens": 4, "followup_mean_gap_s": 0.5, "replay_duration_s": 1.5,
         "paste_at_s": 0.5, "completion_tokens": 6, "admit128k_prompt_tokens": 200,
         "sweep4k_points": [1, 2], "sweep32k_points": [1, 2], "sweepshared_points": [1, 2],
         "sweep4k_prompt_tokens": 50, "sweep32k_prompt_tokens": 60,
         "static_answer_lengths": [2, 3, 4, 5, 2, 3, 4, 5]}


class EndToEnd(unittest.TestCase):
    """Every subcommand against the fake engine: commands run, files land, fields fill."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = pathlib.Path(cls.tmp.name)
        exe = d / "vllm"
        exe.write_text(FAKE.replace("PYTHON", sys.executable), encoding="utf-8")
        exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
        cls.path = os.environ["PATH"]
        os.environ["PATH"] = f"{d}{os.pathsep}{cls.path}"
        cls.saved = {k: run.TRAFFIC[k]["value"] for k in SMALL}
        for k, v in SMALL.items():
            run.TRAFFIC[k]["value"] = v
        cls.results, cls.logs = run.RESULTS, run.LOGS
        run.RESULTS, run.LOGS = d / "results", d / "runs"
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cls.rc = run.main(["all", "--runs", "2"])
        cls.out = out.getvalue()

    @classmethod
    def tearDownClass(cls):
        os.environ["PATH"] = cls.path
        for k, v in cls.saved.items():
            run.TRAFFIC[k]["value"] = v
        run.RESULTS, run.LOGS = cls.results, cls.logs
        cls.tmp.cleanup()

    def load(self, name):
        return json.loads((run.RESULTS / f"{name}.json").read_text(encoding="utf-8"))

    def test_every_subcommand_wrote_a_result(self):
        self.assertEqual(sorted(p.stem for p in run.RESULTS.glob("*.json")), sorted(run.ORDER), self.out)

    def test_replay_fields(self):
        for name in ("runA", "runB", "runC", "runD"):
            r = self.load(name)
            self.assertEqual(r["status"], "ok", (name, r["failures"], self.out))
            f = r["fields"]
            for k in ("followup_ttft_p50_s", "followup_ttft_p99_s", "followup_itl_p99_ms",
                      "paste_ttft_s", "total_output_tokens_per_s"):
                self.assertEqual(f[k]["n"], 2, (name, k))
            self.assertAlmostEqual(f["followup_queue_time_p50_s"]["median"], 0.0015)
            self.assertIn("--enable-per-request-metrics", r["server_commands"][0])
            self.assertTrue(r["provisional_inputs"])
        self.assertIn("--scheduling-policy", self.load("runC")["server_commands"][0])
        self.assertIn("resolved_scheduler", self.load("runA")["fields"])

    def test_sweeps_and_summary(self):
        s = self.load("sweepshared")
        self.assertEqual(sorted(s["fields"]), ["1", "2"])
        self.assertIsNotNone(s["fields"]["2"]["tokens_per_s"]["median"])
        self.assertIsNotNone(s["fields"]["2"]["dollars_per_million_output_tokens"])
        summ = self.load("summary")["fields"]
        self.assertIn("largest_run_to_run_spread", summ)
        self.assertIn("dollars_per_million_by_point", summ)

    def test_static_and_launches(self):
        st = self.load("static")["fields"]
        self.assertEqual(st["static_tokens_per_s"]["n"], 2)
        la = self.load("launches")["fields"]
        self.assertEqual(la["4k"]["max_concurrency_x"], 0.24)
        self.assertIn("error", la["4k"]["config_probe"])   # no real vLLM here


if __name__ == "__main__":
    unittest.main(verbosity=1)
