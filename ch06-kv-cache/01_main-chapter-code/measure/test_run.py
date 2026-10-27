#!/usr/bin/env python3
"""Tests for chapter 6's measurement code. No GPU, no vLLM, no network.

Parser inputs are vLLM v0.30.0's own format strings filled with obviously synthetic values
(1.23, 4242), each prefixed the way vLLM's child processes prefix their lines. They test
the parsing, never a result. The fake engine is a script named `vllm` on PATH that prints a
startup log and either exits (a refused start) or serves /health and /metrics.

  python3 test_run.py
  CH06_DRAFT=/path/to/draft.md python3 test_run.py     check fixtures.md against a draft
"""
import argparse
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
import engine  # noqa: E402
import prompts  # noqa: E402
import run  # noqa: E402
import vlog  # noqa: E402

DRAFT = pathlib.Path(os.environ.get("CH06_DRAFT", "/Users/paoloap/Desktop/🕹️/3 - Cash 🪙/"
                     "1 - Earn 💰/1 - Syntax Studio 💎/🎯 Projects/📕 Manning Book/📚 Chapters/"
                     "ch06/ch6 - The KV cache.md"))
P = "(EngineCore pid=812) INFO 10-27 14:02:11 [{f}] "

REFUSAL = (
    "(EngineCore pid=812) ERROR 10-27 14:03:40 [core.py:1102] EngineCore failed to start.\n"
    "(EngineCore pid=812) Traceback (most recent call last):\n"
    '(EngineCore pid=812)   File "/x/vllm/v1/core/kv_cache_utils.py", line 900, in '
    "_check_enough_kv_cache_memory\n"
    "(EngineCore pid=812)     raise ValueError(\n"
    "(EngineCore pid=812) ValueError: To serve at least one request with the model's max seq "
    "len (131072), (1.23 GiB KV cache is needed, which is larger than the available KV cache "
    "memory (0.45 GiB). Based on the available memory, the estimated maximum model length is "
    "4242. Try increasing `gpu_memory_utilization` (which also controls CPU memory on the CPU "
    "backend) or decreasing `max_model_len` when initializing the engine. See "
    "https://docs.vllm.ai/en/latest/configuration/conserving_memory/ for more details.\n"
    "RuntimeError: Engine core initialization failed. See root cause above. Failed core "
    "proc(s): {}\n")

MAXLEN = ("(APIServer pid=700) ValueError: User-specified max_model_len (131072) is greater "
          "than the derived max_model_len (max_position_embeddings=32768.0 or "
          "model_max_length=None in model's config.json). To allow overriding this maximum, "
          "set the env var VLLM_ALLOW_LONG_MAX_MODEL_LEN=1.\n")

STARTUP = "\n".join([
    P.format(f="cuda.py:539") + "Using FLASH_ATTN attention backend out of potential "
    "backends: ['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION'].",
    P.format(f="gpu_model_runner.py:5417") + "Model loading took 1.11 GiB memory and "
    "12.345678 seconds",
    P.format(f="gpu_model_runner.py:6823") + "Estimated CUDA graph memory: 0.25 GiB total",
    P.format(f="gpu_worker.py:641") + "Available KV cache memory: 2.5 GiB",
    P.format(f="kv_cache_utils.py:2397") + "GPU KV cache size: 46,800 tokens, Maximum "
    "concurrency for 32,768 tokens per request: 1.43x",
    P.format(f="gpu_model_runner.py:6930") + "Graph capturing finished in 9 secs, took 0.31 GiB",
    P.format(f="gpu_worker.py:874") + "Free memory on device (21.9/22.05 GiB) on startup. "
    "Desired GPU memory utilization is (0.92, 20.29 GiB). Actual usage is 1.5 GiB for "
    "consumed memory (weights + non-torch), 0.75 GiB for peak activation, and 0.31 GiB for "
    "CUDAGraph memory. Replace gpu_memory_utilization config with `--kv-cache-memory=1` "
    "(0.0 GiB) to fit into requested memory, or `--kv-cache-memory=2` (0.0 GiB) to fully "
    "utilize gpu memory. Current kv cache memory in use is 2.5 GiB.",
    "(EngineCore pid=812) DEBUG 10-27 14:02:30 [gpu_worker.py:640] Memory profiling takes "
    "3.21 seconds. Total non KV cache memory: 2.22GiB; torch peak memory increase: 0.66GiB; "
    "total consumed (from mem_get_info): 1.5GiB; weights memory: 1.11GiB.",
    "(EngineCore pid=812) DEBUG 10-27 14:02:10 [cuda.py:495] Some attention backends are not "
    "valid for NVIDIA L4 with AttentionSelectorConfig(kv_cache_dtype=fp8). Reasons: "
    "{FLASH_ATTN: [FP8 KV cache requires FA3 on SM90 or FA4 on SM100]}.",
])

METRICS = """# HELP vllm:prefix_cache_queries_total Prefix cache queries, in terms of number of queried tokens.
# TYPE vllm:prefix_cache_queries_total counter
vllm:prefix_cache_queries_total{engine="0",model_name="Qwen/Qwen2.5-7B-Instruct"} 300.0
vllm:prefix_cache_hits_total{engine="0",model_name="Qwen/Qwen2.5-7B-Instruct"} 256.0
vllm:num_requests_running{engine="0",model_name="Qwen/Qwen2.5-7B-Instruct"} 3.0
vllm:num_requests_waiting{engine="0",model_name="Qwen/Qwen2.5-7B-Instruct"} 2.0
vllm:cache_config_info{block_size="16",cache_dtype="auto",engine="0",num_gpu_blocks="2925"} 1.0
"""


class Parsers(unittest.TestCase):
    def test_refusal(self):
        r = vlog.refusal(REFUSAL)
        self.assertEqual((r["max_seq_len"], r["kv_cache_needed_gib"], r["kv_cache_available_gib"],
                          r["estimated_max_model_len"]), (131072, 1.23, 0.45, 4242))
        self.assertTrue(r["error_first_line"].startswith("ValueError: To serve at least one"))

    def test_refusal_without_estimate(self):
        text = REFUSAL.replace("Based on the available memory, the estimated maximum model "
                               "length is 4242. ", "")
        self.assertIsNone(vlog.refusal(text)["estimated_max_model_len"])

    def test_first_error_line_is_root_cause(self):
        self.assertTrue(vlog.first_error_line(REFUSAL).startswith("ValueError: To serve"))
        self.assertTrue(vlog.first_error_line(MAXLEN).startswith("ValueError: User-specified"))

    def test_max_len_rejection(self):
        r = vlog.max_len_rejection(MAXLEN)
        self.assertEqual((r["requested_max_model_len"], r["derived_key"],
                          r["derived_max_model_len"]), (131072, "max_position_embeddings", 32768))

    def test_startup_lines(self):
        s = vlog.startup(STARTUP)
        self.assertEqual(s["available_kv_cache_gib"], 2.5)
        self.assertEqual(s["capacity"]["kv_cache_size_tokens"], 46800)
        self.assertEqual(s["capacity"]["max_model_len"], 32768)
        self.assertEqual(s["capacity"]["max_concurrency"], 1.43)
        self.assertEqual(s["model_loading"]["weights_gib"], 1.11)
        self.assertEqual(s["cudagraph_estimate_gib"], 0.25)
        self.assertEqual(s["cudagraph_capture"], {"seconds": 9.0, "gib": 0.31})
        ms = s["memory_summary"]
        self.assertEqual((ms["total_gib"], ms["gpu_memory_utilization"],
                          ms["consumed_weights_plus_non_torch_gib"], ms["peak_activation_gib"],
                          ms["cudagraph_actual_gib"], ms["kv_cache_in_use_gib"]),
                         (22.05, 0.92, 1.5, 0.75, 0.31, 2.5))
        self.assertEqual(s["profile_debug"]["torch_peak_increase_gib"], 0.66)
        self.assertEqual(s["backend"]["backend"], "FLASH_ATTN")
        self.assertIn("FLASHINFER", s["backend"]["candidates"])
        self.assertIn("FA3 on SM90", s["backend"]["invalid_reasons"])
        self.assertIsNone(s["refusal"])

    def test_backend_explicit(self):
        b = vlog.backend(P.format(f="cuda.py:478") + "Using AttentionBackendEnum.FLASHINFER backend.")
        self.assertEqual(b["backend"], "FLASHINFER")

    def test_prometheus(self):
        m = vlog.prometheus(METRICS)
        self.assertEqual(vlog.metric(m, vlog.PREFIX_HITS), 256.0)
        self.assertEqual(vlog.metric(m, vlog.PREFIX_QUERIES), 300.0)
        self.assertEqual(vlog.metric(m, vlog.RUNNING), 3.0)
        self.assertIsNone(vlog.metric(m, vlog.EXTERNAL_HITS))
        info = vlog.cache_config_info(m)
        self.assertEqual((info["block_size"], info["num_gpu_blocks"]), ("16", "2925"))

    def test_nvidia_smi(self):
        self.assertEqual(vlog.nvidia_smi_memory_total("NVIDIA L4, 1234 MiB\n"),
                         [{"name": "NVIDIA L4", "memory_total_mib": 1234}])


class Config(unittest.TestCase):
    def args(self):
        return argparse.Namespace(yarn_form="v030", dry_run=True, fresh=False, corpus_root=None,
                                  runs=5, offload_gib=8, trials=20, skip_layers="",
                                  m10_seconds=900, skip_existing=False)

    def test_every_flag_exists_in_v030(self):
        for flags in run.all_flag_sets(self.args()):
            for f in flags:
                if f.startswith("--"):
                    self.assertIn(f, engine.FLAGS_V030, f)

    def test_yarn_forms(self):
        printed = json.loads(engine.YARN_AS_PRINTED[1])
        self.assertIn("rope_scaling", printed)
        v030 = json.loads(engine.YARN_V030[1])
        self.assertEqual(v030["rope_parameters"]["rope_type"], "yarn")
        self.assertEqual(v030["max_position_embeddings"], 131072)

    def test_plain_below_32k_yarn_above(self):
        self.assertNotIn("--hf-overrides", run.cfg("bf16", 32_768))
        self.assertIn("--hf-overrides", run.cfg("bf16", 65_536))

    def test_list(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = run.main(["--list"])
        self.assertEqual(code, 0)
        self.assertEqual(buf.getvalue().split(), run.ORDER)
        self.assertEqual(sorted(run.ORDER), sorted(run.TITLES))
        self.assertEqual(sorted(run.ORDER), sorted(run.GPU_MINUTES))

    def test_dry_run_all(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = run.main(["all", "--dry-run"])
        self.assertEqual(code, 0, buf.getvalue())
        self.assertFalse((HERE / "results").exists() and any((HERE / "results").glob("*.tmp")))


class Fixtures(unittest.TestCase):
    def rows(self):
        text = (HERE / "fixtures.md").read_text(encoding="utf-8")
        return [r for r in re.findall(r"^\| (\d+) \| (\[FIXTURE: [^\]]*\]) \| ([^|]+) \| (.+) \|$",
                                      text, re.M)]

    def test_every_placeholder_mapped(self):
        if not DRAFT.exists():
            self.skipTest(f"draft not found at {DRAFT}; set CH06_DRAFT")
        found = []
        for n, line in enumerate(DRAFT.read_text(encoding="utf-8").splitlines(), 1):
            found += [(str(n), m) for m in re.findall(r"\[FIXTURE: [^\]]*\]", line)]
        mapped = [(n, p) for n, p, _, _ in self.rows()]
        missing = [f for f in found if f not in mapped]
        stale = [m for m in mapped if m not in found]
        self.assertFalse(missing, f"placeholders in the draft with no row in fixtures.md: {missing}")
        self.assertFalse(stale, f"rows in fixtures.md no longer in the draft: {stale}")

    def test_rows_name_real_subcommands_and_fields(self):
        src = "".join((HERE / f).read_text(encoding="utf-8") for f in ("run.py", "engine.py"))
        for n, p, subs, fields in self.rows():
            for s in re.findall(r"m\d\d", subs):
                self.assertIn(s, run.ORDER, (n, p))
            for path in re.findall(r"`([\w.]+)`", fields):
                leaf = path.split(".")[-1]
                if not leaf.isdigit():
                    self.assertTrue(re.search(rf"[\"']{leaf}[\"']", src),
                                    f"line {n}: field {path} is never written")


class Prompts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = pathlib.Path(self.tmp.name) / "vllm"
        for sub in ("v1", "model_executor", "entrypoints"):
            d = root / sub
            d.mkdir(parents=True)
            for i in range(40):
                (d / f"f{i:02}.py").write_text(f"# {sub} {i}\n" + "x = 1\n" * 2000)
        self.root = root
        self.tok = prompts.Tokens(None, allow_estimate=True)
        self.tok.tok = None

    def tearDown(self):
        self.tmp.cleanup()

    def test_needles_exact_length_and_deterministic(self):
        a = prompts.needle_trials(self.tok, self.root, 4096, 6)
        b = prompts.needle_trials(self.tok, self.root, 4096, 6)
        self.assertEqual([t["value"] for t in a], [t["value"] for t in b])
        self.assertTrue(all(t["prompt_tokens"] == 4096 - prompts.NEEDLE_ANSWER_TOKENS for t in a))
        depths = [t["depth"] for t in a]
        self.assertEqual(depths, sorted(depths, reverse=True))

    def test_shared_prefix_sessions(self):
        s = prompts.shared_prefix_sessions(self.tok, self.root, 3000, 200, 4)
        self.assertTrue(all(len(x) == 3200 for x in s))

    def test_repo_prefix_fits_two_turns(self):
        n = prompts.repo_prefix_tokens(self.tok)
        p1 = prompts.repo_turn_one(self.tok, self.root, n)
        p2 = prompts.turn_two(p1, [0] * engine.COMPLETION_TOKENS, self.tok)
        self.assertLessEqual(len(p2) + engine.COMPLETION_TOKENS, 32_768)
        self.assertEqual(p2[:len(p1)], p1)

    def test_recovered(self):
        self.assertTrue(prompts.recovered(" 4242424", "4242424"))
        self.assertFalse(prompts.recovered("I do not know", "4242424"))


FAKE_VLLM = r'''#!PYTHON
import json, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer
args = sys.argv[1:]
port = int(args[args.index("--port") + 1])
log = json.load(open(__file__ + ".json"))
print(log["text"], flush=True)
if log["exit"] is not None:
    sys.exit(log["exit"])
METRICS = log["metrics"]
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        body = {"/health": "", "/version": json.dumps({"version": "0.30.0"}),
                "/metrics": METRICS}.get(self.path)
        self.send_response(200 if body is not None else 404); self.end_headers()
        self.wfile.write((body or "").encode())
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        for i in range(req["max_tokens"]):
            c = {"choices": [{"index": 0, "text": " t", "token_ids": [7 + i]}]}
            self.wfile.write(b"data: " + json.dumps(c).encode() + b"\n\n"); self.wfile.flush()
        u = {"choices": [], "usage": {"prompt_tokens": len(req["prompt"]),
             "completion_tokens": req["max_tokens"], "prompt_tokens_details": {"cached_tokens": 16}}}
        self.wfile.write(b"data: " + json.dumps(u).encode() + b"\n\ndata: [DONE]\n\n")
HTTPServer(("127.0.0.1", port), H).serve_forever()
'''


class FakeEngine(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = pathlib.Path(self.tmp.name)
        self.exe = d / "vllm"
        self.exe.write_text(FAKE_VLLM.replace("PYTHON", sys.executable))
        self.exe.chmod(self.exe.stat().st_mode | stat.S_IEXEC)
        self.path = os.environ["PATH"]
        os.environ["PATH"] = f"{d}{os.pathsep}{self.path}"
        self.logs, self.results = engine.LOGS, engine.RESULTS
        engine.RESULTS = d / "results"
        engine.LOGS = engine.RESULTS / "logs"
        engine.HERE = d

    def tearDown(self):
        os.environ["PATH"] = self.path
        engine.LOGS, engine.RESULTS, engine.HERE = self.logs, self.results, HERE
        self.tmp.cleanup()

    def configure(self, text, exit_code):
        pathlib.Path(str(self.exe) + ".json").write_text(
            json.dumps({"text": text, "exit": exit_code, "metrics": METRICS}))

    def test_refused_start_is_a_result(self):
        self.configure(REFUSAL, 1)
        out = engine.startup("bf16-131072", ["--max-model-len", "131072"], engine.RunLog("t"))
        self.assertFalse(out["started"])
        self.assertEqual(out["parsed"]["refusal"]["estimated_max_model_len"], 4242)
        again = engine.startup("bf16-131072", ["--max-model-len", "131072"], engine.RunLog("t"))
        self.assertTrue(again.get("reused"))

    def test_started_engine_gives_block_size_and_timing(self):
        self.configure(STARTUP, None)
        log = engine.RunLog("t")
        out = engine.startup("bf16-32768", ["--max-model-len", "32768"], log)
        self.assertTrue(out["started"])
        self.assertEqual(out["cache_config_info"]["block_size"], "16")
        self.assertEqual(out["parsed"]["capacity"]["kv_cache_size_tokens"], 46800)
        with engine.serving(["--max-model-len", "32768"], "timing", log) as (url, _):
            r = engine.completion(url, [1, 2, 3], max_tokens=4, salt=engine.new_salt())
            pairs = run.turn_pairs(url, [1, 2, 3], prompts.Tokens(None, allow_estimate=True), 2)
        self.assertEqual(r["token_ids"], [7, 8, 9, 10])
        self.assertEqual(r["cached_tokens"], 16)
        self.assertEqual(pairs["turn_two_ttft_ms"]["runs"], 2)
        self.assertEqual(len(pairs["per_run"]), 3)
        self.assertTrue(pairs["per_run"][0]["warmup"])
        self.assertTrue(all(c["argv"][1] == "serve" for c in log.data["commands"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
