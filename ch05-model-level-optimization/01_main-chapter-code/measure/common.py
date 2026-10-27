"""Shared pieces for chapter 5's measurements: pins, servers, log parsers, stats, results.

Every vLLM flag, log line and environment variable below was read from vLLM 0.30.0's own
source (the sdist on PyPI), and the file and line it came from is named beside it, so a
reader on a later version knows exactly what to re-check.
"""
import datetime
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
MINI = REPO_ROOT / "examples" / "mini-session"
RESULTS = HERE / "results"
RAW = RESULTS / "raw"
RECORDINGS = HERE / "recordings"

VLLM_PIN = "0.30.0"
# vllm/config/speculative.py:1581, the version vLLM 0.30.0's own error message asks for.
ARCTIC_PIN = "0.1.1"
RUNS = 5                                   # every timed cell is the median of five
MINI_PROMPT_TOKENS, MINI_COMPLETION_TOKENS = 1457, 180   # chapter 1's request shape
MAX_MODEL_LEN = 131072


def _load_harness():
    """examples/mini-session/measure.py, imported rather than copied."""
    spec = importlib.util.spec_from_file_location("mini_measure", MINI / "measure.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


harness = _load_harness()

# Hugging Face ids and the revisions read on 2026-10-27 from /api/models/<id>.
# size_gb sums the repo's weight, config and tokenizer files from the same call.
MODELS = {
    "qwen7b": {"id": "Qwen/Qwen2.5-7B-Instruct",
               "revision": "a09a35458c702b33eeacc393d103063234e8bc28", "size_gb": 15.24,
               "vocab_size": 152064, "hidden_size": 3584},
    "qwen7b-awq": {"id": "Qwen/Qwen2.5-7B-Instruct-AWQ",
                   "revision": "b25037543e9394b818fdfca67ab2a00ecc7dd641", "size_gb": 5.58,
                   "vocab_size": 152064, "hidden_size": 3584},
    "coder1.5b": {"id": "Qwen/Qwen2.5-Coder-1.5B",
                  "revision": "df3ce67c0e24480f20468b6ef2894622d69eb73b", "size_gb": 3.10},
    "coder7b": {"id": "Qwen/Qwen2.5-Coder-7B-Instruct",
                "revision": "c03e6d358207e414f1eca0bb1891e29f1db0e242", "size_gb": 15.24},
    "swedev7b": {"id": "zai-org/SWE-Dev-7B",
                 "revision": "d04b618d5fa010f5d62aca8987870b4f8af08449", "size_gb": 15.25},
}

# Chapter 6's 128k override, in the form vLLM 0.30.0 accepts.
#  - docs/features/context_extension.md: `--rope-scaling` is gone; pass `rope_parameters`.
#  - vllm/config/model.py:2468-2477: for rope_type "yarn" vLLM does NOT multiply
#    max_position_embeddings by the factor, and model.py:2517-2541 rejects --max-model-len
#    above it. Qwen2.5's config.json says 32768, so the override also raises
#    max_position_embeddings, or every 128k server stops at startup.
YARN = {"rope_parameters": {"rope_type": "yarn", "factor": 4.0,
                            "original_max_position_embeddings": 32768,
                            "rope_theta": 1000000.0},
        "max_position_embeddings": 131072}
# The form the writing plan prints, kept so preflight shows what it does on 0.30.0.
YARN_PLAN = {"rope_scaling": {"rope_type": "yarn", "factor": 4.0,
                              "original_max_position_embeddings": 32768}}


def serve_flags_128k(extra=()):
    return ["--kv-cache-dtype", "fp8", "--max-model-len", str(MAX_MODEL_LEN),
            "--hf-overrides", json.dumps(YARN)] + list(extra)


# The three precisions of section 5.1: the draft's three serve lines.
PRECISIONS = {
    "bf16": ("qwen7b", serve_flags_128k()),
    # --quantization fp8_per_tensor: vllm/config/quantization.py:188 (online shorthand).
    "fp8": ("qwen7b", serve_flags_128k(["--quantization", "fp8_per_tensor"])),
    "awq": ("qwen7b-awq", serve_flags_128k()),
}

# Every serve and bench flag this folder passes, each found in vLLM 0.30.0's argparse
# definitions (engine/arg_utils.py, entrypoints/launchers/cli_args.py, benchmarks/serve.py,
# benchmarks/datasets/datasets.py). validate_command() refuses any other flag, so a typo
# fails on the Mac in --dry-run instead of on a paid GPU.
SERVE_FLAGS = {"--host", "--port", "--revision", "--tokenizer-revision", "--max-model-len",
               "--hf-overrides", "--kv-cache-dtype", "--quantization",
               "--enable-prefix-caching", "--no-enable-prefix-caching",
               "--speculative-config", "--enable-auto-tool-choice", "--tool-call-parser",
               "--gpu-memory-utilization", "--max-num-seqs", "--max-logprobs",
               "--served-model-name", "--seed"}
BENCH_FLAGS = {"--backend", "--base-url", "--model", "--tokenizer", "--dataset-name",
               "--random-input-len", "--random-output-len", "--max-concurrency",
               "--num-prompts", "--seed", "--ignore-eos", "--save-result", "--result-dir",
               "--result-filename", "--percentile-metrics"}
SPEC_METHODS = {"ngram", "suffix"}         # config/speculative.py:66-77


def validate_command(cmd):
    """Raise ValueError when a command uses a flag this folder never verified."""
    if len(cmd) < 3 or cmd[0] != "vllm" or cmd[1] not in ("serve", "bench"):
        raise ValueError(f"not a vllm serve/bench command: {cmd[:3]}")
    allowed = SERVE_FLAGS if cmd[1] == "serve" else BENCH_FLAGS
    for tok in cmd[3:]:
        if tok.startswith("--") and tok not in allowed:
            raise ValueError(f"unverified flag {tok} in: {shlex.join(cmd)}")
    if "--hf-overrides" in cmd:
        json.loads(cmd[cmd.index("--hf-overrides") + 1])
    if "--speculative-config" in cmd:
        spec = json.loads(cmd[cmd.index("--speculative-config") + 1])
        if spec.get("method") not in SPEC_METHODS:
            raise ValueError(f"speculative method {spec.get('method')} not verified")
    return True


# ---------------------------------------------------------------- statistics

def stat5(values):
    """Median, spread as a percent of the median, and every run, for one cell."""
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"median": None, "spread_pct": None, "runs": []}
    med = statistics.median(vals)
    spread = (max(vals) - min(vals)) / med * 100 if med else None
    return {"median": round(med, 6), "spread_pct": None if spread is None else round(spread, 2),
            "min": round(min(vals), 6), "max": round(max(vals), 6), "runs": vals}


def wilson(k, n, z=1.96):
    """95 percent Wilson interval for k events in n trials, as fractions."""
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


GIB = 1 << 30

# ---------------------------------------------------------------- log parsers
# Each regex names the vLLM 0.30.0 source line that prints it. Lines carry a prefix such as
# "(EngineCore pid=123) INFO 10-27 12:00:00 [gpu_model_runner.py:5417] " (logger.py:22,
# utils/system_utils.py:209), so every pattern searches inside the line.

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _clean(text):
    return ANSI.sub("", text)


# model_executor/kernels/linear/__init__.py:759 "Selected %s for %s"
RE_SELECTED = re.compile(r"Selected (\w+) for (\w+)")
# model_executor/layers/quantization/auto_awq.py:447 "Using %s for AutoAWQMarlinLinearMethod"
RE_AWQ_KERNEL = re.compile(r"Using (\w+) for (AutoAWQMarlinLinearMethod)")
# v1/worker/gpu_model_runner.py:5417 "Model loading took %s GiB memory and %.6f seconds"
RE_LOADING = re.compile(r"Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds")
# v1/core/kv_cache_utils.py:2395 "%s KV cache size: %s tokens, Maximum concurrency for %s
# tokens per request: %.2fx"
RE_KV = re.compile(r"(GPU|[A-Z]+) KV cache size: ([\d,]+) tokens, Maximum concurrency for "
                   r"([\d,]+) tokens per request: ([\d.]+)x")
# v1/worker/gpu_worker.py:640 "Available KV cache memory: %s GiB"
RE_AVAIL_KV = re.compile(r"Available KV cache memory: ([\d.]+) GiB")
# v1/worker/gpu_worker.py:863-885, the memory breakdown message
RE_BREAKDOWN = re.compile(
    r"Actual usage is ([\d.]+) GiB for consumed memory \(weights \+ non-torch\), "
    r"([\d.]+) GiB for peak activation, and ([\d.]+) GiB for CUDAGraph memory")
# v1/worker/gpu_model_runner.py:6930 "Graph capturing finished in %.0f secs, took %.2f GiB"
RE_GRAPHS = re.compile(r"Graph capturing finished in ([\d.]+) secs, took ([\d.]+) GiB")
# config/vllm.py:1457 async scheduling switched off by a speculative method
RE_ASYNC_OFF = re.compile(r"Async scheduling not supported with (\S+)-based speculative "
                          r"decoding and will be disabled")
# v1/spec_decode/metrics.py:120 "SpecDecoding metrics: Mean acceptance length: %.2f, ..."
RE_SPEC = re.compile(r"SpecDecoding metrics: Mean acceptance length: ([\d.]+), "
                     r"Accepted throughput: ([\d.]+) tokens/s, Drafted throughput: ([\d.]+) "
                     r"tokens/s, Accepted: (\d+) tokens, Drafted: (\d+) tokens")
# config/model.py:2524 "User-specified max_model_len (%d) is greater than the derived ..."
RE_MAXLEN = re.compile(r"User-specified max_model_len \(\d+\) is greater than the derived "
                       r"max_model_len \([^\n]*")


def parse_startup(text):
    """Every startup fact chapter 5 quotes, from one server log."""
    t = _clean(text)
    out = {"kernels": [], "model_loading_gib": None, "model_loading_s": None,
           "kv_cache_tokens": None, "max_concurrency": None, "max_concurrency_for_tokens": None,
           "kv_cache_line": None, "available_kv_cache_gib": None,
           "consumed_gib": None, "peak_activation_gib": None, "cudagraph_gib": None,
           "graph_capture_gib": None, "async_scheduling_disabled_by": None}
    for line in t.splitlines():
        for rx in (RE_SELECTED, RE_AWQ_KERNEL):
            m = rx.search(line)
            if m and m.group(0) not in [k["line"] for k in out["kernels"]]:
                out["kernels"].append({"kernel": m.group(1), "for": m.group(2),
                                       "line": m.group(0)})
    if (m := RE_LOADING.search(t)):
        out["model_loading_gib"], out["model_loading_s"] = float(m.group(1)), float(m.group(2))
    if (m := RE_KV.search(t)):
        out["kv_cache_tokens"] = int(m.group(2).replace(",", ""))
        out["max_concurrency_for_tokens"] = int(m.group(3).replace(",", ""))
        out["max_concurrency"] = float(m.group(4))
        out["kv_cache_line"] = m.group(0)
    if (m := RE_AVAIL_KV.search(t)):
        out["available_kv_cache_gib"] = float(m.group(1))
    if (m := RE_BREAKDOWN.search(t)):
        out["consumed_gib"], out["peak_activation_gib"], out["cudagraph_gib"] = (
            float(m.group(1)), float(m.group(2)), float(m.group(3)))
    if (m := RE_GRAPHS.search(t)):
        out["graph_capture_gib"] = float(m.group(2))
    if (m := RE_ASYNC_OFF.search(t)):
        out["async_scheduling_disabled_by"] = m.group(1)
    return out


def parse_spec_log(text):
    """Every SpecDecoding metrics line in a server log."""
    return [{"mean_acceptance_length": float(m.group(1)), "accepted": int(m.group(4)),
             "drafted": int(m.group(5))} for m in RE_SPEC.finditer(_clean(text))]


def parse_error(text):
    """The line that stopped a server: the max_model_len refusal first, else the last error."""
    t = _clean(text)
    if (m := RE_MAXLEN.search(t)):
        return m.group(0).strip()
    errs = [l.strip() for l in t.splitlines()
            if re.search(r"(Error|Exception)\b", l) and "Traceback" not in l]
    return errs[-1] if errs else None


SPEC_COUNTERS = ["vllm:spec_decode_num_drafts", "vllm:spec_decode_num_draft_tokens",
                 "vllm:spec_decode_num_accepted_tokens"]   # v1/spec_decode/metrics.py:228


def parse_prometheus(text, names=SPEC_COUNTERS):
    """Sum each named counter over its label sets, from /metrics text format.

    prometheus_client exposes a Counter named X as X_total, so both spellings count.
    """
    out = {n: 0.0 for n in names}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        metric = re.split(r"[{\s]", line, maxsplit=1)[0]
        for n in names:
            if metric in (n, n + "_total"):
                try:
                    out[n] += float(line.rsplit(" ", 1)[1])
                except ValueError:
                    pass
    return out


def acceptance_from_counters(before, after):
    """Mean acceptance length as vLLM computes it (metrics.py:114): 1 + accepted / drafts."""
    drafts = after["vllm:spec_decode_num_drafts"] - before["vllm:spec_decode_num_drafts"]
    acc = (after["vllm:spec_decode_num_accepted_tokens"]
           - before["vllm:spec_decode_num_accepted_tokens"])
    return None if drafts <= 0 else round(1 + acc / drafts, 4)


# ---------------------------------------------------------------- servers

class Ctx:
    """Run context: dry-run flag, the commands a run issued and the checkpoints it used."""

    def __init__(self, dry=False, mid=""):
        self.dry, self.mid, self.commands, self.models = dry, mid, [], set()

    def note(self, cmd):
        validate_command(cmd)
        self.commands.append(shlex.join(cmd))
        if self.dry:
            print("  " + shlex.join(cmd))


def model_args(key):
    m = MODELS[key]
    return m["id"], ["--revision", m["revision"], "--tokenizer-revision", m["revision"]]


def serve_cmd(key, flags, port=8000, prefix_caching=False):
    mid, rev = model_args(key)
    return (["vllm", "serve", mid, "--host", "127.0.0.1", "--port", str(port)] + rev
            + ["--enable-prefix-caching" if prefix_caching else "--no-enable-prefix-caching"]
            + list(flags))


def server_env(extra=None):
    env = dict(os.environ)
    env["NO_COLOR"] = "1"                       # plain log lines for the parsers
    env.pop("ARCTIC_INFERENCE_ENABLED", None)   # the arctic plugin targets old vLLM; keep off
    env.pop("VLLM_ALLOW_LONG_MAX_MODEL_LEN", None)
    env.update(extra or {})
    return env


class Server:
    """One `vllm serve` process. A failed start is a result to record, never a crash."""

    def __init__(self, ctx, key, flags, name, prefix_caching=False, env=None, timeout=2400):
        self.ctx, self.key, self.name = ctx, key, name
        port = 8000 if ctx.dry else harness._free_port()
        self.cmd = serve_cmd(key, flags, port, prefix_caching)
        self.env, self.timeout = env or {}, timeout
        self.proc = self.url = self.logf = None
        self.log_path = RAW / f"{ctx.mid}-{name}-server.log"
        self.ok, self.error, self.exit_code, self.startup_s = False, None, None, None

    def __enter__(self):
        self.ctx.note(self.cmd)
        self.ctx.models.add(self.key)
        if self.ctx.dry:
            return self
        exe = shutil.which("vllm")
        if not exe:
            self.error = f"vllm is not on PATH; pip install vllm=={VLLM_PIN}"
            return self
        RAW.mkdir(parents=True, exist_ok=True)
        self.logf = open(self.log_path, "w", encoding="utf-8")
        t0 = time.monotonic()
        self.proc = subprocess.Popen([exe] + self.cmd[1:], stdout=self.logf,
                                     stderr=subprocess.STDOUT, start_new_session=True,
                                     env=server_env(self.env))
        url = f"http://127.0.0.1:{self.cmd[self.cmd.index('--port') + 1]}"
        while time.monotonic() - t0 < self.timeout:
            if self.proc.poll() is not None:
                self.exit_code = self.proc.returncode
                self.logf.flush()
                self.error = parse_error(self.log_text()) or f"exit code {self.exit_code}"
                return self
            try:
                with urllib.request.urlopen(url + "/health", timeout=5) as r:
                    if r.status == 200:
                        self.ok, self.url = True, url
                        self.startup_s = round(time.monotonic() - t0, 1)
                        return self
            except OSError:
                pass
            time.sleep(2)
        self.error = f"no /health answer within {self.timeout} s"
        return self

    def log_text(self):
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def startup(self):
        return parse_startup(self.log_text())

    def __exit__(self, *exc):
        if self.proc is not None:
            if self.proc.poll() is None:
                os.killpg(self.proc.pid, signal.SIGTERM)
                try:
                    self.proc.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                    self.proc.wait()
            self.logf.close()
            time.sleep(5)                      # let the driver hand the memory back
        return False

    def record(self):
        return {"name": self.name, "command": shlex.join(self.cmd), "started": self.ok,
                "startup_s": self.startup_s, "exit_code": self.exit_code, "error": self.error,
                "log": (str(self.log_path.relative_to(HERE))
                        if self.log_path.exists() else None)}


# ---------------------------------------------------------------- HTTP

def post(url, path, body, timeout=3600):
    req = urllib.request.Request(url.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get_text(url, path, timeout=30):
    with urllib.request.urlopen(url.rstrip("/") + path, timeout=timeout) as r:
        return r.read().decode()


def served_model(url):
    return json.loads(get_text(url, "/v1/models"))["data"][0]["id"]


def parse_sse(lines):
    """Rebuild one streamed chat or completion response from its SSE lines.

    Returns text, tool calls (name and argument string per index), token ids, usage and
    finish reason, plus the arrival index of the first and last content-bearing chunk.
    """
    text, token_ids, usage, finish, tools = [], [], {}, None, {}
    first = last = None
    for i, raw in enumerate(lines):
        line = raw.strip() if isinstance(raw, str) else raw.decode().strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        msg = json.loads(data)
        usage = msg.get("usage") or usage
        for ch in msg.get("choices") or []:
            delta = ch.get("delta") or {}
            piece = ch["text"] if "text" in ch else delta.get("content")
            got = bool(piece) or bool(delta.get("tool_calls")) or bool(ch.get("token_ids"))
            if got:
                first = i if first is None else first
                last = i
            if piece:
                text.append(piece)
            token_ids += ch.get("token_ids") or []
            for tc in delta.get("tool_calls") or []:
                slot = tools.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                fn = tc.get("function") or {}
                slot["name"] += fn.get("name") or ""
                slot["arguments"] += fn.get("arguments") or ""
            finish = ch.get("finish_reason") or finish
    return {"text": "".join(text), "tool_calls": [tools[k] for k in sorted(tools)],
            "token_ids": token_ids, "usage": usage, "finish_reason": finish,
            "first_index": first, "last_index": last}


def stream_timed(url, path, body, timeout=3600):
    """One streamed request: TTFT, TPOT, turn time, text, tool calls and token ids.

    TPOT is (last token time - first token time) / (completion tokens - 1), the same
    definition as the mini-session harness's decode step and vLLM bench's TPOT.
    """
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(url.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    lines, stamps = [], []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            lines.append(raw)
            stamps.append(time.monotonic())
    end = time.monotonic()
    p = parse_sse(lines)
    n = p["usage"].get("completion_tokens") or len(p["token_ids"])
    first = None if p["first_index"] is None else stamps[p["first_index"]]
    last = None if p["last_index"] is None else stamps[p["last_index"]]
    return {"ttft_s": None if first is None else round(first - t0, 4),
            "tpot_ms": (round((last - first) * 1000 / (n - 1), 3)
                        if first is not None and n and n > 1 else None),
            "turn_s": round(end - t0, 4), "completion_tokens": n,
            "prompt_tokens": p["usage"].get("prompt_tokens"), "text": p["text"],
            "tool_calls": p["tool_calls"], "token_ids": p["token_ids"],
            "finish_reason": p["finish_reason"]}


# ---------------------------------------------------------------- vllm bench serve

def bench_cmd(url, model, tokenizer, input_len, output_len, concurrency, prompts, seed, out):
    return ["vllm", "bench", "serve", "--backend", "openai", "--base-url", url,
            "--model", model, "--tokenizer", tokenizer, "--dataset-name", "random",
            "--random-input-len", str(input_len), "--random-output-len", str(output_len),
            "--max-concurrency", str(concurrency), "--num-prompts", str(prompts),
            "--seed", str(seed), "--ignore-eos", "--percentile-metrics", "ttft,tpot,itl,e2el",
            "--save-result", "--result-dir", str(out.parent), "--result-filename", out.name]


def parse_bench(path):
    """The fields chapter 5 reads from one `vllm bench serve --save-result` file.

    Field names from benchmarks/serve.py:1286-1305 and 1392 (median_<metric>_ms). The
    per-request "ttfts" list is in seconds; the median_* fields are in milliseconds.
    """
    d = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    return {"completed": d.get("completed"), "failed": d.get("failed"),
            "median_ttft_ms": d.get("median_ttft_ms"), "median_tpot_ms": d.get("median_tpot_ms"),
            "median_itl_ms": d.get("median_itl_ms"),
            "output_throughput": d.get("output_throughput"),
            "ttfts_s": d.get("ttfts") or [], "input_lens": d.get("input_lens") or [],
            "output_lens": d.get("output_lens") or [],
            "errors": [e for e in (d.get("errors") or []) if e]}


def run_bench(ctx, cmd):
    ctx.note(cmd)
    if ctx.dry:
        return None
    exe = shutil.which("vllm")
    out = (pathlib.Path(cmd[cmd.index("--result-dir") + 1])
           / cmd[cmd.index("--result-filename") + 1])
    out.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.run([exe] + cmd[1:], capture_output=True, text=True, env=server_env())
    if p.returncode != 0 or not out.exists():
        return {"error": (p.stderr or p.stdout)[-2000:], "returncode": p.returncode}
    return parse_bench(out)


# ---------------------------------------------------------------- run log and results

def versions():
    """Package versions and torch's CUDA build, read in the interpreter vLLM runs under."""
    exe = shutil.which("vllm")
    py = (harness._interpreter(exe) if exe else None) or sys.executable
    code = ("import json,importlib.metadata as m\nout={}\n"
            "for p in ['vllm','torch','transformers','xgrammar','llguidance',"
            "'arctic-inference','arctic_inference','flashinfer-python']:\n"
            "  try: out[p]=m.version(p)\n  except Exception: out[p]=None\n"
            "try:\n import torch; out['torch_cuda']=torch.version.cuda\n"
            "except Exception: out['torch_cuda']=None\nprint(json.dumps(out))")
    try:
        p = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=180)
        return json.loads(p.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError, OSError, subprocess.SubprocessError):
        return {}


def run_log(ctx, argv):
    env = harness.environment("http://127.0.0.1:1") if not ctx.dry else {}
    v = versions() if not ctx.dry else {}
    return {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "command": "python " + shlex.join(argv),
            "gpus": env.get("gpus"), "cuda_supported_by_driver": env.get("cuda_supported_by_driver"),
            "cuda_build_torch": v.get("torch_cuda"), "versions": v,
            "vllm_pinned": VLLM_PIN,
            "vllm_matches_pin": (v.get("vllm") == VLLM_PIN) if v else None,
            "issued": ctx.commands,
            "checkpoints": {MODELS[k]["id"]: MODELS[k]["revision"] for k in sorted(ctx.models)}}


def write_result(mid, payload):
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{mid}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_result(mid):
    p = RESULTS / f"{mid}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]
