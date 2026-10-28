"""vLLM engine sessions for chapter 3: start `vllm serve` in one configuration, send the
mini-session's request 5 times, keep every chunk's timestamp, stop the engine.

The start and stop follow chapter 2's harness (examples/mini-session/measure.py: prefix
caching off, wait on /health, kill the process group). Every flag below was read from
vLLM 0.30.0's source:
  --generation-config vllm         engine/arg_utils.py:946, config/model.py:330
  --compilation-config / -cc       engine/arg_utils.py:1723; cudagraph_mode NONE is
                                   config/compilation.py CUDAGraphMode.NONE
  -cc.custom_ops=["all"]           utils/argparse_utils.py dotted-arg parsing,
                                   config/compilation.py custom_ops
  --enforce-eager                  engine/arg_utils.py:908; config/vllm.py:1546 turns off
                                   torch.compile AND CUDA graphs
  -O0                              utils/argparse_utils.py:341 -> --optimization-level 0,
                                   config/vllm.py OptimizationLevel.O0 (no compile, no graphs)
  --profiler-config {"profiler": "cuda"}   config/profiler.py, /start_profile and
                                   /stop_profile in entrypoints/serve/profile/api_router.py
"""
import json
import os
import shutil
import subprocess
import time

import common
import stream_client

GREEDY_DEFAULTS = ["--generation-config", "vllm"]   # the plan's "every run" setting

CONFIGS = {
    # name: (flags, use --generation-config vllm)
    "default": ([], True),
    "graphs_none": (["--compilation-config", json.dumps({"cudagraph_mode": "NONE"})], True),
    "enforce_eager": (["--enforce-eager"], True),
    "custom_ops_all": (['-cc.custom_ops=["all"]'], True),
    "O0": (["-O0"], True),
    "qwen_defaults": ([], False),
}


def serve_cmd(config, port, extra=(), exe="vllm"):
    flags, greedy = CONFIGS[config]
    return [exe, "serve", common.MODEL, "--host", "127.0.0.1", "--port", str(port),
            "--no-enable-prefix-caching", "--dtype", common.DTYPE] + \
        (GREEDY_DEFAULTS if greedy else []) + list(flags) + list(extra)


def validate_configs():
    """Dry-run checks: every JSON flag parses, every config builds a command."""
    problems = []
    for name, (flags, _) in CONFIGS.items():
        for f in flags:
            if f.startswith("{") or "=[" in f or "={" in f:
                payload = f.split("=", 1)[1] if f.startswith("-") else f
                try:
                    json.loads(payload)
                except ValueError as e:
                    problems.append(f"{name}: {f} is not JSON ({e})")
        serve_cmd(name, 8000)
    return problems


class Engine:
    def __init__(self, config, log_path, prefix=(), extra=(), timeout=2400):
        self.config, self.log_path, self.prefix, self.extra = config, log_path, list(prefix), list(extra)
        self.timeout = timeout
        self.proc = self.url = self.cmd = self.log = None

    def __enter__(self):
        import urllib.request
        exe = shutil.which("vllm")
        if not exe:
            raise RuntimeError("vllm is not on PATH; install vllm==0.30.0")
        port = common.mini._free_port()
        self.cmd = self.prefix + serve_cmd(self.config, port, self.extra, exe)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = open(self.log_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen(self.cmd, stdout=self.log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        self.url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                self.log.close()
                raise RuntimeError(f"vllm serve exited with code {self.proc.returncode}; see {self.log_path}")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=5) as r:
                    if r.status == 200:
                        return self
            except OSError:
                pass
            time.sleep(2)
        self.__exit__(None, None, None)
        raise RuntimeError(f"vllm serve did not answer /health in {self.timeout} s; see {self.log_path}")

    def __exit__(self, *exc):
        if self.proc is not None:
            common.mini.stop_engine(self.proc, self.log)

    def post(self, path, timeout=600):
        import urllib.request
        req = urllib.request.Request(self.url + path, data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status

    def log_text(self):
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""


def timed_runs(url, prompt, runs=common.RUNS, warmup=common.WARMUP, max_tokens=None, greedy=True):
    max_tokens = max_tokens or common.COMPLETION_TOKENS
    body = lambda: stream_client.request_body(common.MODEL, prompt, max_tokens, greedy=greedy)
    for _ in range(warmup):
        stream_client.stream_once(url, body())
    per = [stream_client.stream_once(url, body()) for _ in range(runs)]
    return {"per_run": per, **stream_client.aggregate(per)}


def session(config, prompts, fresh=False, runs=common.RUNS):
    # qwen_defaults sends no sampling fields, so the model's own sampling applies
    greedy = CONFIGS[config][1]
    """Start one engine config, time each named prompt, and cache the raw result.

    prompts: {label: text or token-id list}. A cached raw file is reused only when it
    covers every label asked for, so measurements stay independent but never pay for the
    same engine start twice in one Colab session.
    """
    raw_path = common.RAW / f"serve-{config}.json"
    cached = common.read_json(raw_path)
    if cached and not fresh and set(prompts) <= set(cached.get("prompts", {})):
        return cached
    stamp = common.now()
    log_path = common.RAW / f"serve-{config}-{stamp}.log"
    out = {"config": config, "started_utc": stamp, "server_log": str(log_path), "prompts": {}}
    with Engine(config, log_path) as eng:
        out["server_command"] = eng.cmd
        out["environment_ch2_harness"] = common.mini.environment(eng.url, eng.cmd[0])
        for label, prompt in prompts.items():
            out["prompts"][label] = timed_runs(eng.url, prompt, runs=runs, greedy=greedy)
        out["server_log_tail"] = eng.log_text()[-20000:]
    common.RAW.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    return out
