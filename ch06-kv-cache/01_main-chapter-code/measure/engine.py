"""Start vLLM, keep what it printed, time requests, read /metrics.

measure.py's start_engine always adds --no-enable-prefix-caching and treats a failed start
as fatal. Chapter 6 needs caching on, and it needs a refused start as a result, so the
engine handling lives here and the mini-session's helpers are imported where they fit.
"""
import contextlib
import datetime
import json
import os
import pathlib
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
import uuid

import vlog
from prompts import measure

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"
LOGS = RESULTS / "logs"

MODEL = "Qwen/Qwen2.5-7B-Instruct"
VLLM_VERSION = "0.30.0"
COMPLETION_TOKENS = measure.COMPLETION_TOKENS          # 180, the mini-session's answer

# Every flag this fixture passes, each checked in vLLM v0.30.0's vllm/engine/arg_utils.py
# (or entrypoints/launchers/cli_args.py for the server's own flags). test_run.py fails if
# a command uses a flag missing from this set.
FLAGS_V030 = {
    "--host", "--port", "--max-model-len", "--hf-overrides", "--generation-config",
    "--no-enable-prefix-caching", "--kv-cache-dtype", "--kv-cache-dtype-skip-layers",
    "--kv-offloading-size", "--kv-offloading-backend", "--kv-cache-memory-bytes",
    "--enable-prompt-tokens-details",
}

GREEDY = ["--generation-config", "vllm"]

# The draft's 128k command, verbatim (chapter 6, section 6.1).
YARN_AS_PRINTED = ["--hf-overrides", '{"rope_scaling": {"rope_type": "yarn", '
                   '"factor": 4.0, "original_max_position_embeddings": 32768}}']

# The same YaRN setting in the form vLLM v0.30.0 documents (docs/features/
# context_extension.md: `rope_parameters`, with rope_theta), plus max_position_embeddings.
# In v0.30.0 a "yarn" rope type does not multiply max_position_embeddings by its factor
# (vllm/config/model.py, _get_and_verify_max_len, and tests/test_config.py
# test_get_and_verify_max_len_yarn_is_already_scaled), so Qwen2.5-7B-Instruct's 32,768
# would refuse --max-model-len 131072 before the KV check. rope_theta 1000000.0 is the
# value in Qwen2.5-7B-Instruct's config.json.
YARN_V030 = ["--hf-overrides", json.dumps({
    "rope_parameters": {"rope_type": "yarn", "factor": 4.0,
                        "original_max_position_embeddings": 32768, "rope_theta": 1000000.0},
    "max_position_embeddings": 131072})]

YARN_FORM = {"v030": YARN_V030, "printed": YARN_AS_PRINTED}


def yarn(form="v030"):
    return list(YARN_FORM[form])


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- environment / run log

def _py_versions(python):
    code = ("import json, importlib.metadata as m\n"
            "out = {}\n"
            "for p in ('torch', 'vllm', 'transformers', 'flashinfer-python'):\n"
            "    try: out[p] = m.version(p)\n"
            "    except Exception: out[p] = None\n"
            "try:\n"
            "    import torch; out['torch_cuda'] = torch.version.cuda\n"
            "    out['cuda_available'] = torch.cuda.is_available()\n"
            "except Exception as e: out['torch_import_error'] = repr(e)\n"
            "print(json.dumps(out))")
    try:
        r = subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=300)
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as e:
        return {"error": repr(e)}


def environment():
    """GPU name, driver, total memory, CUDA (driver and runtime), PyTorch and vLLM versions.

    Each field names its source, the habit measure.environment() set in chapter 2.
    """
    smi = measure._output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                           "--format=csv,noheader"])
    gpus = []
    for row in smi.splitlines():
        parts = [p.strip() for p in row.split(",")]
        if len(parts) == 3:
            mem = vlog.nvidia_smi_memory_total(f"{parts[0]}, {parts[2]}")
            gpus.append({"name": parts[0], "driver": parts[1],
                         "memory_total_mib": mem[0]["memory_total_mib"] if mem else None})
    header = measure._output(["nvidia-smi"])
    exe = shutil.which("vllm")
    python = measure._interpreter(exe) if exe else None
    return {"gpus": gpus,
            "cuda_supported_by_driver": (header.split("CUDA Version:")[1].split()[0]
                                         if "CUDA Version:" in header else None),
            "python_executable": python or sys.executable,
            "packages": _py_versions(python or sys.executable),
            "vllm_executable": exe,
            "sources": {"gpus": "nvidia-smi --query-gpu=name,driver_version,memory.total",
                        "cuda_supported_by_driver": "nvidia-smi header",
                        "packages": "importlib.metadata and torch.version.cuda in vLLM's "
                                    "interpreter"}}


class RunLog:
    """What every result file carries: the machine, the software, every command run."""

    def __init__(self, sub):
        self.data = {"subcommand": sub, "argv": sys.argv, "started_utc": stamp(),
                     "commands": []}

    def command(self, cmd, env=None, note=None):
        self.data["commands"].append({"argv": cmd, "env": env or {}, "note": note,
                                      "shell": " ".join(_quote(c) for c in cmd)})

    def finish(self, env=None):
        self.data["finished_utc"] = stamp()
        self.data["environment"] = env if env is not None else environment()
        return self.data


def _quote(s):
    import shlex
    return shlex.quote(str(s))


def write_result(name, result):
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_result(name):
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# ---------------------------------------------------------------- the engine

class StartFailed(Exception):
    def __init__(self, info):
        super().__init__(info.get("first_error_line") or "vllm serve did not start")
        self.info = info


def serve_cmd(flags, port=8000):
    exe = shutil.which("vllm") or "vllm"
    return [exe, "serve", MODEL, "--host", "127.0.0.1", "--port", str(port)] + list(flags)


def _get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def scrape(url):
    return vlog.prometheus(_get(url.rstrip("/") + "/metrics"))


@contextlib.contextmanager
def serving(flags, name, runlog, debug=False, timeout=2400, env_extra=None):
    """Start `vllm serve` with `flags`; yield (url, log_path); stop it however the block ends.

    A start that exits before /health answers raises StartFailed with everything the log
    says about why, so a refused start becomes data instead of a crash.
    """
    if not shutil.which("vllm"):
        raise SystemExit(f"vllm is not on PATH; pip install vllm=={VLLM_VERSION}")
    LOGS.mkdir(parents=True, exist_ok=True)
    port = measure._free_port()
    cmd = serve_cmd(flags, port)
    env_add = {"VLLM_LOGGING_LEVEL": "DEBUG" if debug else "INFO", **(env_extra or {})}
    runlog.command(cmd, env_add, note=name)
    log_path = LOGS / f"{name}.log"
    log = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True, env={**os.environ, **env_add})
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + timeout
        while True:
            if proc.poll() is not None:
                log.flush()
                text = log_path.read_text(encoding="utf-8", errors="replace")
                info = vlog.startup(text)
                info.update({"returncode": proc.returncode, "log": str(log_path.relative_to(HERE)),
                             "log_tail": text.splitlines()[-40:]})
                raise StartFailed(info)
            try:
                with urllib.request.urlopen(url + "/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise StartFailed({"first_error_line": f"no /health within {timeout} s",
                                   "log": str(log_path.relative_to(HERE))})
            time.sleep(2)
        yield url, log_path
    finally:
        measure.stop_engine(proc, log)


def startup(name, flags, runlog, fresh=False, debug=True):
    """One engine start whose job is its startup log; cached so later subcommands reuse it.

    Returns the parsed log, the /metrics cache_config_info labels when the start served,
    and whether it started. Cached under results/logs/<name>.json, keyed on the flags.
    """
    cache = LOGS / f"{name}.json"
    if cache.exists() and not fresh:
        old = json.loads(cache.read_text(encoding="utf-8"))
        if old.get("flags") == list(flags):
            old["reused"] = True
            runlog.command(old["command"], old.get("env"), note=f"{name} (reused from cache)")
            return old
    out = {"name": name, "flags": list(flags), "started_utc": stamp()}
    try:
        with serving(flags, name, runlog, debug=debug) as (url, log_path):
            metrics = scrape(url)
            out["cache_config_info"] = vlog.cache_config_info(metrics)
            try:
                out["server_version"] = json.loads(_get(url + "/version")).get("version")
            except (OSError, ValueError):
                out["server_version"] = None
        out["started"] = True
        text = log_path.read_text(encoding="utf-8", errors="replace")
        out["parsed"] = vlog.startup(text)
        out["log"] = str(log_path.relative_to(HERE))
    except StartFailed as e:
        out["started"] = False
        out["parsed"] = e.info
        out["log"] = e.info.get("log")
    last = runlog.data["commands"][-1]
    out["command"], out["env"] = last["argv"], last["env"]
    cache.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


# ---------------------------------------------------------------- timing

def completion(url, prompt_ids, max_tokens=COMPLETION_TOKENS, salt=None, timeout=3600,
               ignore_eos=True):
    """One streamed /v1/completions request on token ids, timed token by token.

    Asks for the generated token ids (return_token_ids, v0.30.0 completion protocol) so
    turn two can resend turn one's answer token for token, and for usage, whose
    prompt_tokens_details.cached_tokens the server fills when started with
    --enable-prompt-tokens-details.
    """
    body = {"model": MODEL, "prompt": list(prompt_ids), "max_tokens": max_tokens,
            "temperature": 0, "ignore_eos": ignore_eos, "stream": True,
            "return_token_ids": True, "stream_options": {"include_usage": True}}
    if salt:
        body["cache_salt"] = salt
    req = urllib.request.Request(url.rstrip("/") + "/v1/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    first = last = None
    text, ids, usage = [], [], {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            msg = json.loads(data)
            usage = msg.get("usage") or usage
            for ch in msg.get("choices") or []:
                got = ch.get("token_ids") or []
                if ch.get("text") or got:
                    last = time.monotonic()
                    first = last if first is None else first
                    text.append(ch.get("text") or "")
                    ids.extend(got)
    n = usage.get("completion_tokens") or len(ids)
    if first is None:
        raise RuntimeError("the endpoint streamed no tokens")
    details = usage.get("prompt_tokens_details") or {}
    return {"ttft_ms": round((first - t0) * 1000, 2),
            "tpot_ms": round((last - first) * 1000 / (n - 1), 3) if n > 1 else None,
            "completion_tokens": n, "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": details.get("cached_tokens"),
            "text": "".join(text), "token_ids": ids}


def median_spread(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return {"median": None, "spread": None, "runs": 0}
    return {"median": round(statistics.median(vals), 3),
            "spread": round(max(vals) - min(vals), 3),
            "min": min(vals), "max": max(vals), "runs": len(vals)}


def new_salt():
    return "ch06-" + uuid.uuid4().hex
