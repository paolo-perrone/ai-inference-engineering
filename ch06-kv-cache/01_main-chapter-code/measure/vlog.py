"""Parsers for what vLLM 0.30.0 prints at startup and serves on /metrics.

Every pattern here was read from the v0.30.0 source tree (`pip download vllm==0.30.0
--no-deps`), and each one names the file and the format string it matches, so a later
version that changes a line breaks a test instead of silently returning nothing.

vLLM prefixes each log line with the process, the level, a timestamp and the source file,
for example `(EngineCore pid=812) INFO 10-27 14:02:11 [gpu_worker.py:641] ...`. The parsers
search inside the line, so the prefix never matters.
"""
import re

NUM = r"(-?\d+(?:\.\d+)?)"


def _last(pattern, text, flags=0):
    """The last match, because a restarted engine in one log should report its final start."""
    found = list(re.finditer(pattern, text, flags))
    return found[-1] if found else None


def refusal(text):
    """The KV check that stops a start whose max_model_len does not fit.

    Source: vllm/v1/core/kv_cache_utils.py, _check_enough_kv_cache_memory:
      f"To serve at least one request with the model's max seq len "
      f"({max_model_len}), ({format_gib(needed_memory)} GiB KV "
      f"cache is needed, which is larger than the available KV cache "
      f"memory ({format_gib(available_memory)} GiB). {estimated_msg}" ...
    with estimated_msg = "Based on the available memory, the estimated maximum model
    length is {estimated_max_len}. " only when that estimate is above zero.
    """
    m = _last(r"To serve at least one request with the model's max seq len "
              r"\((\d+)\), \(" + NUM + r" GiB KV cache is needed, which is larger than "
              r"the available KV cache memory \(" + NUM + r" GiB\)\.", text)
    if not m:
        return None
    line = next(l for l in text.splitlines()[::-1] if m.group(0) in l)
    z = re.search(r"the estimated maximum model length is (\d+)\.", line)
    start = line.find("ValueError: ")
    return {"error_first_line": line[start:].strip() if start >= 0 else
            line[line.find("To serve"):].strip(),
            "max_seq_len": int(m.group(1)),
            "kv_cache_needed_gib": float(m.group(2)),
            "kv_cache_available_gib": float(m.group(3)),
            "estimated_max_model_len": int(z.group(1)) if z else None}


def max_len_rejection(text):
    """The config check that refuses a max_model_len above the model's derived limit.

    Source: vllm/config/model.py, _get_and_verify_max_len:
      f"User-specified max_model_len ({max_model_len}) is greater "
      f"than the derived max_model_len ({max_len_key}="
      f"{derived_max_model_len} or model_max_length="
      f"{model_max_length} in model's config.json)."
    In v0.30.0 a YaRN rope type does not multiply max_position_embeddings by its factor,
    so a 131,072 start of Qwen2.5-7B-Instruct can stop here, before the KV check runs.
    """
    m = _last(r"User-specified max_model_len \((\d+)\) is greater than the derived "
              r"max_model_len \((\w+)=([\d.]+) or model_max_length=(\w+) in model's "
              r"config\.json\)\.", text)
    if not m:
        return None
    line = next(l for l in text.splitlines()[::-1] if m.group(0) in l)
    start = line.find("ValueError: ")
    return {"error_first_line": line[start:].strip() if start >= 0 else
            line[line.find("User-specified"):].strip(),
            "requested_max_model_len": int(m.group(1)), "derived_key": m.group(2),
            "derived_max_model_len": int(float(m.group(3)))}


PROCESS_PREFIX = re.compile(r"^\(\w+(?: \w+)* pid=\d+\) ")


def first_error_line(text):
    """The first line of the error the engine printed, for a start that failed.

    Python ends a traceback with an unindented `ExceptionType: message` line. vLLM's child
    processes prefix every line they print with `(EngineCore pid=N) `, so that prefix is
    stripped first. The first exception line is the root cause; the API server's later
    `RuntimeError: Engine core initialization failed` says nothing about why.
    """
    pattern = re.compile(r"^([\w.]+(?:Error|Exception)): ")
    for raw in text.splitlines():
        line = PROCESS_PREFIX.sub("", raw)
        if pattern.match(line):
            return line.strip()
    return None


def available_kv(text):
    """Source: vllm/v1/worker/gpu_worker.py: "Available KV cache memory: %s GiB"."""
    m = _last(r"Available KV cache memory: " + NUM + r" GiB", text)
    return float(m.group(1)) if m else None


def capacity(text):
    """The pool in tokens and the concurrency line, which v0.30.0 prints as ONE line.

    Source: vllm/v1/core/kv_cache_utils.py, update_kv_cache_capacity:
      "%s KV cache size: %s tokens, Maximum concurrency for %s tokens per request: %.2fx"
    with the device ("GPU" on CUDA) and both token counts comma-grouped (f"{n:,}").
    The token count is int(max_concurrency * max_model_len), not blocks times block size.
    """
    m = _last(r"(\w+) KV cache size: ([\d,]+) tokens, Maximum concurrency for ([\d,]+) "
              r"tokens per request: " + NUM + r"x", text)
    if not m:
        return None
    return {"device": m.group(1), "kv_cache_size_tokens": int(m.group(2).replace(",", "")),
            "max_model_len": int(m.group(3).replace(",", "")),
            "max_concurrency": float(m.group(4)), "line": m.group(0)}


def model_loading(text):
    """Source: vllm/v1/worker/gpu_model_runner.py:
    "Model loading took %s GiB memory and %.6f seconds"."""
    m = _last(r"Model loading took " + NUM + r" GiB memory and " + NUM + r" seconds", text)
    return {"weights_gib": float(m.group(1)), "seconds": float(m.group(2))} if m else None


def cudagraph_estimate(text):
    """Source: gpu_model_runner.py: "Estimated CUDA graph memory: %.2f GiB total"."""
    m = _last(r"Estimated CUDA graph memory: " + NUM + r" GiB total", text)
    return float(m.group(1)) if m else None


def cudagraph_capture(text):
    """Source: gpu_model_runner.py: "Graph capturing finished in %.0f secs, took %.2f GiB"."""
    m = _last(r"Graph capturing finished in " + NUM + r" secs, took " + NUM + r" GiB", text)
    return {"seconds": float(m.group(1)), "gib": float(m.group(2))} if m else None


def memory_summary(text):
    """The one INFO line that names every part of the carve-up.

    Source: vllm/v1/worker/gpu_worker.py, after graph capture:
      f"Free memory on device ({free}/{total} GiB) on startup. "
      f"Desired GPU memory utilization is ({util}, {requested} GiB). "
      f"Actual usage is {consumed} GiB for consumed memory (weights + non-torch), "
      f"{peak} GiB for peak activation, and {graphs} GiB for CUDAGraph memory. ..."
      f"Current kv cache memory in use is {kv} GiB."
    Two things the chapter's waterfall needs to know: v0.30.0 never logs non-torch memory
    on its own (only weights + non-torch), and its "peak activation" is
    transient_peak_headroom + the CUDA graph estimate whenever
    VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS is on, which is the default.
    """
    m = _last(r"Free memory on device \(" + NUM + "/" + NUM + r" GiB\) on startup\. "
              r"Desired GPU memory utilization is \(" + NUM + ", " + NUM + r" GiB\)\. "
              r"Actual usage is " + NUM + r" GiB for consumed memory \(weights \+ non-torch\), "
              + NUM + r" GiB for peak activation, and " + NUM + r" GiB for CUDAGraph memory\.",
              text)
    if not m:
        return None
    kv = re.search(r"Current kv cache memory in use is " + NUM + " GiB", text[m.start():])
    keys = ("free_on_startup_gib", "total_gib", "gpu_memory_utilization", "requested_gib",
            "consumed_weights_plus_non_torch_gib", "peak_activation_gib", "cudagraph_actual_gib")
    out = {k: float(v) for k, v in zip(keys, m.groups())}
    out["kv_cache_in_use_gib"] = float(kv.group(1)) if kv else None
    return out


def profile_debug(text):
    """The DEBUG repr of the profiling result, printed only with VLLM_LOGGING_LEVEL=DEBUG.

    Source: vllm/utils/mem_utils.py, MemoryProfilingResult.__repr__:
      f"Memory profiling takes {t:.2f} seconds. Total non KV cache memory: {a}GiB; "
      f"torch peak memory increase: {b}GiB; total consumed (from mem_get_info): {c}GiB; "
      f"weights memory: {d}GiB."
    """
    m = _last(r"Memory profiling takes " + NUM + r" seconds\. Total non KV cache memory: "
              + NUM + r"GiB; torch peak memory increase: " + NUM + r"GiB; total consumed "
              r"\(from mem_get_info\): " + NUM + r"GiB; weights memory: " + NUM + "GiB", text)
    if not m:
        return None
    keys = ("seconds", "non_kv_cache_gib", "torch_peak_increase_gib", "total_consumed_gib",
            "weights_gib")
    return {k: float(v) for k, v in zip(keys, m.groups())}


def backend(text):
    """The attention backend line.

    Source: vllm/platforms/cuda.py, get_attn_backend_cls:
      "Using %s attention backend out of potential backends: %s."   (auto-selected)
      "Using %s backend."                                           (--attention-backend)
    and, at DEBUG only, "Some attention backends are not valid for <device> with <config>.
    Reasons: {NAME: [reason, ...], ...}." which carries FlashAttention's fp8 rejection,
    "FP8 KV cache requires FA3 on SM90 or FA4 on SM100" (v1/attention/backends/flash_attn.py).
    """
    m = _last(r"Using (\w+) attention backend out of potential backends: \[([^\]]*)\]", text)
    out = None
    if m:
        out = {"backend": m.group(1), "line": m.group(0) + ".",
               "candidates": re.findall(r"'(\w+)'", m.group(2))}
    else:
        m = _last(r"Using (?:AttentionBackendEnum\.)?(\w+) backend\.", text)
        if m:
            out = {"backend": m.group(1), "line": m.group(0), "candidates": None}
    r = _last(r"Some attention backends are not valid for .*? Reasons: (\{.*\})\.", text)
    if out is not None:
        out["invalid_reasons"] = r.group(1) if r else None
    return out


def prometheus(text):
    """Parse the Prometheus text format /metrics serves into {name: [(labels, value)]}."""
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = re.match(r'^([a-zA-Z_:][\w:]*)(\{(.*)\})?\s+(\S+)', line)
        if not m:
            continue
        labels = dict(re.findall(r'(\w+)="((?:[^"\\]|\\.)*)"', m.group(3) or ""))
        try:
            value = float(m.group(4))
        except ValueError:
            continue
        out.setdefault(m.group(1), []).append((labels, value))
    return out


# Metric names, from vllm/v1/metrics/loggers.py in v0.30.0 (PrometheusStatLogger).
# Counters are exported with a _total suffix by prometheus_client.
PREFIX_QUERIES = "vllm:prefix_cache_queries"        # "in terms of number of queried tokens"
PREFIX_HITS = "vllm:prefix_cache_hits"              # "in terms of number of cached tokens"
EXTERNAL_QUERIES = "vllm:external_prefix_cache_queries"   # KV connector, the CPU tier
EXTERNAL_HITS = "vllm:external_prefix_cache_hits"
RUNNING = "vllm:num_requests_running"
WAITING = "vllm:num_requests_waiting"
KV_USAGE = "vllm:kv_cache_usage_perc"
PREEMPTIONS = "vllm:num_preemptions"
CACHE_INFO = "vllm:cache_config_info"


def metric(parsed, name):
    """Sum a metric over its label sets, accepting the counter's _total spelling."""
    rows = parsed.get(name) or parsed.get(name + "_total") or []
    return sum(v for _, v in rows) if rows else None


def cache_config_info(parsed):
    """CacheConfig's fields as labels on the vllm:cache_config_info gauge.

    v0.30.0 never logs block_size at INFO for a single-group model like Qwen2.5-7B, so
    this gauge is where the block size the engine chose can be read. Source:
    vllm/v1/metrics/loggers.py log_metrics_info and CacheConfig.metrics_info.
    """
    rows = parsed.get(CACHE_INFO) or []
    return dict(rows[0][0]) if rows else None


def nvidia_smi_memory_total(text):
    """`nvidia-smi --query-gpu=name,memory.total --format=csv,noheader`: 'NVIDIA L4, 23034 MiB'."""
    rows = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            m = re.match(r"(\d+)\s*MiB", parts[1])
            if m:
                rows.append({"name": parts[0], "memory_total_mib": int(m.group(1))})
    return rows


def startup(text):
    """Everything one startup log gives the chapter, in one dict."""
    return {"refusal": refusal(text), "max_len_rejection": max_len_rejection(text),
            "available_kv_cache_gib": available_kv(text), "capacity": capacity(text),
            "model_loading": model_loading(text),
            "cudagraph_estimate_gib": cudagraph_estimate(text),
            "cudagraph_capture": cudagraph_capture(text),
            "memory_summary": memory_summary(text), "profile_debug": profile_debug(text),
            "backend": backend(text), "first_error_line": first_error_line(text)}
