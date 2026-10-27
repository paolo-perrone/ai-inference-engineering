"""Helpers for chapter 7's measurements: parsers, statistics, traffic and the HTTP client.

Everything here is standard library, so the parsers and the statistics run and test on a
laptop with no GPU. Every metric name and log line below was read from vLLM 0.30.0's own
source (the PyPI sdist, sha256 5f8f4e89...ef62b), file and line given beside each one.
"""
import json
import math
import random
import re
import statistics
import threading
import time
import urllib.request

# ---------------------------------------------------------------------------------------
# Names read from vLLM 0.30.0's source
# ---------------------------------------------------------------------------------------

# vllm/v1/metrics/loggers.py. prometheus_client appends _total to a counter's samples and
# _bucket/_sum/_count to a histogram's, so those are the names /metrics serves.
M_RUNNING = "vllm:num_requests_running"            # gauge, loggers.py:499
M_WAITING = "vllm:num_requests_waiting"            # gauge, loggers.py:509
M_KV_USAGE = "vllm:kv_cache_usage_perc"            # gauge, 1 means 100 percent, loggers.py:567
M_PREEMPT = "vllm:num_preemptions_total"           # counter vllm:num_preemptions, loggers.py:667
M_PREFIX_Q = "vllm:prefix_cache_queries_total"     # counter, loggers.py:590
M_PREFIX_H = "vllm:prefix_cache_hits_total"        # counter, loggers.py:601
M_QUEUE = "vllm:request_queue_time_seconds"        # histogram, loggers.py:852
M_ITER = "vllm:iteration_tokens_total"             # histogram, loggers.py:769

# vllm/v1/core/kv_cache_utils.py:2395, update_kv_cache_capacity()
RE_CAPACITY = re.compile(
    r"(?:GPU|[A-Z]+) KV cache size: ([\d,]+) tokens, "
    r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x")
# vllm/v1/worker/gpu_worker.py:640, format_gib() rounds to 2 places
RE_KV_MEM = re.compile(r"Available KV cache memory: ([\d.]+) GiB")
# vllm/platforms/cuda.py:538
RE_BACKEND = re.compile(r"Using (\w+) attention backend out of potential backends: \[(.*?)\]")
# vllm/config/scheduler.py:288, INFO
RE_CHUNKED = re.compile(r"Chunked prefill is enabled with max_num_batched_tokens=(\d+)\.")
# vllm/engine/arg_utils.py:2979 and :2989, DEBUG only (VLLM_LOGGING_LEVEL=DEBUG)
RE_DEF_TOKENS = re.compile(r"Defaulting max_num_batched_tokens to (\d+) for (\w+) usage context")
RE_DEF_SEQS = re.compile(r"Defaulting max_num_seqs to (\d+) for (\w+) usage context")
# vllm/v1/core/kv_cache_utils.py:898, the refusal when one max-length request does not fit
RE_KV_REFUSAL = re.compile(r"To serve at least one request with the model's max seq len "
                           r"\((\d+)\), \(([\d.]+) GiB KV cache is needed, which is larger "
                           r"than the available KV cache memory \(([\d.]+) GiB\)")
RE_EST_LEN = re.compile(r"the estimated maximum model length is (\d+)")
RE_OOM = re.compile(r"CUDA out of memory|OutOfMemoryError|No available memory for the cache blocks")


# ---------------------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------------------

def percentile(values, p):
    """Linear interpolation between closest ranks, numpy's default method.

    vLLM's own benchmark (vllm/benchmarks/serve.py:752) calls np.percentile with that
    default, so a p99 here means what `vllm bench serve` means by it.
    """
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def across_runs(values, digits=4):
    """One field over the runs: its median, its spread (max minus min) and every value."""
    xs = [v for v in values if v is not None]
    if not xs:
        return {"median": None, "spread": None, "values": list(values), "n": 0}

    def r(v):
        return round(v, digits)
    return {"median": r(statistics.median(xs)), "spread": r(max(xs) - min(xs)),
            "values": [None if v is None else r(v) for v in values], "n": len(xs)}


def histogram_quantile(q, buckets):
    """Prometheus's histogram_quantile on cumulative (upper bound, count) pairs.

    Linear inside the bucket the rank falls in; the +Inf bucket returns the largest finite
    bound, as Prometheus does. Resolution is the bucket width, which for
    vllm:request_queue_time_seconds is coarse (buckets.py:29: 0.3, 0.5, 0.8, 1, 1.5, 2,
    2.5, 5, 10, 15, 20, ...), so the per-request queue time is the primary number and this
    is the cross-check.
    """
    bs = sorted(buckets, key=lambda b: b[0])
    if not bs or bs[-1][1] <= 0:
        return None
    rank = q * bs[-1][1]
    prev_le, prev_c = 0.0, 0.0
    for le, c in bs:
        if c >= rank:
            if math.isinf(le):
                return prev_le
            if c == prev_c:
                return le
            return prev_le + (le - prev_le) * (rank - prev_c) / (c - prev_c)
        prev_le, prev_c = le, c
    return prev_le


# ---------------------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------------------

_SAMPLE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+(\S+)(?:\s+\d+)?\s*$')
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


def parse_prometheus(text):
    """Prometheus text exposition into {sample name: [(labels, value), ...]}."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _SAMPLE.match(line)
        if not m:
            continue
        name, labels, value = m.group(1), m.group(2) or "", m.group(3)
        try:
            v = float(value)
        except ValueError:
            continue
        out.setdefault(name, []).append((dict(_LABEL.findall(labels)), v))
    return out


def metric_value(parsed, name):
    """Sum of a sample over its label sets (one engine, so normally one row). None if absent."""
    rows = parsed.get(name)
    return None if not rows else sum(v for _, v in rows)


def histogram(parsed, base):
    """Cumulative (le, count) pairs for a histogram, summed over label sets."""
    acc = {}
    for labels, v in parsed.get(base + "_bucket", []):
        le = labels.get("le")
        if le is None:
            continue
        key = math.inf if le in ("+Inf", "Inf") else float(le)
        acc[key] = acc.get(key, 0.0) + v
    return sorted(acc.items())


def histogram_delta(before, after):
    """after minus before, bucket by bucket: the observations made between two scrapes."""
    b = dict(before)
    return [(le, c - b.get(le, 0.0)) for le, c in after]


def parse_startup_log(text):
    """Every number chapter 7 reads from a `vllm serve` log, None where the line is absent."""
    r = {"kv_cache_tokens": None, "max_concurrency_for_tokens": None, "max_concurrency_x": None,
         "available_kv_cache_gib": None, "attention_backend": None,
         "attention_candidates": None, "chunked_prefill_max_num_batched_tokens": None,
         "default_max_num_batched_tokens": None, "default_max_num_seqs": None,
         "usage_context": None, "kv_refusal": None, "out_of_memory": False,
         "capacity_line": None}
    m = RE_CAPACITY.search(text)
    if m:
        r["kv_cache_tokens"] = int(m.group(1).replace(",", ""))
        r["max_concurrency_for_tokens"] = int(m.group(2).replace(",", ""))
        r["max_concurrency_x"] = float(m.group(3))
        r["capacity_line"] = m.group(0)
    m = RE_KV_MEM.search(text)
    if m:
        r["available_kv_cache_gib"] = float(m.group(1))
    m = RE_BACKEND.search(text)
    if m:
        r["attention_backend"] = m.group(1)
        r["attention_candidates"] = re.findall(r"'(\w+)'", m.group(2))
    m = RE_CHUNKED.search(text)
    if m:
        r["chunked_prefill_max_num_batched_tokens"] = int(m.group(1))
    m = RE_DEF_TOKENS.search(text)
    if m:
        r["default_max_num_batched_tokens"] = int(m.group(1))
        r["usage_context"] = m.group(2)
    m = RE_DEF_SEQS.search(text)
    if m:
        r["default_max_num_seqs"] = int(m.group(1))
        r["usage_context"] = m.group(2)
    m = RE_KV_REFUSAL.search(text)
    if m:
        e = RE_EST_LEN.search(text)
        r["kv_refusal"] = {"max_model_len": int(m.group(1)), "needed_gib": float(m.group(2)),
                           "available_gib": float(m.group(3)),
                           "estimated_max_model_len": int(e.group(1)) if e else None}
    r["out_of_memory"] = bool(RE_OOM.search(text))
    return r


def parse_stream_line(line):
    """One SSE line from /v1/completions into a dict, or 'done', or None for anything else."""
    line = line.strip()
    if not line.startswith(b"data:"):
        return None
    data = line[len(b"data:"):].strip()
    if data == b"[DONE]":
        return "done"
    return json.loads(data)


# ---------------------------------------------------------------------------------------
# Per-request numbers, defined the way vllm bench serve defines them (serve.py:619)
# ---------------------------------------------------------------------------------------

def request_numbers(rec):
    """TTFT, TPOT, end-to-end and inter-token gaps for one streamed request, in seconds."""
    times = rec.get("chunk_times") or []
    n = rec.get("tokens") or 0
    if not times:
        return {"ttft": None, "tpot": None, "e2e": None, "itl": []}
    ttft = times[0] - rec["send_t"]
    e2e = times[-1] - rec["send_t"]
    tpot = (e2e - ttft) / (n - 1) if n > 1 else None
    itl = [b - a for a, b in zip(times, times[1:])]
    return {"ttft": ttft, "tpot": tpot, "e2e": e2e, "itl": itl}


def steady_window_throughput(recs):
    """Output tokens per second while every request in the set is decoding at once.

    The window opens when the last request gets its first token and closes when the first
    request emits its last. Prefill falls outside it, so the number is decode throughput at
    the batch the set reached. If the window is empty (the batch never formed, which is
    itself a result: memory or the sequence cap held it down), it falls back to every token
    over the whole span and says so.
    """
    live = [r for r in recs if r.get("chunk_times")]
    if not live:
        return {"tokens_per_s": None, "method": "no tokens"}
    start = max(r["chunk_times"][0] for r in live)
    end = min(r["chunk_times"][-1] for r in live)
    if end - start > 0:
        count = 0
        for r in live:
            ts, toks = r["chunk_times"], r.get("chunk_tokens") or [1] * len(r["chunk_times"])
            count += sum(k for t, k in zip(ts, toks) if start < t <= end)
        if count >= 2 * len(live):
            return {"tokens_per_s": count / (end - start), "method": "steady window",
                    "window_s": end - start}
    first = min(r["send_t"] for r in live)
    last = max(r["chunk_times"][-1] for r in live)
    total = sum(r.get("tokens") or 0 for r in live)
    return {"tokens_per_s": total / (last - first) if last > first else None,
            "method": "whole span (no window where every request decoded together)"}


# ---------------------------------------------------------------------------------------
# Traffic
# ---------------------------------------------------------------------------------------

def arrival_schedule(traffic, seed):
    """The replay's fixed arrival schedule: (time in seconds, kind, session, index).

    Each session sends follow-ups with exponential gaps of the configured mean, seeded, so
    the same seed gives the same schedule on every card. The cold paste lands once, at its
    configured time. Every value comes from traffic.json; the ones the draft does not
    define are marked there and listed by `run.py <id> --dry-run`.
    """
    rng = random.Random(seed)
    ev = [(float(traffic["paste_at_s"]["value"]), "paste", None, 0)]
    gap = float(traffic["followup_mean_gap_s"]["value"])
    dur = float(traffic["replay_duration_s"]["value"])
    for s in range(int(traffic["sessions"]["value"])):
        t, i = 0.0, 0
        while True:
            t += rng.expovariate(1.0 / gap)
            if t > dur:
                break
            ev.append((round(t, 3), "followup", s, i))
            i += 1
    return sorted(ev, key=lambda e: (e[0], e[1], -1 if e[2] is None else e[2]))


def tile(base, n, rotate=0, header=()):
    """Exactly n token ids: a header, then `base` rotated and repeated."""
    if not base:
        raise ValueError("no base tokens to tile")
    out = list(header)[:n]
    k = rotate % len(base)
    body = base[k:] + base[:k]
    while len(out) < n:
        out.extend(body[: n - len(out)])
    return out


def dollars_per_million(price_per_hour, tokens_per_s):
    """$ per million output tokens: one million times the hourly price over tokens an hour."""
    if not tokens_per_s:
        return None
    return price_per_hour * 1e6 / (tokens_per_s * 3600.0)


# ---------------------------------------------------------------------------------------
# HTTP: one streamed request, and a /metrics poller
# ---------------------------------------------------------------------------------------

def post_json(url, body, timeout=60):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get_text(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def stream_request(url, model, prompt_ids, max_tokens, priority=None, cache_salt=None,
                   timeout=3600, tag=None):
    """One streamed completion, timed on every chunk.

    ignore_eos makes every request decode exactly max_tokens. return_token_ids counts the
    tokens in each chunk exactly. With --enable-per-request-metrics the final usage chunk
    carries the server's own queue_time_ms, the same interval vllm:request_queue_time_seconds
    observes (vllm/entrypoints/generate/base/serving.py:84).
    """
    body = {"model": model, "prompt": prompt_ids, "max_tokens": max_tokens,
            "temperature": 0, "ignore_eos": True, "stream": True, "return_token_ids": True,
            "stream_options": {"include_usage": True}}
    if priority is not None:
        body["priority"] = priority
    if cache_salt:
        body["cache_salt"] = cache_salt
    rec = {"tag": tag, "prompt_tokens_sent": len(prompt_ids), "max_tokens": max_tokens,
           "send_t": None, "chunk_times": [], "chunk_tokens": [], "tokens": 0,
           "usage": None, "server_metrics": None, "error": None}
    req = urllib.request.Request(url.rstrip("/") + "/v1/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    rec["send_t"] = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for line in r:
                msg = parse_stream_line(line)
                if msg is None:
                    continue
                if msg == "done":
                    break
                now = time.monotonic()
                if msg.get("usage"):
                    rec["usage"] = msg["usage"]
                if msg.get("metrics"):
                    rec["server_metrics"] = msg["metrics"]
                ch = msg.get("choices") or []
                if ch:
                    k = len(ch[0].get("token_ids") or []) or 1
                    rec["chunk_times"].append(now)
                    rec["chunk_tokens"].append(k)
    except Exception as e:  # noqa: BLE001 - a failed request is a recorded result
        rec["error"] = f"{type(e).__name__}: {e}"
    rec["tokens"] = (rec["usage"] or {}).get("completion_tokens") or sum(rec["chunk_tokens"])
    return rec


class Poller:
    """Scrapes /metrics on a fixed interval in a thread: running, waiting, pool use, preemptions."""

    def __init__(self, url, interval=0.25):
        self.url, self.interval = url.rstrip("/") + "/metrics", interval
        self.samples, self._stop = [], threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            try:
                p = parse_prometheus(get_text(self.url, timeout=5))
                self.samples.append({"t": time.monotonic(),
                                     "running": metric_value(p, M_RUNNING),
                                     "waiting": metric_value(p, M_WAITING),
                                     "kv_usage": metric_value(p, M_KV_USAGE),
                                     "preemptions": metric_value(p, M_PREEMPT)})
            except Exception:  # noqa: BLE001 - one missed scrape is not a failure
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(timeout=10)

    def between(self, t0, t1, key):
        return [s[key] for s in self.samples if t0 <= s["t"] <= t1 and s[key] is not None]
