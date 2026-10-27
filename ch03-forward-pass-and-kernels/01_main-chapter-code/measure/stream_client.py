"""A streaming client that timestamps every chunk and counts the tokens in each.

Chapter 2's harness times the first and last token. Chapter 3 also needs every gap (mean
ITL) and the tokens per chunk, so this client keeps the whole timeline. Standard library
only. The request fields come from vLLM 0.30.0's completion protocol:
entrypoints/openai/completion/protocol.py (ignore_eos, cache_salt, return_token_ids) and
entrypoints/generate/base/protocol.py (stream_options.include_usage,
stream_options.continuous_usage_stats).
"""
import json
import statistics
import time
import uuid


def request_body(model, prompt, max_tokens, greedy=True, ignore_eos=True, salt=True):
    """`prompt` is the text, or a list of token IDs for an exact length.

    greedy=False sends no sampling fields at all, so the server's defaults apply: vLLM's
    neutral ones under --generation-config vllm, Qwen's generation_config.json otherwise.
    """
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "stream": True,
            "return_token_ids": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True}}
    if greedy:
        body["temperature"] = 0
    if ignore_eos:
        body["ignore_eos"] = True
    if salt:
        body["cache_salt"] = uuid.uuid4().hex   # no prefix-cache hit from an earlier run
    return body


def parse_sse_line(line):
    """One line of the server-sent event stream: a dict, the string "DONE", or None.

    vLLM 0.30.0 writes `data: {json}\\n\\n` per chunk and `data: [DONE]\\n\\n` at the end
    (entrypoints/openai/completion/serving.py).
    """
    if isinstance(line, bytes):
        line = line.decode("utf-8", errors="replace")
    line = line.strip()
    if not line.startswith("data:"):
        return None
    data = line[len("data:"):].strip()
    if data == "[DONE]":
        return "DONE"
    return json.loads(data)


def chunk_tokens(msg, prev_completion_tokens):
    """Tokens this chunk carries, and the running completion count after it.

    The token_ids list (return_token_ids) is the direct count. The continuous usage stats'
    running completion_tokens give a second count by difference; both are kept.
    """
    choices = msg.get("choices") or []
    usage = msg.get("usage") or {}
    running = usage.get("completion_tokens")
    by_ids = None
    if choices and choices[0].get("token_ids") is not None:
        by_ids = len(choices[0]["token_ids"])
    by_usage = None
    if running is not None and choices:
        by_usage = running - (prev_completion_tokens or 0)
    return by_ids, by_usage, running


def timeline_from_lines(lines, t0, clock):
    """Walk a stream, calling clock() at each chunk, and return the request's timeline.

    Split out from the network so tests can feed recorded lines with a fake clock.
    """
    chunks, usage, running = [], {}, 0
    prompt_ids_len = None
    for raw in lines:
        msg = parse_sse_line(raw)
        if msg is None:
            continue
        if msg == "DONE":
            break
        t = clock()
        if msg.get("usage") and not msg.get("choices"):
            usage = msg["usage"]                 # the final usage-only chunk
            continue
        choices = msg.get("choices") or []
        if not choices:
            continue
        if choices[0].get("prompt_token_ids") is not None and prompt_ids_len is None:
            prompt_ids_len = len(choices[0]["prompt_token_ids"])
        by_ids, by_usage, new_running = chunk_tokens(msg, running)
        if new_running is not None:
            running = new_running
        n = by_ids if by_ids is not None else by_usage
        if not n:
            continue                             # an empty chunk carries no token
        chunks.append({"t": t - t0, "tokens": n, "tokens_by_usage": by_usage})
    return {"chunks": chunks, "usage": usage, "prompt_token_ids_len": prompt_ids_len,
            "end": clock() - t0}


def summarize_timeline(tl):
    """TTFT, E2E, TPOT (per request: (E2E - TTFT) / (n - 1)), chunk gaps, tokens per chunk.

    E2E is the arrival of the last token-carrying chunk, so the final usage chunk and the
    [DONE] line do not stretch it.
    """
    chunks = tl["chunks"]
    if not chunks:
        raise RuntimeError("the stream carried no tokens")
    n = sum(c["tokens"] for c in chunks)
    ttft = chunks[0]["t"]
    e2e = chunks[-1]["t"]
    gaps = [b["t"] - a["t"] for a, b in zip(chunks, chunks[1:])]
    usage_n = tl["usage"].get("completion_tokens")
    return {
        "ttft_ms": ttft * 1000,
        "e2e_ms": e2e * 1000,
        "decode_ms": (e2e - ttft) * 1000,
        "completion_tokens": n,
        "completion_tokens_usage": usage_n,
        "prompt_tokens": tl["usage"].get("prompt_tokens"),
        "prompt_token_ids_len": tl.get("prompt_token_ids_len"),
        "tpot_ms": (e2e - ttft) * 1000 / (n - 1) if n > 1 else None,
        "itl_ms": [g * 1000 for g in gaps],
        "tokens_per_chunk": [c["tokens"] for c in chunks],
    }


def stream_once(url, body, timeout=900):
    import urllib.request

    req = urllib.request.Request(url.rstrip("/") + "/v1/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        tl = timeline_from_lines(r, t0, time.perf_counter)
    return summarize_timeline(tl)


def aggregate(runs):
    """Medians of 5 with their spread, plus mean ITL pooled over every gap of every run."""
    def ms(key):
        vals = [r[key] for r in runs if r.get(key) is not None]
        if not vals:
            return {"median": None, "spread": None, "runs": []}
        return {"median": statistics.median(vals), "spread": max(vals) - min(vals),
                "runs": vals}
    pooled = [g for r in runs for g in r["itl_ms"]]
    tpc = [t for r in runs for t in r["tokens_per_chunk"]]
    return {
        "ttft_ms": ms("ttft_ms"),
        "decode_ms": ms("decode_ms"),
        "e2e_ms": ms("e2e_ms"),
        "tpot_ms": ms("tpot_ms"),
        "mean_itl_ms": statistics.fmean(pooled) if pooled else None,
        "itl_gaps_pooled": len(pooled),
        "tokens_per_chunk": {"min": min(tpc) if tpc else None, "max": max(tpc) if tpc else None,
                             "mean": statistics.fmean(tpc) if tpc else None,
                             "chunks": len(tpc)},
        "completion_tokens": sorted({r["completion_tokens"] for r in runs}),
        "prompt_tokens": sorted({r["prompt_tokens"] for r in runs if r["prompt_tokens"]}),
    }
