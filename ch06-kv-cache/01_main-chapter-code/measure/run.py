#!/usr/bin/env python3
"""Chapter 6's measurements, one subcommand each, run on one L4 before the prose is final.

  python run.py --list            the subcommand ids, in the order to run them on the GPU
  python run.py m03               one measurement; writes results/m03.json
  python run.py all               every measurement in order; a failure does not stop the rest
  python run.py m10 --dry-run     print every command and check the config, no GPU needed

Pinned: vLLM 0.30.0, Qwen/Qwen2.5-7B-Instruct, greedy (--generation-config vllm and
temperature 0). Every timing is the median of five runs after one discarded warm-up, with
the spread (max minus min) beside it. fixtures.md maps each [FIXTURE: ...] in the draft to
the result field that fills it.
"""
import argparse
import json
import os
import pathlib
import shutil
import sys
import threading
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import engine  # noqa: E402
import prompts  # noqa: E402
import vlog  # noqa: E402
from engine import GREEDY, MODEL, yarn  # noqa: E402

BYTES_PER_TOKEN_BF16 = 57_344          # 2 x 28 layers x 4 KV heads x 128 x 2 bytes (section 6.1)
SHARED_PREFIX, SESSION_TAIL = 120_000, 8_000    # section 6.3: a 120k repository, 8k tails
MAX_SESSIONS = 16                      # m10's cap; the draft's arithmetic says 4 to 9
OFFLOAD_POOL_BYTES = int(1.5 * 32_768 * BYTES_PER_TOKEN_BF16)  # holds one 32k agent, not two
PASS_LINE = 95.0                       # percent recovery, the line section 6.6 quotes
DETAILS = ["--enable-prompt-tokens-details"]

ORDER = ["m01", "m14", "m02", "m03", "m04", "m05", "m06", "m07", "m08", "m09",
         "m10", "m11", "m12", "m13"]

TITLES = {
    "m01": "usable device memory and the run environment",
    "m02": "the bf16 start at 131,072 tokens, refused: X, Y, Z and the error's first line",
    "m03": "the bf16 start at 32,768: the carve-up, the pool, the concurrency line, block size",
    "m04": "concurrency lines in bf16 at 4,096 and 32,768",
    "m05": "concurrency lines in fp8 at 4,096, 32,768 and 131,072",
    "m06": "the attention backend line in the bf16 and fp8 starts",
    "m07": "turn one and turn two TTFT over the mini-session, prefix caching off then on",
    "m08": "prefix-cache hit tokens on turn two, and the counter names",
    "m09": "turn one and turn two TTFT over the 32k repository prefix, caching off then on",
    "m10": "sessions running over one shared 120k prefix in fp8",
    "m11": "first agent's turn two TTFT with a second agent, CPU offloading off then on",
    "m12": "TPOT, bf16 and fp8 cache, at 32k and at the longest length bf16 starts at",
    "m13": "retrieval recovery at 32k, 64k and 131,072, fp8 and bf16",
    "m14": "prompt and completion tokens per turn, from baseline.json",
}

# Planning estimates, not measurements: about 3 minutes per engine start (weights from
# local disk, profiling, graph capture), prefill at half of the L4's 121 TFLOPS bf16 peak
# (32k about 12 s, 131,072 about 85 s, by section 6.5's arithmetic), decode near 50 ms a
# token. The first start of a session adds the 15 GB download.
GPU_MINUTES = {"m01": 0.5, "m14": 0, "m02": 6, "m03": 3, "m04": 3, "m05": 9, "m06": 0,
               "m07": 7, "m08": 0, "m09": 14, "m10": 15, "m11": 19, "m12": 27, "m13": 40}


# ------------------------------------------------------------------ engine configurations

def cfg(dtype, max_len, yarn_form="v030", extra=()):
    """The draft's rule: YaRN only above the model's own 32,768 positions."""
    flags = GREEDY + ["--max-model-len", str(max_len)]
    if max_len > 32_768:
        flags += yarn(yarn_form)
    if dtype == "fp8":
        flags += ["--kv-cache-dtype", "fp8"]
    return flags + list(extra)


def startups(a):
    """The startup-only engine starts, by name. Shared across subcommands through a cache."""
    return {
        "bf16-131072-as-printed": ["--max-model-len", "131072"] + yarn("printed"),
        "bf16-131072": cfg("bf16", 131_072, a.yarn_form),
        "bf16-32768": cfg("bf16", 32_768),
        "bf16-4096": cfg("bf16", 4_096),
        "fp8-4096": cfg("fp8", 4_096),
        "fp8-32768": cfg("fp8", 32_768),
        "fp8-131072": cfg("fp8", 131_072, a.yarn_form),
    }


def caching(on):
    return GREEDY + ([] if on else ["--no-enable-prefix-caching"]) + DETAILS


def all_flag_sets(a):
    """Every flag list any subcommand can pass, for test_run.py's flag check."""
    sets = list(startups(a).values()) + [caching(True), caching(False)]
    base = GREEDY + ["--kv-cache-memory-bytes", str(OFFLOAD_POOL_BYTES)] + DETAILS
    sets += [base, base + ["--kv-offloading-size", "8"]]
    sets += [cfg(d, n, a.yarn_form, ["--kv-cache-dtype-skip-layers", "27"])
             for d in ("bf16", "fp8") for n in (32_768, 65_536, 131_072)]
    sets += [cfg("fp8", 131_072, a.yarn_form, DETAILS)]
    return sets


def show(flags):
    print("  " + " ".join(engine._quote(c) for c in engine.serve_cmd(flags)))


# ------------------------------------------------------------------ shared steps

def start_once(a, log, name):
    flags = startups(a)[name]
    if a.dry_run:
        show(flags)
        return None
    return engine.startup(name, flags, log, fresh=a.fresh)


def longest_bf16(a, log):
    """The longest context bf16 starts at: Z from m02's refusal, or 131,072 if it started."""
    m02r = engine.read_result("m02")
    if m02r and m02r.get("bf16_longest_len"):
        return m02r["bf16_longest_len"], "results/m02.json"
    if a.dry_run:
        print("  if results/m02.json is missing, first:")
    s = start_once(a, log, "bf16-131072")
    if s is None:
        return None, "m02 at run time"
    if s.get("started"):
        return 131_072, "bf16-131072 started"
    ref = (s.get("parsed") or {}).get("refusal")
    if ref and ref.get("estimated_max_model_len"):
        return ref["estimated_max_model_len"], "bf16-131072 refusal"
    return None, "bf16-131072 failed without an estimate"


def tok_and_root(a):
    tok = prompts.Tokens(MODEL, allow_estimate=a.dry_run)
    root = prompts.vllm_source_root(a.corpus_root)
    if root is None or not root.exists():
        if a.dry_run:
            print("  (no vllm source here: long prompts are built on the GPU machine; "
                  "pass --corpus-root to check them now)")
            return tok, None
        raise SystemExit("cannot find the installed vllm package's source; pass --corpus-root")
    return tok, root


def describe_prompt(name, ids, max_len, tok, answer=engine.COMPLETION_TOKENS):
    fits = len(ids) + answer <= max_len
    kind = "exact" if tok.exact else "estimated, chars/4"
    print(f"  prompt {name}: {len(ids):,} tokens ({kind}) + {answer} answer "
          f"{'fits' if fits else 'DOES NOT FIT'} max_model_len {max_len:,}")
    if tok.exact and not fits:
        raise SystemExit(f"{name} does not fit {max_len}")


def metric_delta(before, after, name):
    b, c = vlog.metric(before, name), vlog.metric(after, name)
    return None if b is None or c is None else c - b


def turn_pairs(url, p1, tok, runs):
    """Turn one, then turn two over the same prefix; one warm-up pair, then `runs` pairs.

    Each pair carries its own cache_salt, shared by its two turns. The salt folds into the
    first block's hash, so turn one never hits an earlier pair's blocks and pays a full
    prefill, while turn two can still hit its own turn one. With caching off the salt
    changes nothing.
    """
    out = []
    for i in range(runs + 1):
        salt = engine.new_salt()
        t1 = engine.completion(url, p1, salt=salt)
        p2 = prompts.turn_two(p1, t1["token_ids"], tok)
        before = engine.scrape(url)
        t2 = engine.completion(url, p2, salt=salt)
        after = engine.scrape(url)
        out.append({"turn_one_ttft_ms": t1["ttft_ms"], "turn_two_ttft_ms": t2["ttft_ms"],
                    "turn_one_prompt_tokens": t1["prompt_tokens"],
                    "turn_two_prompt_tokens": t2["prompt_tokens"],
                    "turn_two_cached_tokens_usage": t2["cached_tokens"],
                    "turn_two_prefix_cache_queries":
                        metric_delta(before, after, vlog.PREFIX_QUERIES),
                    "turn_two_prefix_cache_hits": metric_delta(before, after, vlog.PREFIX_HITS),
                    "warmup": i == 0})
    timed = [r for r in out if not r["warmup"]]
    ms = engine.median_spread
    return {"per_run": out,
            "turn_one_ttft_ms": ms([r["turn_one_ttft_ms"] for r in timed]),
            "turn_two_ttft_ms": ms([r["turn_two_ttft_ms"] for r in timed]),
            "turn_two_prefix_cache_hits": ms([r["turn_two_prefix_cache_hits"] for r in timed]),
            "turn_two_cached_tokens_usage": ms([r["turn_two_cached_tokens_usage"] for r in timed]),
            "turn_two_prompt_tokens": timed[0]["turn_two_prompt_tokens"] if timed else None}


def ablation(a, log, p1, tok, label):
    """The same turn pairs against caching off, then on, one engine start each."""
    res = {}
    for name, on in (("prefix_off", False), ("prefix_on", True)):
        flags = caching(on)
        if a.dry_run:
            show(flags)
            continue
        with engine.serving(flags, f"{label}-{name}", log) as (url, _):
            res[name] = {"server_flags": flags, **turn_pairs(url, p1, tok, a.runs)}
    return res


def pair(r):
    return [r["turn_one_ttft_ms"]["median"], r["turn_two_ttft_ms"]["median"]]


# ------------------------------------------------------------------ measurements

def m01(a, log):
    """Usable device memory, plus a preflight of everything the other subcommands need."""
    cmd = ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]
    if a.dry_run:
        print("  " + " ".join(cmd))
        return {}
    log.command(cmd)
    smi = vlog.nvidia_smi_memory_total(engine.measure._output(cmd))
    env = engine.environment()
    pk = env["packages"]
    mem = {}
    if os.path.exists("/proc/meminfo"):
        for line in open("/proc/meminfo"):
            k, v = line.split(":", 1)
            if k in ("MemTotal", "MemAvailable"):
                mem[k + "_mib"] = int(v.split()[0]) // 1024
    warnings = []
    if pk.get("vllm") != engine.VLLM_VERSION:
        warnings.append(f"vLLM is {pk.get('vllm')}, the book pins {engine.VLLM_VERSION}")
    if not pk.get("cuda_available"):
        warnings.append("torch.cuda.is_available() is False: the driver is likely older than "
                        "the CUDA build vLLM installed; see README, 'Colab risks'")
    if not smi:
        warnings.append("nvidia-smi printed no GPU")
    return {"gpus": smi, "memory_total_mib": smi[0]["memory_total_mib"] if smi else None,
            "host_memory": mem, "disk_free_gb": round(shutil.disk_usage(str(HERE)).free / 1e9, 1),
            "preflight_warnings": warnings, "environment": env}


def m02(a, log):
    """The 131,072-token bf16 start, which the draft says vLLM refuses.

    Runs the draft's printed command first, verbatim, then the v0.30.0 YaRN form. Whichever
    reaches the KV check gives X, Y and Z; a start stopped by anything else is recorded with
    its first error line, because that is what a reader typing the printed command sees.
    """
    out = {}
    for key, name in (("as_printed", "bf16-131072-as-printed"), ("v030_yarn", "bf16-131072")):
        s = start_once(a, log, name)
        if s is None:
            continue
        p = s.get("parsed") or {}
        out[key] = {"started": s.get("started"), "command": s.get("command"),
                    "refusal": p.get("refusal"), "max_len_rejection": p.get("max_len_rejection"),
                    "first_error_line": (p.get("refusal") or {}).get("error_first_line")
                    or p.get("first_error_line"),
                    "available_kv_cache_gib": p.get("available_kv_cache_gib"),
                    "capacity": p.get("capacity"), "log": s.get("log")}
    if a.dry_run:
        return {}
    ref = next((out[k]["refusal"] for k in ("as_printed", "v030_yarn")
                if out.get(k, {}).get("refusal")), None) or {}
    out.update({
        "error_first_line": ref.get("error_first_line"),
        "error_first_line_as_printed": out["as_printed"]["first_error_line"],
        "kv_cache_needed_gib": ref.get("kv_cache_needed_gib"),
        "kv_cache_available_gib": ref.get("kv_cache_available_gib"),
        "estimated_max_model_len": ref.get("estimated_max_model_len"),
        "bf16_longest_len": (131_072 if out["v030_yarn"]["started"]
                             else ref.get("estimated_max_model_len")),
        "x_against_y": (f"{ref['kv_cache_needed_gib']} GiB needed against "
                        f"{ref['kv_cache_available_gib']} GiB available" if ref else None),
        "complete": bool(ref) or bool(out["v030_yarn"]["started"]),
    })
    return out


def m03(a, log):
    """The bf16 start at 32,768: every line the waterfall in figure 6.6 needs."""
    s = start_once(a, log, "bf16-32768")
    if s is None:
        return {}
    if not s.get("started"):
        raise RuntimeError(f"bf16 at 32,768 did not start: {s['parsed'].get('first_error_line')}")
    p = s["parsed"]
    ms, ml = p.get("memory_summary") or {}, p.get("model_loading") or {}
    cap, info = p.get("capacity") or {}, s.get("cache_config_info") or {}
    est, peak = p.get("cudagraph_estimate_gib"), ms.get("peak_activation_gib")
    consumed, weights = ms.get("consumed_weights_plus_non_torch_gib"), ml.get("weights_gib")
    block = int(info["block_size"]) if str(info.get("block_size", "")).isdigit() else None
    blocks = int(info["num_gpu_blocks"]) if str(info.get("num_gpu_blocks", "")).isdigit() else None
    return {
        "log": s.get("log"),
        "available_kv_cache_gib": p.get("available_kv_cache_gib"),
        "kv_cache_size_tokens": cap.get("kv_cache_size_tokens"),
        "max_concurrency": cap.get("max_concurrency"), "capacity_line": cap.get("line"),
        "model_weights_gib": weights,
        "activation_peak_gib": (round(peak - est, 2) if peak is not None and est is not None
                                else peak),
        "activation_peak_note": "v0.30.0 logs 'peak activation' as transient headroom plus the "
                                "CUDA graph estimate; this field subtracts the estimate",
        "peak_activation_as_logged_gib": peak,
        "non_torch_gib": (round(consumed - weights, 2) if consumed is not None
                          and weights is not None else None),
        "non_torch_note": "derived: v0.30.0 logs only 'consumed memory (weights + non-torch)'",
        "cuda_graph_gib": est, "cuda_graph_note": "the estimate the pool subtraction used",
        "cuda_graph_actual_gib": (p.get("cudagraph_capture") or {}).get("gib"),
        "memory_summary": ms, "profile_debug": p.get("profile_debug"),
        "block_size_tokens": block, "num_gpu_blocks": blocks,
        "blocks_size_over_block": (cap["kv_cache_size_tokens"] // block
                                   if cap.get("kv_cache_size_tokens") and block else None),
        "block_size_source": "/metrics vllm:cache_config_info (v0.30.0 does not log it at INFO)",
        "cache_config_info": info,
    }


def _concurrency(a, log, names):
    out = {}
    for name in names:
        s = start_once(a, log, name)
        if s is None:
            continue
        p = s.get("parsed") or {}
        cap = p.get("capacity") or {}
        out[name.replace("-", "_")] = {
            "started": s.get("started"), "available_kv_cache_gib": p.get("available_kv_cache_gib"),
            "kv_cache_size_tokens": cap.get("kv_cache_size_tokens"),
            "max_concurrency": cap.get("max_concurrency"), "line": cap.get("line"),
            "first_error_line": None if s.get("started") else p.get("first_error_line"),
            "log": s.get("log")}
    return out


def m04(a, log):
    return _concurrency(a, log, ["bf16-4096", "bf16-32768"])


def m05(a, log):
    return _concurrency(a, log, ["fp8-4096", "fp8-32768", "fp8-131072"])


def m06(a, log):
    out = {}
    for key, name in (("bf16", "bf16-32768"), ("fp8", "fp8-32768"), ("fp8_131072", "fp8-131072")):
        s = start_once(a, log, name)
        if s is not None:
            out[key] = (s.get("parsed") or {}).get("backend")
    return out


def m07(a, log):
    tok, _ = tok_and_root(a)
    p1 = prompts.mini_prompt_ids(tok)
    describe_prompt("mini-session turn one", p1, 32_768, tok)
    res = ablation(a, log, p1, tok, "m07")
    if a.dry_run:
        return {}
    off, on = res["prefix_off"], res["prefix_on"]
    return {**res, "off_pair_ms": pair(off), "on_pair_ms": pair(on),
            "turn_one_spread_off_ms": off["turn_one_ttft_ms"]["spread"]}


def m08(a, log):
    """Hit tokens on turn two. Reuses m07's caching-on run when it exists."""
    m07r = engine.read_result("m07")
    src = "results/m07.json"
    if m07r and m07r.get("prefix_on"):
        on = m07r["prefix_on"]
    else:
        tok, _ = tok_and_root(a)
        p1 = prompts.mini_prompt_ids(tok)
        if a.dry_run:
            show(caching(True))
            return {}
        with engine.serving(caching(True), "m08-prefix_on", log) as (url, _):
            on = turn_pairs(url, p1, tok, a.runs)
        src = "this run"
    return {"source": src,
            "turn_two_prefix_cache_hits_tokens": on["turn_two_prefix_cache_hits"]["median"],
            "turn_two_cached_tokens_usage": on["turn_two_cached_tokens_usage"]["median"],
            "turn_two_prompt_tokens": on["turn_two_prompt_tokens"],
            "counter_names": [vlog.PREFIX_QUERIES, vlog.PREFIX_HITS],
            "counter_names_exported": [vlog.PREFIX_QUERIES + "_total", vlog.PREFIX_HITS + "_total"],
            "counter_source": "vllm/v1/metrics/loggers.py, v0.30.0"}


def m09(a, log):
    tok, root = tok_and_root(a)
    prefix = prompts.repo_prefix_tokens(tok)
    p1 = prompts.repo_turn_one(tok, root, prefix) if root else [0] * (prefix + 15)
    describe_prompt("32k repository turn one", p1, 32_768, tok)
    describe_prompt("32k repository turn two", prompts.turn_two(p1, [0] * engine.COMPLETION_TOKENS, tok),
                    32_768, tok)
    res = ablation(a, log, p1, tok, "m09")
    if a.dry_run:
        return {}
    off, on = res["prefix_off"], res["prefix_on"]
    t1, t2 = on["turn_one_ttft_ms"]["median"], on["turn_two_ttft_ms"]["median"]
    return {**res, "repo_prefix_tokens": prefix,
            "corpus": {"root": str(root), "fingerprint": prompts.fingerprint(root, ["v1"])},
            "off_pair_ms": pair(off), "on_pair_ms": pair(on),
            "turn_two_off_ms": off["turn_two_ttft_ms"]["median"], "turn_two_on_ms": t2,
            "turn_two_over_turn_one_on": round(t2 / t1, 4) if t1 else None}


def m10(a, log):
    """Add sessions over one cached 120,000-token prefix until the scheduler stops admitting."""
    tok, root = tok_and_root(a)
    max_new = 131_072 - SHARED_PREFIX - SESSION_TAIL - 72     # 3,000 decoded per session
    flags = cfg("fp8", 131_072, a.yarn_form, DETAILS)
    if root is not None:
        sessions = prompts.shared_prefix_sessions(tok, root, SHARED_PREFIX, SESSION_TAIL,
                                                  MAX_SESSIONS)
        describe_prompt("shared-prefix session", sessions[0], 131_072, tok, max_new)
    if root is None or a.dry_run:
        print(f"  {MAX_SESSIONS} sessions of {SHARED_PREFIX:,} shared + {SESSION_TAIL:,} "
              f"unique tokens, {max_new:,} decoded each")
        show(flags)
        return {}
    samples, errors = [], []
    with engine.serving(flags, "m10-shared-prefix", log) as (url, _):
        engine.completion(url, sessions[0], max_tokens=1)          # cache the shared prefix

        def go(ids):
            try:
                engine.completion(url, ids, max_tokens=max_new, timeout=3600)
            except Exception as e:  # noqa: BLE001  the engine is stopped under them
                errors.append(repr(e)[:200])

        for ids in sessions:
            threading.Thread(target=go, args=(ids,), daemon=True).start()
        t0 = time.monotonic()
        stable_since, last = t0, None
        while time.monotonic() - t0 < a.m10_seconds:
            m = engine.scrape(url)
            row = {"t_s": round(time.monotonic() - t0, 1),
                   "running": vlog.metric(m, vlog.RUNNING), "waiting": vlog.metric(m, vlog.WAITING),
                   "kv_cache_usage": vlog.metric(m, vlog.KV_USAGE),
                   "preemptions": vlog.metric(m, vlog.PREEMPTIONS),
                   "prefix_cache_hits": vlog.metric(m, vlog.PREFIX_HITS)}
            samples.append(row)
            key = (row["running"], row["waiting"])
            if key != last:
                stable_since, last = time.monotonic(), key
            settled = (row["running"] or 0) + (row["waiting"] or 0) >= MAX_SESSIONS
            if settled and time.monotonic() - stable_since > 30:
                break
            time.sleep(1)
    stalled = [r for r in samples if (r["waiting"] or 0) > 0]
    return {"sessions_running_when_admission_stalled":
            max(r["running"] or 0 for r in stalled) if stalled else None,
            "sessions_running_peak": max((r["running"] or 0) for r in samples) if samples else None,
            "admission_stalled": bool(stalled),
            "note": None if stalled else f"all {MAX_SESSIONS} sessions ran: raise MAX_SESSIONS",
            "shared_prefix_tokens": SHARED_PREFIX, "tail_tokens": SESSION_TAIL,
            "decode_tokens_per_session": max_new, "server_flags": flags,
            "corpus": {"root": str(root),
                       "fingerprint": prompts.fingerprint(root, ["v1", "entrypoints"])},
            "samples": samples, "client_errors_after_stop": len(errors)}


def m11(a, log):
    """Two agents on different 32k repositories; agent A's turn two after B has run."""
    tok, root = tok_and_root(a)
    gib = a.offload_gib
    if os.path.exists("/proc/meminfo"):
        avail = next(int(l.split()[1]) for l in open("/proc/meminfo")
                     if l.startswith("MemAvailable"))
        if avail / 2**20 < 3 * gib:
            gib = max(2, int(avail / 2**20 / 3))
            print(f"  host memory is short; CPU tier lowered to {gib} GiB")
    base = GREEDY + ["--kv-cache-memory-bytes", str(OFFLOAD_POOL_BYTES)] + DETAILS
    settings = (("offload_off", base), ("offload_on", base + ["--kv-offloading-size", f"{gib:g}"]))
    if root is not None:
        prefix = prompts.repo_prefix_tokens(tok)
        pa = prompts.repo_turn_one(tok, root, prefix)
        pb = prompts.other_repo_turn_one(tok, root, prefix)
        describe_prompt("agent A turn one", pa, 32_768, tok)
        describe_prompt("agent B turn one", pb, 32_768, tok)
    if root is None or a.dry_run:
        for _, flags in settings:
            show(flags)
        return {}
    out = {}
    for name, flags in settings:
        rows = []
        with engine.serving(flags, f"m11-{name}", log) as (url, _):
            for i in range(a.runs + 1):
                salt = engine.new_salt()
                a1 = engine.completion(url, pa, salt=salt)
                engine.completion(url, pb, salt=salt)                # agent B, between turns
                before = engine.scrape(url)
                a2 = engine.completion(url, prompts.turn_two(pa, a1["token_ids"], tok), salt=salt)
                after = engine.scrape(url)
                rows.append({"warmup": i == 0, "a_turn_two_ttft_ms": a2["ttft_ms"],
                             "a_turn_one_ttft_ms": a1["ttft_ms"],
                             "gpu_prefix_hits": metric_delta(before, after, vlog.PREFIX_HITS),
                             "cpu_tier_hits": metric_delta(before, after, vlog.EXTERNAL_HITS),
                             "cached_tokens_usage": a2["cached_tokens"]})
        timed = [r for r in rows if not r["warmup"]]
        ms = engine.median_spread
        out[name] = {"server_flags": flags, "per_run": rows,
                     "a_turn_two_ttft_ms": ms([r["a_turn_two_ttft_ms"] for r in timed]),
                     "gpu_prefix_hits": ms([r["gpu_prefix_hits"] for r in timed]),
                     "cpu_tier_hits": ms([r["cpu_tier_hits"] for r in timed])}
    out["pair_ms"] = [out["offload_off"]["a_turn_two_ttft_ms"]["median"],
                      out["offload_on"]["a_turn_two_ttft_ms"]["median"]]
    out["pool_bytes"] = OFFLOAD_POOL_BYTES
    out["pool_note"] = ("--kv-cache-memory-bytes caps the pool at 1.5 x 32,768 bf16 tokens so "
                        "agent B's prefill evicts agent A, as at 128k fp8 in section 6.5")
    out["offload_gib"] = gib
    return out


def _timed_at(a, log, dtype, max_len, tok, root):
    """Start at `max_len`, stepping down 1,024 tokens up to three times if the start fails."""
    for step in range(4):
        n = max_len - 1024 * step
        try:
            with engine.serving(cfg(dtype, n, a.yarn_form), f"m12-{dtype}-{n}", log) as (url, _):
                ids = prompts.filler(tok, root, n - engine.COMPLETION_TOKENS - 8)
                runs = [engine.completion(url, ids, salt=engine.new_salt())
                        for _ in range(a.runs + 1)][1:]
            return {"max_model_len": n, "prompt_tokens": len(ids),
                    "tpot_ms": engine.median_spread([r["tpot_ms"] for r in runs]),
                    "per_run_tpot_ms": [r["tpot_ms"] for r in runs]}
        except engine.StartFailed as e:
            print(f"  {dtype} at {n} did not start: {e}")
    return {"did_not_start": f"{dtype} between {max_len - 3072} and {max_len}"}


def m12(a, log):
    """TPOT at a batch of one, bf16 against fp8 cache, at 32k and at bf16's longest length."""
    tok, root = tok_and_root(a)
    lb, lb_src = longest_bf16(a, log)
    if a.dry_run or root is None:
        for d in ("bf16", "fp8"):
            show(cfg(d, 32_768))
            show(cfg(d, lb or 131_072, a.yarn_form))
        print(f"  longest bf16 length: {lb or 'Z from m02, at run time'}")
        return {}
    if not lb:
        raise RuntimeError(f"no longest bf16 length ({lb_src})")
    out = {"longest_bf16_len": lb, "longest_bf16_len_source": lb_src}
    for label, n in (("32k", 32_768), ("longest_bf16", lb)):
        for d in ("bf16", "fp8"):
            out[f"{d}_{label}"] = _timed_at(a, log, d, n, tok, root)
    return out


def m13(a, log):
    """Plant a fact, ask for it back; recovery = found / trials, against the 95 percent line."""
    tok, root = tok_and_root(a)
    lb, lb_src = longest_bf16(a, log)
    skip = ["--kv-cache-dtype-skip-layers"] + a.skip_layers.split(",") if a.skip_layers else []
    groups = [("fp8", 32_768, (32_768,), []), ("fp8", 131_072, (65_536, 131_072), []),
              ("bf16", 32_768, (32_768,), [])]
    if lb and lb >= 65_536:
        groups.append(("bf16", min(lb, 131_072),
                       (65_536,) + ((131_072,) if lb >= 131_072 else ()), []))
    if skip:
        groups += [("fp8", 32_768, (32_768,), skip), ("fp8", 131_072, (65_536, 131_072), skip)]
    if root is not None:
        for L in (32_768, 65_536, 131_072):
            t = prompts.needle_trials(tok, root, L, a.trials)
            sizes = {x["prompt_tokens"] for x in t}
            describe_prompt(f"needle at {L:,} ({len(t)} trials, sizes {sorted(sizes)})",
                            t[0]["prompt"], L, tok, prompts.NEEDLE_ANSWER_TOKENS)
    if a.dry_run or root is None:
        for d, n, lens, ex in groups:
            show(cfg(d, n, a.yarn_form, ex))
            print(f"    lengths {', '.join(f'{x:,}' for x in lens)}")
        print(f"  {a.trials} trials per length; bf16 at 65,536 runs only if Z >= 65,536 "
              f"(Z: {lb or 'from m02 at run time'})")
        return {}
    out = {"trials_per_length": a.trials, "pass_line_percent": PASS_LINE,
           "longest_bf16_len": lb, "longest_bf16_len_source": lb_src, "recovery": {},
           "corpus": {"root": str(root),
                      "fingerprint": prompts.fingerprint(root, ["model_executor"])}}
    for d, n, lens, ex in groups:
        key = d + ("_skip_layers" if ex else "")
        flags = cfg(d, n, a.yarn_form, ex)
        try:
            with engine.serving(flags, f"m13-{key}-{n}", log) as (url, _):
                for L in lens:
                    rows = []
                    for t in prompts.needle_trials(tok, root, L, a.trials):
                        r = engine.completion(url, t["prompt"], ignore_eos=False,
                                              max_tokens=prompts.NEEDLE_ANSWER_TOKENS)
                        rows.append({"depth": t["depth"], "expected": t["value"],
                                     "answer": r["text"][:80],
                                     "found": prompts.recovered(r["text"], t["value"])})
                    pct = round(100 * sum(r["found"] for r in rows) / len(rows), 1)
                    out["recovery"].setdefault(key, {})[str(L)] = {
                        "percent": pct, "passes": pct >= PASS_LINE, "server_flags": flags,
                        "trials": rows}
        except engine.StartFailed as e:
            for L in lens:
                out["recovery"].setdefault(key, {})[str(L)] = {"did_not_start": str(e)}
    bf = out["recovery"].setdefault("bf16", {})
    if not (lb and lb >= 65_536):
        bf.setdefault("65536", {"skipped": f"bf16's longest length is {lb}"})
    bf.setdefault("131072", {"skipped": "bf16 does not start at 131,072 (m02)"})
    return out


def m14(a, log):
    """Tokens per turn from baseline.json, beside the model tokenizer's and engine's counts."""
    base = json.loads((prompts.MINI / "baseline.json").read_text(encoding="utf-8"))
    t1 = base["per_turn"][0]
    out = {"baseline_prompt_tokens_turn_one": t1["prompt_tokens"],
           "baseline_completion_tokens": t1["completion_tokens"],
           "tokens_per_session": t1["prompt_tokens"] + t1["completion_tokens"],
           "baseline_counting": "measure.py counts characters over four",
           "per_turn": base["per_turn"]}
    if a.dry_run:
        print(f"  baseline.json: {out['tokens_per_session']} tokens per session")
        return out
    tok = prompts.Tokens(MODEL, allow_estimate=True)
    out["qwen_tokenizer_prompt_tokens"] = len(prompts.mini_prompt_ids(tok)) if tok.exact else None
    m07r = engine.read_result("m07")
    if m07r and m07r.get("prefix_off"):
        out["engine_prompt_tokens_turn_one"] = (
            m07r["prefix_off"]["per_run"][0]["turn_one_prompt_tokens"])
    return out


FUNCS = {k: globals()[k] for k in ORDER}


# ------------------------------------------------------------------ driver

def run_one(sub, a):
    print(f"== {sub}: {TITLES[sub]} (planning estimate {GPU_MINUTES[sub]} GPU-minutes)")
    if a.skip_existing and not a.dry_run:
        old = engine.read_result(sub)
        if old and old.get("status") == "ok":
            print(f"   results/{sub}.json exists; skipped (--skip-existing)")
            return 0
    log = engine.RunLog(sub)
    try:
        fields = FUNCS[sub](a, log)
        if a.dry_run:
            print("   dry run: config valid")
            return 0
        status = "incomplete" if fields.get("complete") is False else "ok"
        result = {"id": sub, "title": TITLES[sub], "status": status, **fields,
                  "run_log": log.finish()}
        code = 0 if status == "ok" else 1
    except (Exception, SystemExit) as e:  # noqa: BLE001  one failure must not stop `all`
        if a.dry_run:
            print(f"   dry run FAILED: {e}")
            return 1
        result = {"id": sub, "title": TITLES[sub], "status": "error", "error": repr(e),
                  "traceback": traceback.format_exc()[-4000:], "run_log": log.finish()}
        code = 1
    path = engine.write_result(sub, result)
    print(f"   {result['status']}: {path.relative_to(HERE)}")
    return code


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sub", nargs="?", help="a measurement id, or 'all'")
    ap.add_argument("--list", action="store_true", help="print the ids in GPU order and exit")
    ap.add_argument("--dry-run", action="store_true", help="print commands, check config, no GPU")
    ap.add_argument("--runs", type=int, default=5, help="timed runs per setting (median, spread)")
    ap.add_argument("--fresh", action="store_true", help="ignore cached startup logs")
    ap.add_argument("--skip-existing", action="store_true", help="skip ids with an ok result")
    ap.add_argument("--yarn-form", choices=sorted(engine.YARN_FORM), default="v030",
                    help="the --hf-overrides form for starts above 32,768 (m02 runs both)")
    ap.add_argument("--corpus-root", help="the vllm package source dir, if not importable")
    ap.add_argument("--offload-gib", type=float, default=8, help="m11's CPU tier, GiB")
    ap.add_argument("--trials", type=int, default=20, help="m13's needles per length")
    ap.add_argument("--skip-layers", default="", help="m13: layers kept in bf16, e.g. 27")
    ap.add_argument("--m10-seconds", type=int, default=900, help="m10's time cap")
    a = ap.parse_args(argv)
    if a.list:
        print("\n".join(ORDER))
        return 0
    if not a.sub:
        ap.print_help()
        return 2
    subs = ORDER if a.sub == "all" else [a.sub]
    for s in subs:
        if s not in FUNCS:
            ap.error(f"unknown id {s}; ids: {' '.join(ORDER)}")
    codes = [run_one(s, a) for s in subs]
    if a.dry_run and a.sub == "all":
        print(f"\nplanning estimate: {sum(GPU_MINUTES.values())} GPU-minutes in total")
    return 1 if any(codes) else 0


if __name__ == "__main__":
    sys.exit(main())
