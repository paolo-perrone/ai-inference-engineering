#!/usr/bin/env python3
"""Chapter 5's measurements on one L4, one subcommand per measurement.

  python run.py --list            the measurement ids, in the order to run them on the GPU
  python run.py m05               one measurement; writes results/m05.json
  python run.py m05 --dry-run     print every command and validate the config, no GPU
  python run.py all               every measurement in --list order; one failure never
                                  stops the next

Every timed cell is the median of five runs with its spread. Every result file carries a
run log: GPU, driver, CUDA, vLLM and PyTorch versions, the exact commands and flags, and
the checkpoint ids and revisions. A server that refuses to start, or runs out of memory,
is recorded as a result, never raised. fixtures.md maps every [FIXTURE: ...] placeholder
in the chapter draft to the subcommand and field that fills it.
"""
import argparse
import concurrent.futures
import json
import pathlib
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common as C  # noqa: E402
import turns as T   # noqa: E402

DEFAULT_TURNS = C.RECORDINGS / "turns.jsonl"
PREFILL_LENGTHS = {"8k": 8192, "32k": 32768,
                   # 131072 + 64 output tokens would exceed --max-model-len 131072 and vLLM
                   # rejects the request, so the 128k point leaves room for the 64.
                   "128k": C.MAX_MODEL_LEN - 64}
DECODE_CONCURRENCY = [1, 4, 8, 16]
SPEEDUP_CONCURRENCY = [1, 2, 4, 8, 16]
SPEC = {"off": None,
        "ngram": {"method": "ngram", "num_speculative_tokens": 5},
        "suffix": {"method": "suffix", "num_speculative_tokens": 32}}


def spec_flags(name):
    return [] if SPEC[name] is None else ["--speculative-config", json.dumps(SPEC[name])]


TOOL_FLAGS = ["--enable-auto-tool-choice", "--tool-call-parser", "hermes"]


class Blocked(Exception):
    """A prerequisite is missing; the result file says which."""


def live(ctx, srv):
    return srv.ok and not ctx.dry


def model_name(key):
    return C.MODELS[key]["id"]


# ---------------------------------------------------------------- startup probes

def startup_probe(ctx, prec):
    """Start one precision's server, read its startup lines, stop it.

    Cached under results/raw by the exact command, so m01, m02 and m04 share one start per
    precision when they run in the same session.
    """
    key, flags = C.PRECISIONS[prec]
    cache = C.RAW / f"startup-{prec}-{C.digest([key, flags, C.MODELS[key]['revision']])}.json"
    if cache.exists() and not ctx.dry:
        rec = json.loads(cache.read_text())
        ctx.commands.append(rec["server"]["command"] + "   (startup log reused)")
        ctx.models.add(key)
        return rec
    with C.Server(ctx, key, flags, f"startup-{prec}") as srv:
        rec = {"server": srv.record(), "startup": srv.startup() if live(ctx, srv) else None}
        if not ctx.dry and srv.proc is not None and not srv.ok:
            rec["startup"] = srv.startup()           # a failed start still logs what it got to
    if not ctx.dry:
        cache.write_text(json.dumps(rec, indent=2))
    return rec


# ---------------------------------------------------------------- m00 preflight

PREFLIGHT_CODE = r"""
import json, sys
from vllm.engine.arg_utils import EngineArgs
out = []
for name, model, rev, mml, ov in json.loads(sys.argv[1]):
    try:
        cfg = EngineArgs(model=model, revision=rev, tokenizer_revision=rev,
                         max_model_len=mml, hf_overrides=ov).create_model_config()
        out.append({"case": name, "ok": True, "max_model_len": cfg.max_model_len})
    except Exception as e:
        out.append({"case": name, "ok": False, "error": f"{type(e).__name__}: {e}"[:600]})
print(json.dumps(out))
"""


def m00(ctx, a):
    """Preflight: versions, disk, checkpoints reachable, and how vLLM resolves each 128k config."""
    cases = [("bf16 with YARN (this folder)", "qwen7b", C.MAX_MODEL_LEN, C.YARN),
             ("bf16 with the writing plan's rope_scaling", "qwen7b", C.MAX_MODEL_LEN, C.YARN_PLAN),
             ("bf16 with no override", "qwen7b", C.MAX_MODEL_LEN, None),
             ("awq with YARN", "qwen7b-awq", C.MAX_MODEL_LEN, C.YARN),
             ("coder1.5b with YARN", "coder1.5b", C.MAX_MODEL_LEN, C.YARN),
             ("coder7b at 32768", "coder7b", 32768, None),
             ("swedev7b at 32768", "swedev7b", 32768, None)]
    payload = [[n, C.MODELS[k]["id"], C.MODELS[k]["revision"], m, o] for n, k, m, o in cases]
    for k in {c[1] for c in cases}:
        ctx.models.add(k)
    if ctx.dry:
        print("  (preflight builds vLLM's ModelConfig on the CPU for each case:)")
        for c in cases:
            print(f"    {c[0]}: max_model_len={c[2]} hf_overrides={json.dumps(c[3])}")
        return {"status": "dry-run"}
    disk = shutil.disk_usage(pathlib.Path.home())
    py = C.harness._interpreter(shutil.which("vllm")) if shutil.which("vllm") else None
    resolved = None
    if py:
        p = subprocess.run([py, "-c", PREFLIGHT_CODE, json.dumps(payload)], capture_output=True,
                           text=True, timeout=900, env=C.server_env())
        try:
            resolved = json.loads(p.stdout.strip().splitlines()[-1])
        except (IndexError, ValueError):
            resolved = {"error": (p.stderr or p.stdout)[-2000:]}
    need = sum(m["size_gb"] for m in C.MODELS.values())
    return {"status": "ok" if py else "failed",
            "disk_free_gb": round(disk.free / 1e9, 1), "download_needed_gb": round(need, 2),
            "disk_ok": disk.free / 1e9 > need + 20,
            "config_resolution": resolved,
            "note": "the 128k servers need the YARN case to resolve to 131072"}


# ---------------------------------------------------------------- m01, m02, m04

def cutlass_check(ctx):
    """Does this vLLM build put FP8 on CUTLASS on this card? Read from the build itself.

    csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_entry.cu:145-154: on SM89 the
    CUTLASS FP8 kernels need CUDA_VERSION >= 12040.
    """
    if ctx.dry or not shutil.which("vllm"):
        return None
    code = ("import json,torch\nfrom vllm import _custom_ops as ops\n"
            "c=torch.cuda.get_device_capability(); cap=c[0]*10+c[1]\n"
            "print(json.dumps({'capability':cap,'torch_cuda':torch.version.cuda,"
            "'cutlass_scaled_mm_supports_fp8':bool(ops.cutlass_scaled_mm_supports_fp8(cap))}))")
    p = subprocess.run([C.harness._interpreter(shutil.which("vllm")), "-c", code],
                       capture_output=True, text=True, timeout=300)
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return {"error": (p.stderr or p.stdout)[-1000:]}


def m01(ctx, a):
    """Kernel selected per precision: the two startup lines, and the CUDA 12.4 check."""
    out = {"status": "ok"}
    for prec in ("awq", "fp8"):
        rec = startup_probe(ctx, prec)
        lines = (rec.get("startup") or {}).get("kernels") or []
        out[prec] = {"server": rec["server"], "kernel_lines": [k["line"] for k in lines],
                     "kernels": sorted({k["kernel"] for k in lines})}
        if not ctx.dry and not lines:
            out["status"] = "partial"
    out["build"] = cutlass_check(ctx)
    if not ctx.dry:
        fp8 = out["fp8"]["kernels"]
        out["fp8_on_cutlass"] = "CutlassFP8ScaledMMLinearKernel" in fp8
        out["fp8_on_torch_fallback"] = any("Torch" in k for k in fp8)
        out["awq_on_marlin"] = "MarlinLinearKernel" in out["awq"]["kernels"]
    return out


def m02(ctx, a):
    """Weights loaded per precision: vLLM's model-loading line, GiB and GB."""
    out = {"status": "ok"}
    for prec in C.PRECISIONS:
        rec = startup_probe(ctx, prec)
        s = rec.get("startup") or {}
        gib = s.get("model_loading_gib")
        out[prec] = {"server": rec["server"], "weights_loaded_gib": gib,
                     "weights_loaded_gb": None if gib is None else round(gib * C.GIB / 1e9, 3),
                     "load_seconds": s.get("model_loading_s")}
        if gib is None and not ctx.dry:
            out["status"] = "partial"
    return out


def m04(ctx, a):
    """128k sessions that fit, the cache size line, and the activation + CUDA graph reserve."""
    out = {"status": "ok"}
    for prec in C.PRECISIONS:
        rec = startup_probe(ctx, prec)
        s = rec.get("startup") or {}
        act, graph = s.get("peak_activation_gib"), s.get("cudagraph_gib")
        out[prec] = {"server": rec["server"], "kv_cache_line": s.get("kv_cache_line"),
                     "kv_cache_tokens": s.get("kv_cache_tokens"),
                     "sessions_128k": s.get("max_concurrency"),
                     "available_kv_cache_gib": s.get("available_kv_cache_gib"),
                     "peak_activation_gib": act, "cudagraph_gib": graph,
                     "reserve_gib": None if act is None or graph is None else round(act + graph, 2),
                     "graph_capture_gib": s.get("graph_capture_gib")}
        if s.get("kv_cache_line") is None and not ctx.dry:
            out["status"] = "partial"
    return out


# ---------------------------------------------------------------- m03

def timed_mini(ctx, srv, model):
    """One warm-up and five timed mini-session requests at batch 1 (the chapter 2 harness)."""
    if not live(ctx, srv):
        return None
    prompt = C.harness.build_prompt()
    C.harness.one_timed_request(srv.url, model, prompt)
    runs = [C.harness.one_timed_request(srv.url, model, prompt) for _ in range(C.RUNS)]
    return {"tpot_ms": C.stat5([r["decode_step_ms"] for r in runs]),
            "ttft_s": C.stat5([r["ttft_s"] for r in runs]),
            "engine_prompt_tokens": runs[0]["engine_prompt_tokens"]}


def m03(ctx, a):
    """Bytes read per decode step per precision, and the bandwidth they imply at batch 1.

    Bytes per step = weights loaded (vLLM's line) minus the embedding table, which a decode
    step never reads whole. The table is vocab_size x hidden_size x 2 bytes from config.json;
    it stays bf16 in every precision here, as section 5.1 says.
    """
    out = {"status": "ok"}
    for prec, (key, flags) in C.PRECISIONS.items():
        m = C.MODELS[key]
        embed = m["vocab_size"] * m["hidden_size"] * 2
        with C.Server(ctx, key, flags, f"m03-{prec}") as srv:
            t = timed_mini(ctx, srv, model_name(key))
            s = srv.startup() if live(ctx, srv) else {}
            row = {"server": srv.record(), "embedding_bytes": embed}
        if t and s.get("model_loading_gib"):
            step_bytes = s["model_loading_gib"] * C.GIB - embed
            row.update(weights_loaded_gib=s["model_loading_gib"],
                       bytes_per_step_gb=round(step_bytes / 1e9, 3),
                       tpot_b1_ms=t["tpot_ms"],
                       achieved_gb_per_s=C.stat5([step_bytes / (x / 1000) / 1e9
                                                  for x in t["tpot_ms"]["runs"]]))
        elif not ctx.dry:
            out["status"] = "partial"
        out[prec] = row
    return out


# ---------------------------------------------------------------- m05, m06

def m05(ctx, a):
    """Prefill time at 8k, 32k and 128k per precision: TTFT at concurrency 1, output 64."""
    out = {"status": "ok"}
    for prec, (key, flags) in C.PRECISIONS.items():
        mid = model_name(key)
        row = {}
        with C.Server(ctx, key, flags, f"m05-{prec}") as srv:
            row["server"] = srv.record()
            for label, n in PREFILL_LENGTHS.items():
                url = srv.url or "http://127.0.0.1:8000"
                warm = C.bench_cmd(url, mid, mid, n, 64, 1, 1, 0,
                                   C.RAW / f"m05-{prec}-{label}-warmup.json")
                cmd = C.bench_cmd(url, mid, mid, n, 64, 1, C.RUNS, 1,
                                  C.RAW / f"m05-{prec}-{label}.json")
                if ctx.dry:
                    C.run_bench(ctx, warm), C.run_bench(ctx, cmd)
                    continue
                if not srv.ok:
                    row[label] = {"error": "server did not start"}
                    continue
                C.run_bench(ctx, warm)
                r = C.run_bench(ctx, cmd)
                if not r or "error" in r or r["errors"]:
                    row[label] = {"error": (r or {}).get("error") or r["errors"][:3]}
                    out["status"] = "partial"
                    continue
                row[label] = {"prefill_s": C.stat5(r["ttfts_s"]), "input_lens": r["input_lens"],
                              "completed": r["completed"]}
        out[prec] = row
    return out


def fit_long_messages(url, model, target, seed=0, keys=0):
    """Haystack messages whose chat-templated length is at most `target` tokens."""
    probe, _ = T.needle_prompt(2000, seed, keys or 1)
    per_line = C.post(url, "/tokenize", {"model": model, "messages": probe})["count"] / 2000
    n = int(target / per_line)
    for _ in range(20):
        msgs, expected = T.needle_prompt(n, seed, keys or 1)
        count = C.post(url, "/tokenize", {"model": model, "messages": msgs})["count"]
        if count <= target:
            return msgs, expected, count
        n = int(n * target / count) - 50
    raise RuntimeError("could not fit the haystack under the target length")


def warm_128k_tpot(ctx, srv, model):
    """TPOT at batch 1 on a warm 128k session: one cold pass fills the prefix cache, then
    five requests that share it, each with a short distinct tail, 180 tokens each."""
    if not live(ctx, srv):
        return None
    msgs, _, count = fit_long_messages(srv.url, model, C.MAX_MODEL_LEN - 400)
    base = msgs[0]["content"]
    body = {"model": model, "max_tokens": 180, "ignore_eos": True, "temperature": 0,
            "cache_salt": "warm128k"}
    C.stream_timed(srv.url, "/v1/chat/completions",
                   dict(body, messages=[{"role": "user", "content": base + "\nrun 0"}]))
    runs = [C.stream_timed(srv.url, "/v1/chat/completions",
                           dict(body, messages=[{"role": "user", "content": base + f"\nrun {i}"}]))
            for i in range(1, C.RUNS + 1)]
    return {"prompt_tokens": count, "tpot_ms": C.stat5([r["tpot_ms"] for r in runs]),
            "ttft_s": C.stat5([r["ttft_s"] for r in runs])}


def m06(ctx, a):
    """Decode TPOT at batch 1, 4, 8 and 16 per precision on the mini-session request shape,
    plus one point on a warm 128k session."""
    out = {"status": "ok"}
    for prec, (key, flags) in C.PRECISIONS.items():
        mid = model_name(key)
        row = {}
        # Prefix caching on for the warm 128k point. The sweep's random prompts change
        # with every seed, so they never hit the cache.
        with C.Server(ctx, key, flags, f"m06-{prec}", prefix_caching=True) as srv:
            row["server"] = srv.record()
            url = srv.url or "http://127.0.0.1:8000"
            for c in DECODE_CONCURRENCY:
                warm = C.bench_cmd(url, mid, mid, C.MINI_PROMPT_TOKENS, C.MINI_COMPLETION_TOKENS,
                                   c, c, 100, C.RAW / f"m06-{prec}-c{c}-warmup.json")
                cmds = [C.bench_cmd(url, mid, mid, C.MINI_PROMPT_TOKENS,
                                    C.MINI_COMPLETION_TOKENS, c, c, 100 + i,
                                    C.RAW / f"m06-{prec}-c{c}-run{i}.json")
                        for i in range(1, C.RUNS + 1)]
                if ctx.dry:
                    for cmd in [warm] + cmds:
                        C.run_bench(ctx, cmd)
                    continue
                if not srv.ok:
                    row[f"c{c}"] = {"error": "server did not start"}
                    continue
                C.run_bench(ctx, warm)
                rs = [C.run_bench(ctx, cmd) for cmd in cmds]
                bad = [r for r in rs if not r or "error" in r or r["errors"]]
                if bad:
                    row[f"c{c}"] = {"error": str(bad[0])[:1000]}
                    out["status"] = "partial"
                    continue
                tpot = C.stat5([r["median_tpot_ms"] for r in rs])
                row[f"c{c}"] = {"tpot_ms": tpot,
                                "per_sequence_tok_s": round(1000 / tpot["median"], 2),
                                "output_throughput_tok_s": C.stat5([r["output_throughput"]
                                                                    for r in rs])}
            try:
                row["warm_128k_b1"] = warm_128k_tpot(ctx, srv, mid)
            except Exception as e:                          # an OOM here is a result
                row["warm_128k_b1"] = {"error": f"{type(e).__name__}: {e}"[:1000]}
                out["status"] = "partial"
        out[prec] = row
    return out


# ---------------------------------------------------------------- recorded turns

def load_recording(ctx, a, need_tools=False):
    path = pathlib.Path(a.turns)
    if not path.exists():
        if ctx.dry:
            print(f"  (no recording at {path}; a GPU run of this measurement would stop here."
                  " Record one with record_turns.py first.)")
            return None
        raise Blocked(f"no recording at {path}: record the agent's turns with record_turns.py")
    turns = T.load_turns(path)
    if need_tools:
        turns = [t for t in turns if t["request"].get("tools")]
        if not turns:
            raise Blocked("the recording has no turns with tools")
    if a.max_turns:
        turns = turns[:a.max_turns]
    return turns


def replay_body(turn, model, greedy, max_tokens_cap=4096):
    body = {k: v for k, v in turn["request"].items()
            if k not in ("stream", "stream_options", "n", "model")}
    body["model"] = model
    used = (turn.get("response") or {}).get("usage", {}).get("completion_tokens")
    body["max_tokens"] = body.get("max_tokens") or body.get("max_completion_tokens") or (
        min(max_tokens_cap, 2 * used) if used else max_tokens_cap)
    body.pop("max_completion_tokens", None)
    if greedy:
        for k in ("top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty",
                  "repetition_penalty"):
            body.pop(k, None)
        body.update(temperature=0, seed=0)
    return body


def replay_file(prec, turns, kind):
    return C.RAW / f"replay-{kind}-{prec}-{C.digest([t['turn_id'] for t in turns])}.jsonl"


def ensure_replays(ctx, a, turns, with_kl):
    """Greedy replays of every turn per precision, and (with_kl) the teacher-forced pass.

    bf16 runs first: its tokens are the twin every other precision is scored against and
    the text the teacher-forced pass feeds back.
    """
    status = {}
    ref_tokens = None
    for prec in C.PRECISIONS:
        key, flags = C.PRECISIONS[prec]
        gen_path, kl_path = replay_file(prec, turns, "gen"), replay_file(prec, turns, "kl")
        need_gen = ctx.dry or not gen_path.exists()
        need_kl = with_kl and (ctx.dry or not kl_path.exists())
        if not (need_gen or need_kl):
            status[prec] = "cached"
        else:
            with C.Server(ctx, key, flags + TOOL_FLAGS, f"replay-{prec}",
                          prefix_caching=True) as srv:
                status[prec] = srv.record()
                if live(ctx, srv):
                    mid = model_name(key)
                    if need_gen:
                        with open(gen_path, "w", encoding="utf-8") as f:
                            for t in turns:
                                body = dict(replay_body(t, mid, greedy=True), return_token_ids=True)
                                try:
                                    r = C.post(srv.url, "/v1/chat/completions", body)
                                    ch = r["choices"][0]
                                    msg = ch.get("message") or {}
                                    rec = {"turn_id": t["turn_id"], "text": msg.get("content") or "",
                                           "tool_calls": [{"name": c["function"]["name"],
                                                           "arguments": c["function"]["arguments"]}
                                                          for c in msg.get("tool_calls") or []],
                                           "token_ids": ch.get("token_ids") or [],
                                           "prompt_token_ids": r.get("prompt_token_ids") or [],
                                           "finish_reason": ch.get("finish_reason")}
                                except Exception as e:     # one turn's OOM or timeout is data
                                    rec = {"turn_id": t["turn_id"], "error": f"{type(e).__name__}: {e}"[:500]}
                                f.write(json.dumps(rec) + "\n")
                    if need_kl:
                        if ref_tokens is None:
                            ref_tokens = {r["turn_id"]: r for r in _read_jsonl(replay_file("bf16", turns, "gen"))}
                        with open(kl_path, "w", encoding="utf-8") as f:
                            for t in turns:
                                ref = ref_tokens.get(t["turn_id"]) or {}
                                out_ids = ref.get("token_ids") or []
                                if not out_ids or not ref.get("prompt_token_ids"):
                                    continue
                                body = {"model": mid, "prompt": ref["prompt_token_ids"] + out_ids,
                                        "max_tokens": 1, "temperature": 0,
                                        "prompt_logprobs": a.kl_topk}
                                try:
                                    r = C.post(srv.url, "/v1/completions", body)
                                    pl = r["choices"][0].get("prompt_logprobs") or []
                                    rec = {"turn_id": t["turn_id"], "positions": pl[-len(out_ids):]}
                                except Exception as e:
                                    rec = {"turn_id": t["turn_id"], "error": f"{type(e).__name__}: {e}"[:500]}
                                f.write(json.dumps(rec) + "\n")
    return status


def _read_jsonl(path):
    if not pathlib.Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def run_check(turn, resp):
    """Run a turn's recorded outcome check against one replayed answer. True means pass."""
    if not (turn.get("check") and turn.get("repo") and turn.get("commit")):
        return None
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["git", "clone", "--quiet", turn["repo"], d], check=True, timeout=600)
        subprocess.run(["git", "-C", d, "checkout", "--quiet", turn["commit"]], check=True,
                       timeout=120)
        calls = pathlib.Path(d) / ".tool_calls.json"
        calls.write_text(json.dumps(resp.get("tool_calls") or []))
        p = subprocess.run(turn["check"], shell=True, cwd=d, timeout=1800,
                           env={**C.server_env(), "TOOL_CALLS_JSON": str(calls),
                                "RESPONSE_TEXT": resp.get("text") or ""})
        return p.returncode == 0


def flips_for(turns, ref, cand, long_from=65536):
    n = flips = 0
    long_n = long_flips = 0
    checked = r2w = w2r = 0
    for t in turns:
        r, c = ref.get(t["turn_id"]), cand.get(t["turn_id"])
        if not r or not c or "error" in r or "error" in c:
            continue
        n += 1
        changed = T.action(r) != T.action(c)
        flips += changed
        if len(r.get("prompt_token_ids") or []) >= long_from:
            long_n += 1
            long_flips += changed
        pr, pc = run_check(t, r), run_check(t, c)
        if pr is not None and pc is not None:
            checked += 1
            r2w += pr and not pc
            w2r += pc and not pr
    out = {"turns": n, "flips": flips, "flip_rate": round(flips / n, 4) if n else None,
           "ci95": C.wilson(flips, n), "outcome_definition": "action (tool calls or final text)",
           "long_context_turns": long_n, "long_context_flips": long_flips,
           "long_context_ci95": C.wilson(long_flips, long_n)}
    if checked:
        out.update(checked_turns=checked, outcome_flips=r2w + w2r, right_to_wrong=r2w,
                   wrong_to_right=w2r, outcome_ci95=C.wilson(r2w + w2r, checked))
    return out


def m07(ctx, a):
    """Flips per quantized model on the agent's recorded turns, against bf16 twins."""
    turns = load_recording(ctx, a)
    if turns is None:
        ensure_replays(ctx, a, [{"turn_id": "dry"}], with_kl=False)
        return {"status": "dry-run"}
    servers = ensure_replays(ctx, a, turns, with_kl=False)
    rows = {p: {r["turn_id"]: r for r in _read_jsonl(replay_file(p, turns, "gen"))}
            for p in C.PRECISIONS}
    out = {"status": "ok", "servers": servers, "recording": str(a.turns),
           "replay": "greedy (temperature 0), prefix caching on, one request at a time"}
    for cand in ("fp8", "awq"):
        out[cand] = flips_for(turns, rows["bf16"], rows[cand])
    return out


def m08(ctx, a):
    """Mean per-token KL (teacher-forced on bf16's tokens) and tool-call validity."""
    turns = load_recording(ctx, a)
    if turns is None:
        ensure_replays(ctx, a, [{"turn_id": "dry"}], with_kl=True)
        return {"status": "dry-run"}
    servers = ensure_replays(ctx, a, turns, with_kl=True)
    by = {t["turn_id"]: t for t in turns}
    out = {"status": "ok", "servers": servers, "kl_topk": a.kl_topk,
           "kl_definition": ("KL(bf16 || candidate) per output position, summed over bf16's "
                             "top-k prompt logprobs; a token missing from the candidate's "
                             "top-k takes the candidate's smallest returned logprob")}
    ref_kl = {r["turn_id"]: r for r in _read_jsonl(replay_file("bf16", turns, "kl"))}
    for prec in C.PRECISIONS:
        gen = _read_jsonl(replay_file(prec, turns, "gen"))
        att = val = 0
        fails = []
        for r in gen:
            if "error" in r:
                continue
            v = T.validity(r, by[r["turn_id"]]["request"])
            att += v["attempts"]
            val += v["valid"]
            fails += v["failures"]
        row = {"tool_call_attempts": att, "valid": val,
               "validity_rate": round(val / att, 4) if att else None,
               "validity_ci95": C.wilson(val, att), "failure_examples": fails[:10]}
        if prec != "bf16":
            cand = {r["turn_id"]: r for r in _read_jsonl(replay_file(prec, turns, "kl"))}
            total, count = 0.0, 0
            for tid, rr in ref_kl.items():
                cc = cand.get(tid)
                if not cc or "error" in rr or "error" in cc:
                    continue
                m, k = T.mean_kl(rr["positions"], cc["positions"])
                if m is not None:
                    total += m * k
                    count += k
            row.update(mean_kl=round(total / count, 6) if count else None, kl_positions=count)
        out[prec] = row
    return out


# ---------------------------------------------------------------- m09 needle

def m09(ctx, a):
    """The 128k multi-key needle probe per precision: five probes, pass = all five pass."""
    out = {"status": "ok", "keys": 4, "probes": C.RUNS}
    for prec, (key, flags) in C.PRECISIONS.items():
        mid = model_name(key)
        with C.Server(ctx, key, flags, f"m09-{prec}") as srv:
            row = {"server": srv.record()}
            if live(ctx, srv):
                results = []
                for seed in range(C.RUNS):
                    try:
                        msgs, expected, count = fit_long_messages(
                            srv.url, mid, C.MAX_MODEL_LEN - 256, seed=seed, keys=4)
                        r = C.post(srv.url, "/v1/chat/completions",
                                   {"model": mid, "messages": msgs, "max_tokens": 128,
                                    "temperature": 0})
                        ans = r["choices"][0]["message"].get("content") or ""
                        results.append({"seed": seed, "prompt_tokens": count,
                                        "passed": T.needle_passed(ans, expected),
                                        "answer": ans[:400]})
                    except Exception as e:
                        results.append({"seed": seed, "error": f"{type(e).__name__}: {e}"[:500]})
                ok = [r for r in results if "passed" in r]
                row.update(probes=results, passed=sum(r["passed"] for r in ok),
                           result="pass" if len(ok) == C.RUNS and all(r["passed"] for r in ok)
                           else "fail")
            elif not ctx.dry:
                out["status"] = "partial"
        out[prec] = row
    return out


# ---------------------------------------------------------------- m10

def m10(ctx, a):
    """The small-model rejection at 128k, and step cost of SWE-Dev-7B against stock Coder-7B."""
    out = {"status": "ok", "rejections": {}}
    cases = {"coder1.5b_agent_flags": ("coder1.5b", C.serve_flags_128k()),
             "coder1.5b_no_yarn": ("coder1.5b", ["--kv-cache-dtype", "fp8", "--max-model-len",
                                                 str(C.MAX_MODEL_LEN)]),
             "qwen7b_no_yarn_control": ("qwen7b", ["--kv-cache-dtype", "fp8", "--max-model-len",
                                                   str(C.MAX_MODEL_LEN)])}
    for name, (key, flags) in cases.items():
        with C.Server(ctx, key, flags, f"m10-{name}", timeout=900) as srv:
            out["rejections"][name] = {"started": srv.ok, "error_line": srv.error,
                                       "server": srv.record()}
    # Both checkpoints' tokenizer configs cap the context below 128k (Coder-7B-Instruct at
    # 32768), so the step-cost pair runs at 32768 with bf16 weights, as shipped.
    flags = ["--kv-cache-dtype", "fp8", "--max-model-len", "32768"]
    for key in ("coder7b", "swedev7b"):
        with C.Server(ctx, key, flags, f"m10-{key}") as srv:
            row = {"server": srv.record()}
            t = timed_mini(ctx, srv, model_name(key))
            if t:
                row.update(tpot_b1_ms=t["tpot_ms"], ttft_s=t["ttft_s"])
            elif not ctx.dry:
                out["status"] = "partial"
        out[key] = row
    try:
        out["swedev_over_stock_tpot"] = round(out["swedev7b"]["tpot_b1_ms"]["median"]
                                              / out["coder7b"]["tpot_b1_ms"]["median"], 4)
    except (KeyError, TypeError, ZeroDivisionError):
        out["swedev_over_stock_tpot"] = None
    return out


# ---------------------------------------------------------------- session replays

def pick_session(turns, wanted=None, limit=None):
    """The named session, or the one with the most turns; its first `limit` turns."""
    ss = T.sessions(turns)
    if wanted:
        if wanted not in ss:
            raise Blocked(f"session {wanted} is not in the recording")
        sid = wanted
    else:
        sid = max(ss, key=lambda s: (len(ss[s]), s))
    return sid, ss[sid][:limit] if limit else ss[sid]


def replay_session(url, model, session, conc, warm, greedy=False, return_ids=False):
    """Replay one recorded session, `conc` copies at once. Warm: each copy keeps one cache
    salt for all its turns, so every turn after the first finds its prefix cached. Cold:
    every turn gets a fresh salt and pays a full prefill."""
    run_id = uuid.uuid4().hex[:8]

    def copy(i):
        res = []
        for t in session:
            body = replay_body(t, model, greedy)
            body["cache_salt"] = f"{run_id}-{i}" if warm else uuid.uuid4().hex
            if return_ids:
                body["return_token_ids"] = True
            try:
                r = C.stream_timed(url, "/v1/chat/completions", body)
                r["turn_id"] = t["turn_id"]
            except Exception as e:
                r = {"turn_id": t["turn_id"], "error": f"{type(e).__name__}: {e}"[:300]}
            res.append(r)
        return res

    with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as ex:
        return list(ex.map(copy, range(conc)))


def session_cell(url, model, session, conc, warm):
    """Five repeats; each repeat's value is the median over all its turns."""
    before = C.parse_prometheus(C.get_text(url, "/metrics"))
    tpots, turn_s, ttfts, errors = [], [], [], 0
    for _ in range(C.RUNS):
        copies = replay_session(url, model, session, conc, warm)
        flat = [r for c in copies for r in c]
        ok = [r for r in flat if "error" not in r]
        errors += len(flat) - len(ok)
        tpots.append(statistics.median([r["tpot_ms"] for r in ok if r["tpot_ms"]] or [None]))
        turn_s.append(statistics.median([r["turn_s"] for r in ok] or [None]))
        ttfts.append(statistics.median([r["ttft_s"] for r in ok if r["ttft_s"]] or [None]))
    after = C.parse_prometheus(C.get_text(url, "/metrics"))
    return {"tpot_ms": C.stat5(tpots), "turn_s": C.stat5(turn_s), "ttft_s": C.stat5(ttfts),
            "mean_acceptance_length": C.acceptance_from_counters(before, after),
            "failed_requests": errors}


def m11(ctx, a):
    """Speculation off, n-gram and suffix on a recorded editing session, FP8 weights."""
    turns = load_recording(ctx, a)
    session = None
    if turns is not None:
        sid, session = pick_session(turns, a.session, a.session_turns)
    key, flags = C.PRECISIONS["fp8"]
    mid = model_name(key)
    out = {"status": "ok" if turns is not None else "dry-run",
           "session": None if session is None else sid,
           "session_turns": None if session is None else len(session),
           "sampling": "the recorded request's own sampling settings",
           "spec_configs": SPEC}
    greedy = {}
    for name in SPEC:
        with C.Server(ctx, key, flags + spec_flags(name), f"m11-{name}",
                      prefix_caching=True) as srv:
            row = {"server": srv.record()}
            if live(ctx, srv) and session:
                s = srv.startup()
                row.update(kv_cache_line=s["kv_cache_line"], kv_cache_tokens=s["kv_cache_tokens"],
                           sessions_128k=s["max_concurrency"],
                           available_kv_cache_gib=s["available_kv_cache_gib"],
                           async_scheduling_disabled_by=s["async_scheduling_disabled_by"])
                for prefix in ("warm", "cold"):
                    for conc in (1, 4):
                        if prefix == "cold" and conc == 4 and a.no_cold_c4:
                            row["cold_c4"] = {"skipped": "--no-cold-c4"}
                            continue
                        row[f"{prefix}_c{conc}"] = session_cell(srv.url, mid, session, conc,
                                                                prefix == "warm")
                greedy[name] = {c: replay_session(srv.url, mid, session, c, True, greedy=True,
                                                  return_ids=True) for c in (1, 4)}
            elif not ctx.dry:
                out["status"] = "partial"
        out[name] = row
    if "off" in greedy:
        ref = {r["turn_id"]: r.get("token_ids") for r in greedy["off"][1][0]}
        for name in ("ngram", "suffix"):
            if name not in greedy:
                continue
            for conc, copies in greedy[name].items():
                same = total = 0
                for copy in copies:
                    for r in copy:
                        if r.get("token_ids") and ref.get(r["turn_id"]):
                            total += 1
                            same += r["token_ids"] == ref[r["turn_id"]]
                out[name][f"greedy_identical_c{conc}"] = {"identical": same, "turns": total}
    return out


def m12(ctx, a):
    """Turn time split, warm and cold, and the suffix tree's build time on a 128k prompt."""
    turns = load_recording(ctx, a)
    key, flags = C.PRECISIONS["fp8"]
    mid = model_name(key)
    out = {"status": "ok" if turns is not None else "dry-run"}
    target = prev = None
    if turns is not None:
        sid, session = pick_session(turns, a.session)
        cands = [i for i in range(1, len(session))]
        if not cands:
            raise Blocked("the session needs at least two turns for a warm turn")
        i = max(cands, key=lambda j: len(json.dumps(session[j]["request"]["messages"])))
        target, prev = session[i], session[i - 1]
        out.update(session=sid, turn_id=target["turn_id"])
    with C.Server(ctx, key, flags, "m12-fp8", prefix_caching=True) as srv:
        out["server"] = srv.record()
        if live(ctx, srv) and target:
            cells = {}
            for prefix in ("warm", "cold"):
                pre, dec, tot, ptoks = [], [], [], None
                for _ in range(C.RUNS):
                    salt = uuid.uuid4().hex
                    if prefix == "warm":
                        C.post(srv.url, "/v1/chat/completions",
                               dict(replay_body(prev, mid, False), max_tokens=1, cache_salt=salt))
                    r = C.stream_timed(srv.url, "/v1/chat/completions",
                                       dict(replay_body(target, mid, False), cache_salt=salt))
                    pre.append(r["ttft_s"])
                    dec.append(r["turn_s"] - r["ttft_s"])
                    tot.append(r["turn_s"])
                    ptoks = r["prompt_tokens"]
                cells[prefix] = {"prefill_s": C.stat5(pre), "decode_s": C.stat5(dec),
                                 "turn_s": C.stat5(tot), "prompt_tokens": ptoks,
                                 "decode_share": round(statistics.median(dec)
                                                       / statistics.median(tot), 4)}
            out.update(cells)
            out["suffix_tree_build"] = suffix_build_time(srv.url, mid, target)
        elif not ctx.dry and target:
            out["status"] = "partial"
    if ctx.dry:
        print("  (then times arctic_inference SuffixDecodingCache.start_request on 131,072 "
              "token ids, five times, on the CPU)")
    return out


def suffix_build_time(url, model, turn):
    """Time what vLLM calls when a request starts (v1/spec_decode/suffix_decoding.py:72):
    SuffixDecodingCache.start_request, which builds the prompt's suffix tree."""
    try:
        import numpy as np
        from arctic_inference.suffix_decoding import SuffixDecodingCache
    except ImportError as e:
        return {"error": f"arctic_inference not importable: {e}"}
    ids = C.post(url, "/tokenize", {"model": model, "messages": turn["request"]["messages"]})["tokens"]
    source = f"recorded turn {turn['turn_id']}"
    if len(ids) < 120000:
        msgs, _, _ = fit_long_messages(url, model, C.MAX_MODEL_LEN)
        ids = C.post(url, "/tokenize", {"model": model, "messages": msgs})["tokens"]
        source = "synthetic haystack (the recorded turn is shorter than 120k tokens)"
    arr = np.ascontiguousarray(np.array(ids[:C.MAX_MODEL_LEN], dtype=np.int32))
    times = []
    for i in range(C.RUNS):
        cache = SuffixDecodingCache(max_tree_depth=24, max_cached_requests=10000)
        t0 = time.perf_counter()
        cache.start_request(f"r{i}", arr)
        times.append(time.perf_counter() - t0)
    return {"tokens": int(arr.size), "source": source, "build_s": C.stat5(times),
            "us_per_token": round(statistics.median(times) / arr.size * 1e6, 3),
            "max_tree_depth": 24}


def m13(ctx, a):
    """Suffix decoding's speedup against batch size, and where it falls to 1x."""
    turns = load_recording(ctx, a)
    session = pick_session(turns, a.session, a.session_turns)[1] if turns is not None else None
    key, flags = C.PRECISIONS["fp8"]
    mid = model_name(key)
    out = {"status": "ok" if turns is not None else "dry-run", "concurrency": SPEEDUP_CONCURRENCY}
    for name in ("off", "suffix"):
        with C.Server(ctx, key, flags + spec_flags(name), f"m13-{name}",
                      prefix_caching=True) as srv:
            row = {"server": srv.record()}
            if live(ctx, srv) and session:
                for c in SPEEDUP_CONCURRENCY:
                    row[f"c{c}"] = session_cell(srv.url, mid, session, c, True)
            elif not ctx.dry:
                out["status"] = "partial"
        out[name] = row
    speedups = {}
    for c in SPEEDUP_CONCURRENCY:
        try:
            speedups[c] = round(out["off"][f"c{c}"]["tpot_ms"]["median"]
                                / out["suffix"][f"c{c}"]["tpot_ms"]["median"], 4)
        except (KeyError, TypeError, ZeroDivisionError):
            speedups[c] = None
    out["speedup"] = speedups
    hit = [c for c in SPEEDUP_CONCURRENCY if speedups[c] is not None and speedups[c] <= 1.0]
    out["crossover_batch"] = hit[0] if hit else None
    out["crossover_note"] = (None if hit else
                             "speedup stayed above 1x through concurrency 16, or cells failed")
    return out


# ---------------------------------------------------------------- m14

def strict_tools(tools, run=None):
    """Mark every tool strict (the per-request switch in vLLM 0.30.0,
    tool_parsers/structural_tag_registry.py:97-121). With `run`, add one optional property
    so the schema, and so the compiled grammar, is new to the server's grammar cache."""
    out = []
    for i, tool in enumerate(json.loads(json.dumps(tools))):
        fn = tool.setdefault("function", {})
        if run is not None and i == 0:
            params = fn.setdefault("parameters", {"type": "object", "properties": {}})
            params.setdefault("properties", {})[f"_run{run}"] = {"type": "string"}
        out.append(tool)
    return out


def set_strict(tools, on):
    for tool in tools:
        tool["function"]["strict"] = bool(on)
    return tools


def m14(ctx, a):
    """Strict tool calls off and on with the hermes parser: compile cost in TTFT, TPOT,
    and tool-call parse failures across the recorded turns."""
    turns = load_recording(ctx, a, need_tools=True)
    key, flags = C.PRECISIONS["fp8"]
    mid = model_name(key)
    out = {"status": "ok" if turns is not None else "dry-run",
           "switch": ("vLLM 0.30.0 has no --tool-strict-level flag; strictness is the "
                      "per-tool \"strict\": true in the request, which activates the hermes "
                      "structural tag under tool_choice auto (structural_tag_registry.py:120)."
                      " VLLM_ENFORCE_STRICT_TOOL_CALLING (envs.py:1751, default on) then "
                      "enforces the parameter schemas.")}
    with C.Server(ctx, key, flags + TOOL_FLAGS, "m14-fp8", prefix_caching=True) as srv:
        out["server"] = srv.record()
        if live(ctx, srv) and turns:
            first = turns[0]
            ttft = {"off_first": [], "off_second": [], "on_first": [], "on_second": []}
            for r in range(C.RUNS):
                for mode in ("off", "on"):
                    tools = set_strict(strict_tools(first["request"]["tools"], run=f"{mode}{r}"),
                                       mode == "on")
                    for which in ("first", "second"):
                        body = dict(replay_body(first, mid, True), tools=tools,
                                    cache_salt=uuid.uuid4().hex)
                        ttft[f"{mode}_{which}"].append(
                            C.stream_timed(srv.url, "/v1/chat/completions", body)["ttft_s"])
            out["ttft_s"] = {k: C.stat5(v) for k, v in ttft.items()}
            out["compile_s_estimate"] = round(out["ttft_s"]["on_first"]["median"]
                                              - out["ttft_s"]["on_second"]["median"], 4)
            sub = turns[:a.tpot_turns]
            for mode in ("off", "on"):
                med = []
                for _ in range(C.RUNS):
                    vals = []
                    for t in sub:
                        body = dict(replay_body(t, mid, False),
                                    tools=set_strict(strict_tools(t["request"]["tools"]),
                                                     mode == "on"))
                        r = C.stream_timed(srv.url, "/v1/chat/completions", body)
                        if r["tpot_ms"]:
                            vals.append(r["tpot_ms"])
                    med.append(statistics.median(vals) if vals else None)
                out[f"tpot_ms_{mode}"] = C.stat5(med)
                out[f"parse_{mode}"] = parse_failures(srv.url, mid, turns, mode == "on", 1)
        elif not ctx.dry:
            out["status"] = "partial"
    # Section 5.5's instruction: strict calls with suffix decoding on, concurrency 4.
    with C.Server(ctx, key, flags + TOOL_FLAGS + spec_flags("suffix"), "m14-fp8-suffix",
                  prefix_caching=True) as srv:
        out["suffix_server"] = srv.record()
        if live(ctx, srv) and turns:
            out["suffix_strict_c4"] = parse_failures(srv.url, mid, turns, True, 4)
    return out


def parse_failures(url, model, turns, strict, conc):
    by = {t["turn_id"]: t for t in turns}

    def one(t):
        body = dict(replay_body(t, model, False),
                    tools=set_strict(strict_tools(t["request"]["tools"]), strict))
        try:
            return t["turn_id"], C.stream_timed(url, "/v1/chat/completions", body)
        except Exception as e:
            return t["turn_id"], {"error": f"{type(e).__name__}: {e}"[:300]}

    with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as ex:
        res = list(ex.map(one, turns))
    att = valid = truncated = errors = 0
    examples = []
    for tid, r in res:
        if "error" in r:
            errors += 1
            continue
        v = T.validity(r, by[tid]["request"])
        att += v["attempts"]
        valid += v["valid"]
        truncated += r.get("finish_reason") == "length"
        examples += v["failures"][:1]
    return {"turns": len(turns), "concurrency": conc, "tool_call_attempts": att,
            "parse_failures": att - valid, "truncated": truncated, "request_errors": errors,
            "failure_examples": examples[:10]}


# ---------------------------------------------------------------- m15, report

def walk_spreads(obj, path=""):
    if isinstance(obj, dict):
        if obj.get("spread_pct") is not None and len(obj.get("runs") or []) >= 2:
            yield path, obj["spread_pct"]
        for k, v in obj.items():
            yield from walk_spreads(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk_spreads(v, f"{path}[{i}]")


def m15(ctx, a):
    """The widest spread across five runs, as a percent of the median, over every result."""
    if ctx.dry:
        print("  (reads results/m01.json to results/m14.json; no GPU)")
        return {"status": "dry-run"}
    found = []
    for mid in MEASUREMENTS:
        if mid in ("m15", "report", "m00"):
            continue
        r = C.read_result(mid)
        if r:
            found += [(f"{mid}.{p}", s) for p, s in walk_spreads(r.get("values", r))]
    if not found:
        return {"status": "blocked", "reason": "no results yet"}
    found.sort(key=lambda x: -x[1])
    return {"status": "ok", "largest_spread_pct": found[0][1], "where": found[0][0],
            "cells": len(found), "top5": found[:5]}


def _g(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def report(ctx, a):
    """The three tables the chapter prints, from whatever results exist."""
    if ctx.dry:
        print("  (formats results/*.json into results/report.md; no GPU)")
        return {"status": "dry-run"}
    m01, m04, m05, m06 = (C.read_result(x) or {} for x in ("m01", "m04", "m05", "m06"))
    m07, m08, m09, m11 = (C.read_result(x) or {} for x in ("m07", "m08", "m09", "m11"))
    v = lambda r: r.get("values", {})
    head = (C.read_result("m05") or {}).get("run_log", {})
    lines = [f"GPU {head.get('gpus')}, driver CUDA {head.get('cuda_supported_by_driver')}, "
             f"vLLM {_g(head, 'versions', 'vllm')} (CUDA build {head.get('cuda_build_torch')})",
             "", "precision  prefill_s_8k  prefill_s_32k  prefill_s_128k  decode_tok_s_b1  "
             "decode_tok_s_b16  sessions_128k"]
    for p in C.PRECISIONS:
        f = lambda *k: _g(v(m05), p, *k)
        row = [p, f("8k", "prefill_s", "median"), f("32k", "prefill_s", "median"),
               f("128k", "prefill_s", "median"), _g(v(m06), p, "c1", "per_sequence_tok_s"),
               _g(v(m06), p, "c16", "per_sequence_tok_s"), _g(v(m04), p, "sessions_128k")]
        lines.append("  ".join(str(x) for x in row))
    lines += ["", "flips (vs bf16), mean KL, tool-call validity, 128k needle"]
    for p in ("fp8", "awq"):
        fl = _g(v(m07), p) or {}
        lines.append(f"{p}: {fl.get('flips')} of {fl.get('turns')} turns, 95% CI "
                     f"{fl.get('ci95')}; mean KL {_g(v(m08), p, 'mean_kl')}; validity "
                     f"{_g(v(m08), p, 'validity_rate')}; needle {_g(v(m09), p, 'result')}")
    lines += ["", "TPOT ms, FP8, warm, concurrency 1: off "
              f"{_g(v(m11), 'off', 'warm_c1', 'tpot_ms', 'median')}, suffix "
              f"{_g(v(m11), 'suffix', 'warm_c1', 'tpot_ms', 'median')}"]
    text = "\n".join(lines) + "\n"
    (C.RESULTS / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return {"status": "ok", "report": text}


# ---------------------------------------------------------------- registry and CLI

MEASUREMENTS = {
    "m00": (m00, "preflight: versions, disk, checkpoint configs, 128k override resolution"),
    "m01": (m01, "kernel selected per precision (AWQ and FP8 startup lines, CUDA 12.4 check)"),
    "m02": (m02, "weights loaded per precision (model-loading line)"),
    "m04": (m04, "128k sessions that fit, cache size line, activation + CUDA graph reserve"),
    "m03": (m03, "bytes read per decode step and achieved bandwidth at batch 1"),
    "m05": (m05, "prefill time at 8k, 32k, 128k per precision"),
    "m06": (m06, "decode TPOT at batch 1, 4, 8, 16, plus warm 128k at batch 1"),
    "m09": (m09, "128k multi-key needle probe per precision"),
    "m10": (m10, "small-model rejection at 128k; SWE-Dev-7B vs stock Coder-7B step cost"),
    "m07": (m07, "flips per quantized model on recorded turns (needs a recording)"),
    "m08": (m08, "mean per-token KL and tool-call validity (needs a recording)"),
    "m11": (m11, "speculation off / n-gram / suffix on a recorded session (needs a recording)"),
    "m12": (m12, "turn time split warm vs cold; suffix tree build time (needs a recording)"),
    "m13": (m13, "suffix speedup against batch size 1 to 16 (needs a recording)"),
    "m14": (m14, "strict tool calls off and on (needs a recording with tools)"),
    "m15": (m15, "largest spread across five runs, all chapter 5 results"),
    "report": (report, "print the precision table, flip blocks and TPOT pair"),
}


def run_one(mid, a, argv):
    fn, desc = MEASUREMENTS[mid]
    ctx = C.Ctx(dry=a.dry_run, mid=mid)
    print(f"== {mid}: {desc}" + ("  [dry run]" if a.dry_run else ""), flush=True)
    t0 = time.monotonic()
    code = 0
    try:
        values = fn(ctx, a)
        status = values.pop("status", "ok")
        code = 0 if status in ("ok", "dry-run") else (2 if status == "blocked" else 1)
    except Blocked as e:
        values, status, code = {"reason": str(e)}, "blocked", 2
    except Exception as e:                       # one measurement's crash is its own result
        values = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-4000:]}
        status, code = "failed", 1
    result = {"measurement": mid, "description": desc, "status": status,
              "wall_minutes": round((time.monotonic() - t0) / 60, 2),
              "values": values, "run_log": C.run_log(ctx, argv)}
    if a.dry_run:
        print(f"   {status}: {len(ctx.commands)} commands validated")
    else:
        path = C.write_result(mid, result)
        print(f"   {status}: wrote {path.relative_to(C.HERE)}")
    return code


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("measurement", nargs="?", help="a measurement id from --list, or 'all'")
    ap.add_argument("--list", action="store_true", help="print the ids in run order and exit")
    ap.add_argument("--dry-run", action="store_true", help="print and validate, no GPU")
    ap.add_argument("--turns", default=str(DEFAULT_TURNS), help="the recorded turns, JSONL")
    ap.add_argument("--session", help="session id to replay (default: the longest)")
    ap.add_argument("--max-turns", type=int, help="m07, m08, m14: replay only the first N turns")
    ap.add_argument("--session-turns", type=int, default=8,
                    help="m11, m13: replay the session's first N turns (default 8)")
    ap.add_argument("--no-cold-c4", action="store_true",
                    help="m11: skip the cold-prefix cell at concurrency 4, the costliest one")
    ap.add_argument("--kl-topk", type=int, default=5,
                    help="prompt logprobs per position for KL (vLLM's max_logprobs is 20)")
    ap.add_argument("--tpot-turns", type=int, default=20,
                    help="m14: turns per TPOT repeat, strict off and on")
    a = ap.parse_args(argv)
    if a.list:
        print("\n".join(MEASUREMENTS))
        return 0
    if not a.measurement:
        ap.error("name a measurement (see --list) or 'all'")
    ids = list(MEASUREMENTS) if a.measurement == "all" else [a.measurement]
    unknown = [i for i in ids if i not in MEASUREMENTS]
    if unknown:
        ap.error(f"unknown measurement {unknown[0]}; see --list")
    codes = [run_one(i, a, ["run.py"] + argv) for i in ids]
    return max(codes)


if __name__ == "__main__":
    sys.exit(main())
