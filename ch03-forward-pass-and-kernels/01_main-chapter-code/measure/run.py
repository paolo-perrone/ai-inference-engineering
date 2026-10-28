#!/usr/bin/env python3
"""Chapter 3's measurements, one subcommand each, run on an NVIDIA L4.

  python run.py --list            the measurement ids, one per line, in GPU run order
  python run.py m01               one measurement; writes results/m01.json
  python run.py all               every measurement in --list order; one failure never
                                  stops the rest
  python run.py m07 --dry-run     print every command it would run and check the config;
                                  needs no GPU, no vLLM, no torch
  python run.py m05 --fresh       ignore cached engine sessions and traces in results/raw

Pins: vLLM 0.30.0, PyTorch 2.13.0 (vLLM 0.30.0's requirement), transformers 5.17.0, model
Qwen/Qwen2.5-7B-Instruct in bf16, greedy. Every result carries a run log with the GPU,
driver, CUDA, vLLM, PyTorch and transformers versions and the exact commands.

Engine sessions and traces are cached under results/raw and results/traces, so m02 to m04
reuse m01's engine start and m09 and m11 reuse m07's trace. Each subcommand still runs on
its own: if the cache is missing it does the work itself.
"""
import argparse
import json
import pathlib
import subprocess
import sys
import traceback

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import logparse  # noqa: E402
import nsys_tools  # noqa: E402
import vllm_session  # noqa: E402

SWEEP = ["mini", 4096, 8192]          # the draft's 1,457 (the real prompt), 4,096, 8,192
ABLATION_LENGTHS = "mini,4096,8192"
ABLATION_IMPLS = ["eager", "sdpa", "flash_attention_2"]
OOM_SEARCH_STEP = 256                 # the chapter's proximal win is a length, to 256 tokens
BENCH_INPUT_LEN = 1457                # the draft's command; vllm bench latency uses random IDs
BENCH_OUTPUT_LEN = 8
PROFILER_CUDA = json.dumps({"profiler": "cuda"})


class Ctx:
    def __init__(self, dry, fresh):
        self.dry, self.fresh = dry, fresh
        self.commands = []

    def note(self, cmd, what=""):
        text = common.shell(cmd) if isinstance(cmd, (list, tuple)) else str(cmd)
        self.commands.append(text)
        if self.dry:
            print(f"  would run{(' (' + what + ')') if what else ''}: {text}")

    def sh(self, cmd, what="", log=None, timeout=7200, check=True):
        self.note(cmd, what)
        if self.dry:
            return None
        if log:
            log.parent.mkdir(parents=True, exist_ok=True)
            with open(log, "w", encoding="utf-8") as f:
                p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
        else:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if check and p.returncode != 0:
            raise RuntimeError(f"exit {p.returncode}: {common.shell(cmd)}"
                               + (f"; see {log}" if log else f"\n{(p.stderr or '')[-3000:]}"))
        return p


# ------------------------------------------------------------------ shared steps

def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(common.MODEL)


def prompts_for(labels):
    """{label: prompt}. "mini" is the mini-session's text; a number is an exact token-ID list."""
    out, tok = {}, None
    for lab in labels:
        if lab == "mini":
            out["mini"] = common.mini_prompt()
        else:
            tok = tok or tokenizer()
            out[str(lab)] = common.padded_ids(tok, int(lab))
    return out


def serve(ctx, config, labels=("mini",)):
    if ctx.dry:
        ctx.note(vllm_session.serve_cmd(config, "<port>"), f"engine '{config}'")
        for lab in labels:
            ctx.note(f"POST /v1/completions x {common.WARMUP} warm-up + {common.RUNS} timed, "
                     f"prompt={lab}, max_tokens={common.COMPLETION_TOKENS}, temperature=0, "
                     f"ignore_eos, stream, return_token_ids, cache_salt", "requests")
        return None
    s = vllm_session.session(config, prompts_for(labels), fresh=ctx.fresh)
    ctx.commands.append(common.shell(s["server_command"]))
    return s


def hf(ctx, mode, *args, name=None):
    out = common.RAW / f"hf-{name or mode}.json"
    cmd = [sys.executable, str(HERE / "hf_worker.py"), mode, *map(str, args), "--out", str(out)]
    if ctx.dry:
        ctx.note(cmd, "transformers worker")
        return None
    cached = common.read_json(out)
    if cached and cached.get("ok") and not ctx.fresh:
        ctx.commands.append(common.shell(cmd) + "   # cached")
        return cached
    common.RAW.mkdir(parents=True, exist_ok=True)
    ctx.sh(cmd, "transformers worker", log=common.RAW / f"hf-{name or mode}.log", check=False)
    return common.read_json(out) or {"ok": False, "error": "worker wrote no result; see its log"}


def bench_app(config_flags, ranged):
    app = ["vllm", "bench", "latency", "--model", common.MODEL,
           "--num-iters-warmup", "5", "--num-iters", "1", "--batch-size", "1",
           "--input-len", str(BENCH_INPUT_LEN), "--output-len", str(BENCH_OUTPUT_LEN)]
    app += list(config_flags)
    if ranged:
        app += ["--profile", "--profiler-config", PROFILER_CUDA]
    return app


BENCH_FLAGS = {
    "default": [],
    "enforce_eager": ["--enforce-eager"],
    "graphs_none": ["--compilation-config", json.dumps({"cudagraph_mode": "NONE"})],
}


def bench_trace(ctx, config):
    """One `vllm bench latency` capture under nsys, read into kernels and decode steps.

    First try: one generate call only (--profile with the CUDA profiler, nsys
    --capture-range=cudaProfilerApi). If that captures no kernels, fall back to the
    draft's exact command, which captures the whole run, warm-ups included.
    """
    if not ctx.dry:
        common.TRACES.mkdir(parents=True, exist_ok=True)
    attempts = [("range", nsys_tools.profile_cmd(common.TRACES / f"bench-{config}-range",
                                                 bench_app(BENCH_FLAGS[config], True), "stop")),
                ("whole", nsys_tools.profile_cmd(common.TRACES / f"bench-{config}-whole",
                                                 bench_app(BENCH_FLAGS[config], False)))]
    for kind, cmd in attempts:
        base = common.TRACES / f"bench-{config}-{kind}"
        rep, db = base.with_suffix(".nsys-rep"), base.with_suffix(".sqlite")
        if ctx.dry:
            ctx.note(cmd, f"nsys capture, {kind}" + (" (fallback)" if kind == "whole" else ""))
            ctx.note(nsys_tools.export_cmd(rep, db), "export")
            ctx.note(nsys_tools.stats_cmd(rep), "kernel summary for the record")
            continue
        if not (db.exists() and not ctx.fresh):
            ctx.sh(cmd, f"nsys {kind}", log=base.with_suffix(".log"), timeout=5400, check=False)
            if not rep.exists():
                continue
            ctx.sh(nsys_tools.export_cmd(rep, db), "export", timeout=3600)
            p = ctx.sh(nsys_tools.stats_cmd(rep), "stats", timeout=3600, check=False)
            base.with_suffix(".kern_sum.csv").write_text(p.stdout if p else "", encoding="utf-8")
        else:
            ctx.commands.append(common.shell(cmd) + "   # cached trace")
        trace = nsys_tools.read_sqlite(db)
        if trace["kernels"]:
            steps, method = nsys_tools.decode_steps(trace["kernels"])
            return {"capture": kind, "command": common.shell(cmd), "report": str(rep),
                    "kernels_captured": len(trace["kernels"]), "trace": trace,
                    "steps": steps, "method": method}
    if ctx.dry:
        return None
    raise RuntimeError("nsys captured no kernels in either mode; see results/traces/*.log "
                       "(CUPTI permission in the container is the usual cause)")


def strip_trace(t):
    return {k: v for k, v in t.items() if k not in ("trace", "steps")}


# ------------------------------------------------------------------ measurements

def m01(ctx):
    s = serve(ctx, "default")
    if ctx.dry:
        return None
    p = s["prompts"]["mini"]
    env = s["environment_ch2_harness"]
    gpu = env["gpus"][0] if env["gpus"] else {}
    return {
        "gpu": gpu.get("name"), "driver": gpu.get("driver"), "vllm_version": env.get("vllm"),
        "prompt_tokens_engine": (p["prompt_tokens"] or [None])[0],
        "completion_tokens": p["completion_tokens"],
        "prefill_ms_median": p["ttft_ms"]["median"], "prefill_ms_spread": p["ttft_ms"]["spread"],
        "decode_ms_median": p["decode_ms"]["median"], "decode_ms_spread": p["decode_ms"]["spread"],
        "e2e_ms_median": p["e2e_ms"]["median"],
        "definition": "prefill = TTFT; decode = last token's arrival minus TTFT; "
                      "5 runs after 1 warm-up; spread = max minus min",
        "per_run": [{k: r[k] for k in ("ttft_ms", "decode_ms", "e2e_ms", "completion_tokens",
                                       "prompt_tokens")} for r in p["per_run"]],
        "server_command": s["server_command"],
    }


def m02(ctx):
    s = serve(ctx, "default")
    if ctx.dry:
        return None
    p = s["prompts"]["mini"]
    return {
        "ttft_ms_median": p["ttft_ms"]["median"], "ttft_ms_spread": p["ttft_ms"]["spread"],
        "tpot_ms_median": p["tpot_ms"]["median"], "tpot_ms_spread": p["tpot_ms"]["spread"],
        "mean_itl_ms": p["mean_itl_ms"], "itl_gaps_pooled": p["itl_gaps_pooled"],
        "tokens_per_chunk_min": p["tokens_per_chunk"]["min"],
        "tokens_per_chunk_max": p["tokens_per_chunk"]["max"],
        "tokens_per_chunk_mean": p["tokens_per_chunk"]["mean"],
        "definition": "TPOT = (E2E - TTFT)/(n - 1) per request, median of 5; mean ITL = mean of "
                      "every chunk-to-chunk gap pooled over the 5 runs; tokens per chunk from "
                      "each chunk's token_ids (return_token_ids)",
    }


def m03(ctx):
    s = serve(ctx, "default")
    if ctx.dry:
        return None
    step = s["prompts"]["mini"]["tpot_ms"]
    floor = common.BOOK["decode_floor_ms"]
    return {
        "decode_step_ms_median": step["median"], "decode_step_ms_spread": step["spread"],
        "decode_floor_ms": floor,
        "floor_pct_of_step": 100 * floor / step["median"],
        "step_pct_of_floor": 100 * step["median"] / floor,
        "achieved_gb_per_s": common.BOOK["decode_bytes_gb"] / (step["median"] / 1000),
        "book_constants": {k: common.BOOK[k] for k in ("decode_bytes_gb", "l4_hbm_gb_per_s")},
        "note": "floor_pct_of_step is the draft's wording ('the 47 ms floor as a percent of the "
                "step'); step_pct_of_floor is the plan's ('decode step time as a percent of the floor')",
    }


def m04(ctx):
    s = serve(ctx, "default")
    if ctx.dry:
        return None
    p = s["prompts"]["mini"]
    prefill_s = p["ttft_ms"]["median"] / 1000
    tf = common.BOOK["l4_dense_bf16_tflops"] * 1e12
    n_engine = (p["prompt_tokens"] or [None])[0]
    out = {"prefill_s_median": prefill_s, "prefill_ms_spread": p["ttft_ms"]["spread"],
           "prompt_tokens_engine": n_engine}
    for tag, ops in (("draft_19T", common.BOOK["prefill_ops_draft"]),
                     ("plan_22_1T", common.BOOK["prefill_ops_plan"])):
        floor = ops / tf
        out[f"floor_s_{tag}"] = floor
        out[f"floor_pct_of_prefill_{tag}"] = 100 * floor / prefill_s
        out[f"prefill_pct_of_floor_{tag}"] = 100 * prefill_s / floor
        out[f"achieved_tflops_{tag}"] = ops / prefill_s / 1e12
    if n_engine:
        ops = 2 * common.BOOK["layer_weights"] * n_engine
        out["ops_scaled_to_engine_count"] = ops
        out["floor_s_scaled"] = ops / tf
        out["floor_pct_of_prefill_scaled"] = 100 * (ops / tf) / prefill_s
    out["note"] = ("draft and chapter 2 count 19 trillion operations (0.16 s floor); the writing "
                   "plan counts 22.1 trillion (0.18 s). Both are computed; 'scaled' uses "
                   "2 x 6.5e9 x the engine's own prompt count, chapter 2's step 5 rule.")
    return out


def m05(ctx):
    s = serve(ctx, "default", labels=SWEEP)
    if ctx.dry:
        return None
    rows = []
    for lab in SWEEP:
        p = s["prompts"][str(lab)]
        rows.append({"label": str(lab), "prompt_tokens": (p["prompt_tokens"] or [None])[0],
                     "ttft_ms_median": p["ttft_ms"]["median"], "ttft_ms_spread": p["ttft_ms"]["spread"],
                     "decode_step_ms_median": p["tpot_ms"]["median"],
                     "decode_step_ms_spread": p["tpot_ms"]["spread"]})
    return {"sweep": rows, "decode_floor_ms": common.BOOK["decode_floor_ms"],
            "padding": "lengths other than 'mini' are exact token-ID lists: the mini-session "
                       "prompt extended with more of the fixture repo's own text"}


def m06(ctx):
    res = {name: serve(ctx, name) for name in ("default", "graphs_none", "enforce_eager")}
    if ctx.dry:
        return None
    st = {n: s["prompts"]["mini"]["tpot_ms"] for n, s in res.items()}
    tt = {n: s["prompts"]["mini"]["ttft_ms"] for n, s in res.items()}
    out = {}
    for n in res:
        out[f"decode_step_ms_{n}"] = st[n]["median"]
        out[f"decode_step_ms_spread_{n}"] = st[n]["spread"]
        out[f"ttft_ms_{n}"] = tt[n]["median"]
        out[f"server_command_{n}"] = res[n]["server_command"]
    out["ratio_graphs_none_over_default"] = st["graphs_none"]["median"] / st["default"]["median"]
    out["ratio_enforce_eager_over_default"] = st["enforce_eager"]["median"] / st["default"]["median"]
    out["paper_ratio_reported"] = common.BOOK["paper_graphs_ratio_l4"]
    out["note"] = ("graphs_none = --compilation-config '{\"cudagraph_mode\": \"NONE\"}' (the draft "
                   "and chapter 2: graphs off, compiler on); enforce_eager = --enforce-eager (the "
                   "plan: graphs off AND compiler off)")
    return out


def m07(ctx):
    t = bench_trace(ctx, "default")
    if ctx.dry:
        return None
    rep = nsys_tools.step_report(t["steps"], t["method"])
    return {
        "kernels_per_step": rep.get("kernels_per_step"),
        "kernels_per_layer": rep.get("kernels_per_layer"),
        "layer_kernels_total": rep.get("layer_kernels_total"),
        "kernels_outside_layers": rep.get("kernels_outside_layers"),
        "layer_sequence": rep.get("layer_sequence"),
        "rope_kernels": nsys_tools.rope_candidates(rep.get("layer_sequence") or []),
        "splitk_reduction_kernels_in_layer": rep.get("splitk_reduction_kernels_in_layer"),
        "copy_like_kernels_in_step": rep.get("copy_like_kernels_in_step"),
        "cublas_kernel_names_sm89": rep.get("cublas_kernels_in_layer"),
        "outside_layers_in_graph": rep.get("outside_layers_in_graph"),
        "after_graph": rep.get("after_graph"),
        "step_report": rep, "capture": strip_trace(t),
        "caution": "kernel roles (cuBLAS, split-K, copy, RoPE) are name heuristics; "
                   "layer_sequence holds every full name for a reader's check",
    }


def m08(ctx):
    out = {}
    for config in ("enforce_eager", "graphs_none"):
        t = bench_trace(ctx, config)
        if ctx.dry:
            continue
        rep = nsys_tools.step_report(t["steps"], t["method"])
        names = [k["full_name"] for k in rep.get("layer_sequence") or []]
        out[config] = {
            "kernels_per_layer": rep.get("kernels_per_layer"),
            "kernels_per_step": rep.get("kernels_per_step"),
            "layer_sequence": rep.get("layer_sequence"),
            "vllm_custom_kernels_found": {v: any(v in n for n in names)
                                          for v in nsys_tools.VLLM_EAGER_NAMES},
            "triton_kernels_in_layer": rep.get("triton_kernels_in_layer"),
            "capture": strip_trace(t),
        }
    if ctx.dry:
        return None
    return {"enforce_eager_layer_sequence": out["enforce_eager"]["layer_sequence"],
            "enforce_eager_vllm_kernels_found": out["enforce_eager"]["vllm_custom_kernels_found"],
            "graphs_none_triton_kernels_present": bool(out["graphs_none"]["triton_kernels_in_layer"]),
            "graphs_none_layer_sequence": out["graphs_none"]["layer_sequence"],
            "details": out}


def serve_trace(ctx, config="qwen_defaults"):
    """`vllm serve` under nsys, one request between /start_profile and /stop_profile.
    qwen_defaults: no --generation-config vllm and no sampling fields, so Qwen's file applies.
    default: --generation-config vllm and temperature 0, the greedy pin the chapter uses."""
    greedy = config != "qwen_defaults"
    base = common.TRACES / f"serve-{config}"
    rep, db = base.with_suffix(".nsys-rep"), base.with_suffix(".sqlite")
    prefix = nsys_tools.profile_cmd(base, [], "stop-shutdown")
    extra = ["--profiler-config", PROFILER_CUDA]
    if ctx.dry:
        ctx.note(prefix + vllm_session.serve_cmd(config, "<port>", extra), "nsys + serve")
        ctx.note("POST /start_profile; POST /v1/completions prompt=mini max_tokens=8, "
                 + ("temperature=0" if greedy else "no sampling fields")
                 + ", no ignore_eos; POST /stop_profile", "requests")
        ctx.note(nsys_tools.export_cmd(rep, db), "export")
        return None
    if not (db.exists() and not ctx.fresh):
        import stream_client
        common.TRACES.mkdir(parents=True, exist_ok=True)
        eng = vllm_session.Engine(config, base.with_suffix(".log"), prefix=prefix, extra=extra)
        with eng:
            ctx.commands.append(common.shell(eng.cmd))
            eng.post("/start_profile")
            stream_client.stream_once(eng.url, stream_client.request_body(
                common.MODEL, common.mini_prompt(), 8, greedy=greedy, ignore_eos=False))
            try:
                eng.post("/stop_profile", timeout=120)
            except OSError:
                pass                          # stop-shutdown may close the server mid-reply
            try:
                eng.proc.wait(timeout=900)    # let nsys finish writing the report
            except subprocess.TimeoutExpired:
                pass
        if not rep.exists():
            raise RuntimeError(f"nsys wrote no report; see {base.with_suffix('.log')}")
        ctx.sh(nsys_tools.export_cmd(rep, db), "export", timeout=3600)
    trace = nsys_tools.read_sqlite(db)
    steps, method = nsys_tools.decode_steps(trace["kernels"])
    return nsys_tools.step_report(steps, method)


def m09(ctx):
    t = bench_trace(ctx, "default")
    caps = {}
    for config in ("default", "qwen_defaults"):
        try:
            caps[config] = serve_trace(ctx, config)
        except Exception as e:  # the head time still stands without the serve captures
            if ctx.dry:
                raise
            caps[config] = {"error": f"{type(e).__name__}: {e}"}
    q, g = caps["qwen_defaults"], caps["default"]
    if ctx.dry:
        return None
    rep = nsys_tools.step_report(t["steps"], t["method"])
    head = rep.get("output_head") or {}
    qh = (q or {}).get("output_head") or {}
    gh = (g or {}).get("output_head") or {}
    return {
        "output_head_ms_median": head.get("ms_median"), "output_head_ms_spread": head.get("ms_spread"),
        "output_head_kernel": head.get("full_name"),
        "step_split_method": t["method"],   # "cuda-graph runs": the head ran outside the graph
        "head_ms_derived": common.BOOK["head_ms_derived"],
        "sampling_kernels_bench_default": head.get("after_head"),
        "sampling_kernel_count_bench_default": len(head.get("after_head") or []),
        "sampling_kernels_greedy": gh.get("after_head"),
        "sampling_kernel_count_greedy": len(gh.get("after_head") or []) if gh else None,
        "sampling_kernels_qwen_defaults": qh.get("after_head"),
        "sampling_kernel_count_qwen_defaults": len(qh.get("after_head") or []) if qh else None,
        "qwen_defaults_report": q,
        "greedy_report": g,
        "caution": "kernels after the head run until the next step's graph replay, so the "
                   "lists can include the next step's input preparation; read the names",
        "bench_sampling": "vllm bench latency hard-codes temperature=1.0, top_p=1.0, ignore_eos "
                          "(vllm/benchmarks/latency.py)",
    }


def m10(ctx):
    if ctx.dry:
        ctx.note(vllm_session.serve_cmd("qwen_defaults", "<port>"),
                 "engine without --generation-config vllm")
        ctx.note("huggingface_hub.hf_hub_download(MODEL, 'generation_config.json')", "read the file")
        return None
    from huggingface_hub import hf_hub_download
    gen = json.loads(pathlib.Path(hf_hub_download(common.MODEL, "generation_config.json")).read_text())
    log_path = common.RAW / f"serve-qwen_defaults-{common.now()}.log"
    with vllm_session.Engine("qwen_defaults", log_path) as eng:
        ctx.commands.append(common.shell(eng.cmd))
        text = eng.log_text()
    backend = logparse.attention_backend(text)
    override = logparse.sampling_override(text)
    greedy = common.read_json(common.RAW / "serve-default.json") or {}
    glog = pathlib.Path(greedy.get("server_log", "/nonexistent"))
    g_backend = logparse.attention_backend(glog.read_text(errors="replace")) if glog.exists() else None
    return {"flash_attn_backend_line": backend and backend["line"],
            "attention_backend": backend and backend["backend"],
            "backend_candidates": backend and backend["candidates"],
            "sampling_override_line": override and override["line"],
            "sampling_override_params": override and override["params"],
            "generation_config_json": gen,
            "backend_line_greedy_engine": g_backend and g_backend["line"],
            "server_log": str(log_path)}


def m11(ctx):
    t = bench_trace(ctx, "default")
    if ctx.dry:
        return None
    pre = nsys_tools.prefill_kernels(t["trace"]["kernels"], t["steps"])
    pre_attn = sorted({k["short"] for k in pre if nsys_tools.ATTN_RE.search(k["name"])})
    rep = nsys_tools.step_report(t["steps"], t["method"])
    dec_attn = rep.get("attention_kernels_in_step") or []
    every = {k["short"] for k in t["trace"]["kernels"]}
    return {"attention_kernels_prefill": pre_attn, "attention_kernels_decode": dec_attn,
            "combine_in_decode": any("combine" in n for n in dec_attn),
            "combine_in_prefill": any("combine" in n for n in pre_attn),
            "flash_fwd_kernel_seen_anywhere": any(n.startswith("flash_fwd_kernel") for n in every),
            "prefill_kernels_counted": len(pre), "capture": strip_trace(t),
            "caution": "prefill = kernels before the first decode step; exact for the "
                       "one-generate ('range') capture, mixed with warm-ups for 'whole'"}


def m12(ctx):
    r = hf(ctx, "kernels", "--impl", "sdpa", name="kernels-sdpa")
    if ctx.dry:
        return None
    if not r.get("ok"):
        return {"error": r.get("error"), "worker": r}
    names = [f'{k["role"]}: {k["op"]} -> {k["kernel"]}' for k in r["add_and_norm_kernels"]]
    return {"kernels_per_layer": r["kernels_per_layer"],
            "kernels_per_layer_values": r["kernels_per_layer_values"],
            "kernels_per_step": r["kernels_per_step"],
            "add_and_norm_kernel_names": names, "layer0": r["layer0"],
            "attn_implementation": r["impl"],
            "note": "transformers in eager PyTorch (no compile), sdpa attention (transformers' "
                    "default), one decode step after the mini-session prefill, torch.profiler"}


def m13(ctx):
    out = {}
    for comp in (0, 1):
        r = hf(ctx, "fusion", "--impl", "flash_attention_2", "--compile", comp,
               name=f"fusion-fa2-compile{comp}")
        if ctx.dry:
            continue
        tag = "hf_compile" if comp else "hf_eager"
        for k in ("prefill_ms", "prefill_ms_spread", "decode_step_ms", "decode_step_ms_spread"):
            out[f"{tag}_{k}"] = r.get(k)
        out[f"{tag}_attn"] = r.get("impl")
        if r.get("error"):
            out[f"{tag}_error"] = r["error"]
    for config in ("default", "custom_ops_all", "O0"):
        try:
            s = serve(ctx, config)
        except Exception as e:
            out[f"vllm_{config}_error"] = f"{type(e).__name__}: {e}"
            continue
        if ctx.dry:
            continue
        p = s["prompts"]["mini"]
        out[f"vllm_{config}_prefill_ms"] = p["ttft_ms"]["median"]
        out[f"vllm_{config}_prefill_ms_spread"] = p["ttft_ms"]["spread"]
        out[f"vllm_{config}_decode_step_ms"] = p["tpot_ms"]["median"]
        out[f"vllm_{config}_decode_step_ms_spread"] = p["tpot_ms"]["spread"]
    if ctx.dry:
        return None
    out["note"] = ("draft exercise 2: transformers eager vs torch.compile (mode='default', no "
                   "CUDA graphs), flash_attention_2 in both; vLLM default vs "
                   "-cc.custom_ops=[\"all\"]. The plan's vLLM -O0 arm is measured too.")
    return out


def m14(ctx):
    per = {}
    for impl in ABLATION_IMPLS:
        search = ["--search", OOM_SEARCH_STEP] if impl == "eager" else []
        r = hf(ctx, "ablation", "--impl", impl, "--lengths", ABLATION_LENGTHS, *search,
               name=f"ablation-{impl}")
        if not ctx.dry:
            per[impl] = r
    if ctx.dry:
        return None
    rows = []
    for impl, r in per.items():
        for row in r.get("rows", []):
            rows.append({"impl": impl, **{k: row.get(k) for k in
                         ("label", "prompt_len", "prefill_ms", "prefill_ms_spread", "peak_gb",
                          "peak_above_weights_gb", "oom", "error")}})

    def at(impl, label, key):
        for row in rows:
            if row["impl"] == impl and row["label"] == label:
                return row.get(key)
        return None
    e, f = at("eager", "mini", "prefill_ms"), at("flash_attention_2", "mini", "prefill_ms")
    log = common.run_log()
    return {"driver": log["driver"],
            "torch_version": log["versions"]["torch"],
            "transformers_version": log["versions"]["transformers"],
            "rows": rows,
            "first_oom_length": {impl: r.get("first_oom_length") for impl, r in per.items()},
            "oom_length_eager": (per.get("eager", {}).get("oom_search") or {}).get("oom_length"),
            "longest_ran_eager": (per.get("eager", {}).get("oom_search") or {}).get("longest_ran"),
            "oom_search_eager": per.get("eager", {}).get("oom_search"),
            "peak_gb_eager_mini": at("eager", "mini", "peak_gb"),
            "peak_gb_fa2_mini": at("flash_attention_2", "mini", "peak_gb"),
            "prefill_gap_ms_eager_minus_fa2_mini": (e - f) if e is not None and f is not None else None,
            "prefill_ratio_eager_over_fa2_mini": (e / f) if e and f else None,
            "attn_implementation": {impl: r.get("impl") for impl, r in per.items()},
            "errors": {impl: r.get("error") for impl, r in per.items() if r.get("error")},
            "settings": next((r.get("settings") for r in per.values() if r.get("settings")), None)}


def m15(ctx):
    r = hf(ctx, "memfree", name="memfree")
    if ctx.dry:
        return None
    if not r.get("ok"):
        return {"error": r.get("error")}
    return {k: r[k] for k in ("free_gb", "total_gb", "allocated_gb", "reserved_gb",
                              "weights_gb_from_parameters", "free_bytes", "total_bytes")}


def m16(ctx):
    r = hf(ctx, "tokens", name="tokens")
    if ctx.dry:
        print(f"  fixture prompt builds: {len(common.mini_prompt())} characters, "
              f"estimate {len(common.mini_prompt()) // 4} tokens")
        return None
    if not r.get("ok"):
        return {"error": r.get("error")}
    m01_raw = common.read_json(common.RAW / "serve-default.json") or {}
    engine = (m01_raw.get("prompts", {}).get("mini", {}).get("prompt_tokens") or [None])[0]
    return {"first_ids": r["first_ids"], "first_tokens": r["first_tokens"], "count": r["count"],
            "fixture_estimate": common.BOOK["fixture_prompt_estimate"],
            "matches_estimate": r["count"] == common.BOOK["fixture_prompt_estimate"],
            "engine_prompt_tokens": engine,
            "matches_engine": (r["count"] == engine) if engine else None,
            "tokenizer_class": r["tokenizer_class"], "characters": r["characters"]}


# id, function, title, estimated L4-minutes, what it needs, the fields placeholders read.
# The order is the GPU run order --list prints: cheap and robust first, Nsight last.
MEASUREMENTS = [
    ("m16", m16, "The request as tokens", 1, "tokenizer only (CPU)",
     ["first_ids", "count"]),
    ("m01", m01, "Prefill and decode split of the fixture request", 6, "vllm",
     ["driver", "gpu", "vllm_version", "prompt_tokens_engine", "completion_tokens",
      "prefill_ms_median", "decode_ms_median", "prefill_ms_spread", "decode_ms_spread"]),
    ("m02", m02, "TTFT, TPOT, mean ITL, tokens per chunk", 0, "vllm (reuses m01's engine run)",
     ["ttft_ms_median", "ttft_ms_spread", "tpot_ms_median", "tpot_ms_spread", "mean_itl_ms",
      "tokens_per_chunk_min", "tokens_per_chunk_max", "tokens_per_chunk_mean"]),
    ("m03", m03, "Decode step against the 47 ms floor", 0, "vllm (reuses m01's engine run)",
     ["decode_step_ms_median", "floor_pct_of_step", "step_pct_of_floor"]),
    ("m04", m04, "Prefill against the compute floor", 0, "vllm (reuses m01's engine run)",
     ["floor_pct_of_prefill_draft_19T", "prefill_pct_of_floor_draft_19T",
      "floor_pct_of_prefill_plan_22_1T"]),
    ("m05", m05, "Prompt-length sweep", 8, "vllm", ["sweep"]),
    ("m06", m06, "CUDA graphs off and on", 9, "vllm",
     ["ratio_graphs_none_over_default", "ratio_enforce_eager_over_default",
      "decode_step_ms_default", "decode_step_ms_graphs_none", "decode_step_ms_enforce_eager"]),
    ("m10", m10, "Startup log lines", 4, "vllm",
     ["flash_attn_backend_line", "sampling_override_line", "sampling_override_params",
      "generation_config_json"]),
    ("m13", m13, "Fusion on and off", 16, "transformers + vllm",
     ["hf_eager_prefill_ms", "hf_eager_decode_step_ms", "hf_compile_prefill_ms",
      "hf_compile_decode_step_ms", "vllm_default_prefill_ms", "vllm_custom_ops_all_prefill_ms",
      "vllm_O0_prefill_ms"]),
    ("m15", m15, "Free memory beside the weights", 2, "transformers", ["free_gb", "total_gb"]),
    ("m14", m14, "The attention ablation", 10, "transformers (+ kernels for flash_attention_2)",
     ["driver", "torch_version", "transformers_version", "rows", "peak_gb_eager_mini",
      "peak_gb_fa2_mini", "prefill_gap_ms_eager_minus_fa2_mini", "first_oom_length",
      "oom_length_eager", "longest_ran_eager"]),
    ("m12", m12, "Transformers eager kernels per layer", 3, "transformers",
     ["kernels_per_layer", "add_and_norm_kernel_names"]),
    ("m07", m07, "Kernels of one decode step, default config", 8, "vllm + Nsight Systems",
     ["kernels_per_step", "kernels_per_layer", "layer_sequence", "rope_kernels",
      "splitk_reduction_kernels_in_layer", "copy_like_kernels_in_step", "cublas_kernel_names_sm89"]),
    ("m11", m11, "Attention kernels in prefill and decode", 0, "vllm + Nsight (reuses m07's trace)",
     ["attention_kernels_prefill", "attention_kernels_decode", "combine_in_decode"]),
    ("m09", m09, "Graph boundary and sampling tail", 8, "vllm + Nsight (reuses m07's trace)",
     ["output_head_ms_median", "output_head_kernel", "sampling_kernels_bench_default",
      "sampling_kernels_greedy", "sampling_kernels_qwen_defaults"]),
    ("m08", m08, "Kernel names under --enforce-eager", 14, "vllm + Nsight Systems",
     ["enforce_eager_layer_sequence", "enforce_eager_vllm_kernels_found",
      "graphs_none_triton_kernels_present"]),
]
BY_ID = {m[0]: m for m in MEASUREMENTS}


def run_one(mid, dry, fresh):
    _, fn, title, est, needs, fields = BY_ID[mid]
    ctx = Ctx(dry, fresh)
    print(f"{mid}: {title}  [{needs}; about {est} L4-minutes]")
    payload = {"id": mid, "title": title}
    try:
        fields_out = fn(ctx)
        if dry:
            return 0
        payload["fields"] = fields_out
        missing = [f for f in fields if (fields_out or {}).get(f) is None]
        if missing:
            payload["missing_fields"] = missing
        ok = not (fields_out or {}).get("error") and not missing
    except Exception as e:
        if dry:
            print(f"  DRY-RUN FAILED: {type(e).__name__}: {e}")
            return 1
        payload["error"] = f"{type(e).__name__}: {e}"
        payload["traceback"] = traceback.format_exc()[-6000:]
        ok = False
    payload["commands"] = ctx.commands
    payload["run_log"] = common.run_log()
    path = common.write_result(mid, payload)
    print(f"  {'ok' if ok else 'FAILED or incomplete (recorded in the file)'}: {path}")
    return 0 if ok else 1


def dry_checks():
    problems = vllm_session.validate_configs()
    for flags in BENCH_FLAGS.values():
        for f in flags:
            if f.startswith("{"):
                json.loads(f)
    json.loads(PROFILER_CUDA)
    if common.mini.count_tokens(common.mini_prompt()) != common.BOOK["fixture_prompt_estimate"]:
        problems.append("the mini-session prompt no longer estimates to baseline.json's 1,457")
    ids = [m[0] for m in MEASUREMENTS]
    if sorted(ids) != [f"m{i:02d}" for i in range(1, 17)]:
        problems.append(f"measurement ids are not m01..m16: {ids}")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", nargs="?", help="a measurement id from --list, or 'all'")
    ap.add_argument("--list", action="store_true", help="print the ids in GPU run order")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="ignore cached sessions and traces")
    a = ap.parse_args()
    if a.list:
        print("\n".join(m[0] for m in MEASUREMENTS))
        return 0
    if not a.what:
        ap.print_help()
        return 2
    ids = [m[0] for m in MEASUREMENTS] if a.what == "all" else [a.what]
    unknown = [i for i in ids if i not in BY_ID]
    if unknown:
        print(f"unknown measurement {unknown[0]}; see --list", file=sys.stderr)
        return 2
    if a.dry_run:
        problems = dry_checks()
        for p in problems:
            print("CONFIG PROBLEM:", p)
        if problems:
            return 1
    rc = 0
    for mid in ids:
        rc |= run_one(mid, a.dry_run, a.fresh)
    if a.what == "all":
        print(f"estimated total: {sum(m[3] for m in MEASUREMENTS)} L4-minutes plus setup")
    return rc


if __name__ == "__main__":
    sys.exit(main())
