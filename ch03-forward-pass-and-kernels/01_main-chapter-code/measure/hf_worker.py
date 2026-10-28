#!/usr/bin/env python3
"""The transformers side of chapter 3: one configuration per process.

run.py starts this script once per configuration, so an out-of-memory error, a compile
cache or a leftover allocation never leaks into the next configuration.

  python hf_worker.py tokens   --out t.json
  python hf_worker.py memfree  --out m.json
  python hf_worker.py ablation --impl eager --lengths mini,4096,8192 --search 256 --out a.json
  python hf_worker.py fusion   --impl flash_attention_2 --compile 1 --out f.json
  python hf_worker.py kernels  --impl sdpa --out k.json

Names come from transformers 5.17.0's source: attn_implementation values "eager", "sdpa",
"flash_attention_2" (modeling_utils.py, AttentionInterface._global_mapping); without the
flash_attn package, "flash_attention_2" falls back to the hub kernel
"kernels-community/flash-attn2" when the `kernels` package is installed
(modeling_flash_attention_utils.py FLASH_ATTN_KERNEL_FALLBACK), and then
model.config._attn_implementation names that kernel instead. Both are recorded.
"""
import argparse
import importlib.util
import json
import statistics
import sys
import time
import traceback

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import common  # noqa: E402


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(common.MODEL)


def load_model(impl):
    import torch
    from transformers import AutoModelForCausalLM
    # dtype (not torch_dtype) and use_kernels per transformers 5.17 modeling_utils.py;
    # use_kernels=False keeps hub kernels from replacing the RMSNorm the eager trace counts.
    kw = {"dtype": torch.bfloat16, "device_map": "cuda", "use_kernels": False}
    if impl:
        kw["attn_implementation"] = impl
    model = AutoModelForCausalLM.from_pretrained(common.MODEL, **kw)
    model.eval()
    return model


def attn_record(model, requested):
    return {"requested": requested,
            "config_attn_implementation": getattr(model.config, "_attn_implementation", None),
            "flash_attn_installed": importlib.util.find_spec("flash_attn") is not None,
            "kernels_installed": importlib.util.find_spec("kernels") is not None}


def ids_for(tok, length):
    if length == "mini":
        return tok(common.mini_prompt())["input_ids"]
    return common.padded_ids(tok, int(length))


def cuda_ms(fn):
    import torch
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    a.record()
    out = fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b), out


# ------------------------------------------------------------------ measurement 16

def mode_tokens(a):
    tok = load_tokenizer()
    prompt = common.mini_prompt()
    ids = tok(prompt)["input_ids"]
    return {"count": len(ids), "first_ids": ids[:16], "first_tokens": tok.convert_ids_to_tokens(ids[:16]),
            "vocab_size_tokenizer": len(tok), "characters": len(prompt),
            "fixture_estimate_chars_over_4": len(prompt) // 4,
            "tokenizer_class": type(tok).__name__}


# ------------------------------------------------------------------ measurement 15

def mode_memfree(a):
    import torch
    model = load_model(a.impl)
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {"free_bytes": free, "total_bytes": total, "free_gb": free / 1e9, "total_gb": total / 1e9,
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "weights_gb_from_parameters": sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9,
            "attn": attn_record(model, a.impl)}


# ------------------------------------------------------------------ measurement 14

def mode_ablation(a):
    import torch
    tok = load_tokenizer()
    model = load_model(a.impl)
    torch.cuda.synchronize()
    after_load = torch.cuda.memory_allocated()
    rows = []
    for length in a.lengths.split(","):
        ids = ids_for(tok, length)
        x = torch.tensor([ids], device="cuda")
        row = {"prompt_len": len(ids), "label": length}
        try:
            with torch.inference_mode():
                # logits_to_keep=1: only the last position's scores, as generate() asks for in
                # prefill; use_cache=False: the pass allocates no KV cache. Both recorded.
                for _ in range(common.WARMUP):
                    model(x, use_cache=False, logits_to_keep=1)
                times, peaks = [], []
                for _ in range(a.runs):
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    ms, _ = cuda_ms(lambda: model(x, use_cache=False, logits_to_keep=1))
                    times.append(ms)
                    peaks.append(torch.cuda.max_memory_allocated())
            med, spread = common.med_spread(times)
            row.update({"oom": False, "prefill_ms": med, "prefill_ms_spread": spread,
                        "prefill_ms_runs": times,
                        "peak_gb": max(peaks) / 1e9,
                        "peak_above_weights_gb": (max(peaks) - after_load) / 1e9})
        except torch.cuda.OutOfMemoryError as e:
            row.update({"oom": True, "error": str(e).splitlines()[0]})
        except Exception as e:  # a missing kernel package is a result too
            row.update({"oom": False, "error": f"{type(e).__name__}: {e}"})
        finally:
            del x
            torch.cuda.empty_cache()
        rows.append(row)
    ooms = [r["prompt_len"] for r in rows if r.get("oom")]
    out = {"impl": attn_record(model, a.impl), "weights_allocated_gb": after_load / 1e9,
           "settings": {"logits_to_keep": 1, "use_cache": False, "runs": a.runs,
                        "warmup": common.WARMUP, "timer": "torch.cuda.Event"},
           "rows": rows, "first_oom_length": min(ooms) if ooms else None}
    if a.search:
        ran = [r["prompt_len"] for r in rows if r.get("oom") is False and not r.get("error")]
        out["oom_search"] = oom_search(model, tok, max(ran) if ran else 0,
                                       min(ooms) if ooms else None, a.search)
    return out


def fits(model, tok, length):
    """One prefill at exactly `length` tokens: True if it ran, False if it ran out of memory."""
    import gc
    import torch
    x = torch.tensor([common.padded_ids(tok, length)], device="cuda")
    try:
        with torch.inference_mode():
            model(x, use_cache=False, logits_to_keep=1)
        torch.cuda.synchronize()
        return True
    except torch.cuda.OutOfMemoryError:
        return False
    finally:
        del x
        gc.collect()
        torch.cuda.empty_cache()


def oom_search(model, tok, ran, failed, step):
    """The out-of-memory prompt length, to the nearest `step` tokens: the shortest length
    that fails, with the longest that runs `step` below it. With no failure yet, the bound
    doubles up to the model's 32,768-token context; then it bisects on multiples of `step`."""
    limit = common.MAX_CONTEXT
    probes = []
    lo = (ran // step) * step
    hi = failed
    while hi is None:
        nxt = min(max(lo, step) * 2, limit)
        ok = fits(model, tok, nxt)
        probes.append({"prompt_len": nxt, "ran": ok})
        if ok:
            lo = nxt
            if nxt >= limit:
                return {"step": step, "probes": probes, "longest_ran": lo, "oom_length": None,
                        "note": f"ran at the model's {limit}-token limit; no out-of-memory length"}
        else:
            hi = nxt
    hi = -(-hi // step) * step
    while hi - lo > step:
        mid = ((lo + hi) // 2 // step) * step
        ok = fits(model, tok, mid)
        probes.append({"prompt_len": mid, "ran": ok})
        lo, hi = (mid, hi) if ok else (lo, mid)
    return {"step": step, "probes": probes, "longest_ran": lo, "oom_length": hi,
            "note": "one prefill per probe; an earlier out-of-memory error can leave the "
                    "allocator fragmented, so the length is a lower bound on what fits"}


# ------------------------------------------------------------------ measurement 13

def _static_cache(model, max_len):
    from transformers import StaticCache
    return StaticCache(config=model.config, max_cache_len=max_len)


def mode_fusion(a):
    """Prefill and decode-step time, eager PyTorch against torch.compile, same attention."""
    import torch
    tok = load_tokenizer()
    model = load_model(a.impl)
    ids = ids_for(tok, "mini")
    n = len(ids)
    x = torch.tensor([ids], device="cuda")
    steps = a.decode_steps
    cache = _static_cache(model, n + steps + 8)
    fwd = model.forward
    if a.compile:
        # mode="default": inductor fusion without CUDA graphs ("reduce-overhead" would add
        # graphs), so the comparison isolates fusion.
        fwd = torch.compile(model.forward, mode="default", fullgraph=False)

    # transformers 5.17's forward takes no cache_position: the StaticCache tracks its own
    # length (cache_utils.py StaticLayer.cumulative_length) and reset() rewinds it.
    def prefill():
        cache.reset()
        return fwd(input_ids=x, past_key_values=cache, use_cache=True, logits_to_keep=1)

    def decode_run():
        out = prefill()
        nxt = out.logits[:, -1:].argmax(-1)
        per = []
        for i in range(steps):
            ms, out = cuda_ms(lambda: fwd(input_ids=nxt, past_key_values=cache,
                                          use_cache=True, logits_to_keep=1))
            nxt = out.logits[:, -1:].argmax(-1)
            per.append(ms)
        return statistics.median(per[2:] if len(per) > 4 else per)

    result = {"impl": attn_record(model, a.impl), "compile": bool(a.compile), "prompt_len": n,
              "decode_steps_per_run": steps, "cache": "StaticCache"}
    try:
        with torch.inference_mode():
            t0 = time.perf_counter()
            for _ in range(2):                     # compile both shapes, then warm
                prefill()
                decode_run()
            result["warmup_s"] = time.perf_counter() - t0
            pre = [cuda_ms(prefill)[0] for _ in range(a.runs)]
            dec = [decode_run() for _ in range(a.runs)]
        result["prefill_ms"], result["prefill_ms_spread"] = common.med_spread(pre)
        result["decode_step_ms"], result["decode_step_ms_spread"] = common.med_spread(dec)
        result["prefill_ms_runs"], result["decode_step_ms_runs"] = pre, dec
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()[-4000:]
    return result


# ------------------------------------------------------------------ measurement 12

def _annotate(model):
    """Wrap each layer, its attention, MLP and both RMSNorms in a profiler label, so every
    kernel can be traced to the module that launched it. The residual adds are the
    kernels a layer launches outside all 4 of its children."""
    import torch
    from torch.profiler import record_function

    def wrap(mod, label):
        orig = mod.forward

        def fwd(*args, **kw):
            with record_function(label):
                return orig(*args, **kw)
        mod.forward = fwd

    for i, layer in enumerate(model.model.layers):
        wrap(layer, f"ch3.layer.{i}")
        wrap(layer.self_attn, "ch3.attn")
        wrap(layer.mlp, "ch3.mlp")
        wrap(layer.input_layernorm, "ch3.input_norm")
        wrap(layer.post_attention_layernorm, "ch3.post_attn_norm")
    wrap(model.model.norm, "ch3.final_norm")
    wrap(model.lm_head, "ch3.lm_head")
    return torch


def kernel_rows(events):
    """Flatten torch.profiler FunctionEvents into rows: the aten op, its kernels in launch
    order, and the chain of ch3.* labels above it. Pure on objects with .name, .kernels
    (each with .name), .cpu_parent, .time_range.start, .is_user_annotation."""
    rows = []
    for e in events:
        if not getattr(e, "kernels", None) or e.name.startswith("ch3."):
            continue
        labels, p = [], e.cpu_parent
        while p is not None:
            if p.name.startswith("ch3."):
                labels.append(p.name)
            p = p.cpu_parent
        rows.append({"op": e.name, "start": e.time_range.start,
                     "kernels": [k.name for k in e.kernels], "labels": labels})
    rows.sort(key=lambda r: r["start"])
    return rows


def per_layer(rows, layer="ch3.layer.0"):
    """Kernels of one layer by role. Role = the innermost ch3 label under the layer."""
    out = []
    for r in rows:
        if layer not in r["labels"]:
            continue
        inner = r["labels"][0]
        role = "residual/other (layer body)" if inner == layer else inner[len("ch3."):]
        for k in r["kernels"]:
            out.append({"role": role, "op": r["op"], "kernel": k})
    return out


def mode_kernels(a):
    import torch
    from torch.profiler import ProfilerActivity, profile
    tok = load_tokenizer()
    model = load_model(a.impl)
    _annotate(model)
    ids = ids_for(tok, "mini")
    x = torch.tensor([ids], device="cuda")
    with torch.inference_mode():
        out = model(x, use_cache=True)
        cache = out.past_key_values
        nxt = out.logits[:, -1:].argmax(-1)
        for _ in range(3):                       # warm the decode path
            out = model(nxt, past_key_values=cache, use_cache=True)
            nxt = out.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            out = model(nxt, past_key_values=cache, use_cache=True)
            torch.cuda.synchronize()
    rows = kernel_rows(prof.events())
    layer0 = per_layer(rows, "ch3.layer.0")
    layers = {}
    for r in rows:
        for lab in r["labels"]:
            if lab.startswith("ch3.layer."):
                layers[lab] = layers.get(lab, 0) + len(r["kernels"])
    counts = sorted(set(layers.values()))
    total = sum(len(r["kernels"]) for r in rows)
    return {"impl": attn_record(model, a.impl), "cache": "DynamicCache", "profiler": "torch.profiler (CUPTI)",
            "kernels_per_step": total, "kernels_per_layer_values": counts,
            "kernels_per_layer": counts[0] if len(counts) == 1 else None,
            "layers_seen": len(layers), "layer0": layer0,
            "add_and_norm_kernels": [k for k in layer0 if k["role"] in
                                     ("residual/other (layer body)", "input_norm", "post_attn_norm")],
            "outside_layers": [k for r in rows if not any(l.startswith("ch3.layer.") for l in r["labels"])
                               for k in r["kernels"]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tokens", "memfree", "ablation", "fusion", "kernels"])
    ap.add_argument("--impl", default=None)
    ap.add_argument("--lengths", default="mini,4096,8192")
    ap.add_argument("--runs", type=int, default=common.RUNS)
    ap.add_argument("--compile", type=int, default=0)
    ap.add_argument("--decode-steps", type=int, default=32)
    ap.add_argument("--search", type=int, default=0,
                    help="ablation: find the out-of-memory prompt length to this many tokens")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fn = {"tokens": mode_tokens, "memfree": mode_memfree, "ablation": mode_ablation,
          "fusion": mode_fusion, "kernels": mode_kernels}[a.mode]
    try:
        res = {"ok": True, **fn(a)}
    except Exception as e:
        res = {"ok": False, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-4000:]}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=str)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
