"""Nsight Systems capture and trace analysis for chapter 3's kernel measurements.

Capture: the draft's command (vLLM's profiling documentation's own), optionally narrowed
to one generate call with vLLM 0.30.0's CUDA profiler hook (config/profiler.py
profiler="cuda", profiler/wrapper.py CudaProfilerWrapper calls cudaProfilerStart/Stop)
and nsys's --capture-range=cudaProfilerApi.

Read: `nsys export --type sqlite`, then the kernel table the nsys 2026.3.2 report scripts
read themselves (reports/kernel_helper.py QUERY_KERNEL: CUPTI_ACTIVITY_KIND_KERNEL joined
to StringIds on demangledName / shortName; graphNodeId per nsys_recipe/lib/table_config.py).
A CSV reader for `nsys stats --report cuda_gpu_trace --format csv` is the fallback.
"""
import csv
import io
import re
import sqlite3
import statistics

LAYERS = 28                          # Qwen2.5-7B num_hidden_layers

NSYS_BASE = ["nsys", "profile", "--trace-fork-before-exec=true", "--cuda-graph-trace=node",
             "--force-overwrite", "true"]


def profile_cmd(output, app_cmd, capture_range=None):
    """capture_range: None (whole run, the draft's command), "stop" or "stop-shutdown"."""
    cmd = NSYS_BASE + ["--output", str(output)]
    if capture_range:
        cmd += ["--capture-range=cudaProfilerApi", f"--capture-range-end={capture_range}"]
    return cmd + list(app_cmd)


def export_cmd(rep, sqlite_path):
    return ["nsys", "export", "--type", "sqlite", "--force-overwrite", "true",
            "--output", str(sqlite_path), str(rep)]


def stats_cmd(rep, report="cuda_gpu_kern_sum"):
    return ["nsys", "stats", "--report", report, "--format", "csv", "--output", "-", str(rep)]


# ---------------------------------------------------------------- reading a trace

def _columns(con, table):
    return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}


def _tables(con):
    return {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")}


def read_sqlite(path):
    """Kernels in start order, plus memcpy and memset events, from an nsys sqlite export."""
    con = sqlite3.connect(str(path))
    try:
        tables = _tables(con)
        if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables:
            return {"kernels": [], "memops": []}
        cols = _columns(con, "CUPTI_ACTIVITY_KIND_KERNEL")
        graph = "k.graphNodeId" if "graphNodeId" in cols else "NULL"
        short = "s.value" if "shortName" in cols else "NULL"
        short_join = ("LEFT JOIN StringIds s ON s.id = k.shortName" if "shortName" in cols else "")
        rows = con.execute(f"""
            SELECT k.start, k.end, {graph}, d.value, {short}, k.streamId
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            LEFT JOIN StringIds d ON d.id = k.demangledName
            {short_join}
            ORDER BY k.start""").fetchall()
        kernels = [{"start": a, "end": b, "graph_node": g, "name": n, "short": s or n,
                    "stream": st} for a, b, g, n, s, st in rows]
        memops = []
        for table, label in (("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"),
                             ("CUPTI_ACTIVITY_KIND_MEMSET", "memset")):
            if table in tables:
                memops += [{"start": a, "end": b, "kind": label, "bytes": n}
                           for a, b, n in con.execute(
                               f"SELECT start, end, bytes FROM {table} ORDER BY start")]
        memops.sort(key=lambda m: m["start"])
        return {"kernels": kernels, "memops": memops}
    finally:
        con.close()


def _col(header, prefix):
    for i, h in enumerate(header):
        if h.strip().lower().startswith(prefix.lower()):
            return i
    raise KeyError(prefix)


def read_gpu_trace_csv(text):
    """`nsys stats --report cuda_gpu_trace --format csv` output: kernels and memory ops.

    The report's columns (reports/cuda_gpu_trace.py) are Start, Duration, CorrId, GrdX ...
    Strm, Name; the binary renders units into the header ("Start (ns)"), so columns are
    found by prefix. Rows whose Name starts with "[CUDA mem" are memory operations.
    """
    lines = [l for l in text.splitlines() if l.strip()]
    start = next(i for i, l in enumerate(lines) if l.lstrip('"').startswith("Start"))
    reader = csv.reader(io.StringIO("\n".join(lines[start:])))
    header = next(reader)
    i_start, i_dur, i_name = _col(header, "Start"), _col(header, "Duration"), _col(header, "Name")
    kernels, memops = [], []
    for row in reader:
        if len(row) <= i_name:
            continue
        s, d, name = int(float(row[i_start])), int(float(row[i_dur])), row[i_name]
        if name.startswith("[CUDA mem"):
            memops.append({"start": s, "end": s + d, "kind": name, "bytes": None})
        else:
            kernels.append({"start": s, "end": s + d, "graph_node": None, "name": name,
                            "short": name, "stream": None})
    return {"kernels": kernels, "memops": memops}


# ---------------------------------------------------------------- finding the steps

def find_period(names, min_reps=3, max_period=4000):
    """The repeating unit at the end of a kernel sequence: (period, repeats).

    Picks the period that covers the most kernels (period x repeats), smallest on a tie, so
    a short accidental repeat (3 identical small kernels) never wins over the decode step.
    """
    n = len(names)
    best = (0, 0, 0)  # coverage, -period, repeats
    for p in range(1, min(n // min_reps, max_period) + 1):
        k = 1
        while (k + 1) * p <= n and names[n - (k + 1) * p: n - k * p] == names[n - p:]:
            k += 1
        if k >= min_reps and (p * k, -p) > best[:2]:
            best = (p * k, -p, k)
    return (-best[1], best[2]) if best[0] else (None, 0)


def find_layer_block(names, layers=LAYERS):
    """The smallest L and first offset where `layers` back-to-back copies of an L-kernel
    sequence sit in `names`. Returns (L, offset) or (None, None)."""
    n = len(names)
    for L in range(1, n // layers + 1):
        for s in range(0, n - L * layers + 1):
            unit = names[s:s + L]
            if all(names[s + i * L: s + (i + 1) * L] == unit for i in range(1, layers)):
                return L, s
    return None, None


def graph_runs(kernels):
    """Maximal runs of kernels replayed from a CUDA graph (graphNodeId set)."""
    runs, cur = [], None
    for i, k in enumerate(kernels):
        in_graph = k.get("graph_node") not in (None, 0)
        if in_graph and cur is None:
            cur = [i, i]
        elif in_graph:
            cur[1] = i
        elif cur is not None:
            runs.append(tuple(cur))
            cur = None
    if cur is not None:
        runs.append(tuple(cur))
    return runs


def decode_steps(kernels, layers=LAYERS):
    """Split a capture into decode steps. Returns a list of steps, each
    {"graph": [kernels replayed from the graph], "tail": [kernels launched after it]},
    and the method used. Graph runs decide when the trace has them; otherwise the
    repeating period at the end of the trace does, aligned on the 28-layer block."""
    runs = [r for r in graph_runs(kernels) if r[1] - r[0] + 1 >= layers]
    if len(runs) >= 2:
        steps = []
        for i, (a, b) in enumerate(runs):
            end = runs[i + 1][0] if i + 1 < len(runs) else len(kernels)
            steps.append({"graph": kernels[a:b + 1], "tail": kernels[b + 1:end],
                          "first_index": a})
        lengths = [len(s["graph"]) for s in steps]
        common = statistics.mode(lengths)
        return [s for s in steps if len(s["graph"]) == common], "cuda-graph runs"
    if len(kernels) > 30000:          # a whole-run capture: the last decode steps suffice
        offset = len(kernels) - 30000
        steps, how = decode_steps(kernels[offset:], layers)
        for st in steps:
            st["first_index"] += offset
        return steps, how
    names = [k["short"] for k in kernels]
    period, reps = find_period(names)
    if not period:
        return [], "no repeating period found"
    region = kernels[len(kernels) - period * reps:]
    rnames = [k["short"] for k in region]
    L, off = find_layer_block(rnames[:period * 2] if reps >= 2 else rnames, layers)
    if L is None:
        steps = [{"graph": [], "tail": region[i * period:(i + 1) * period],
                  "first_index": len(kernels) - period * reps + i * period}
                 for i in range(reps)]
        return steps, "repeating period, no layer block found"
    # Align each step to start where the layer block starts; the first partial step is dropped.
    first = len(kernels) - period * reps + (off % period)
    steps = []
    i = first
    while i + period <= len(kernels):
        block = kernels[i:i + period]
        steps.append({"graph": block[:L * layers], "tail": block[L * layers:], "first_index": i})
        i += period
    return steps, "repeating period aligned on the layer block"


GEMM_RE = re.compile(r"gemm|gemv|cublas|cutlass|xmma|sm\d{2}_|splitK|nvjet", re.I)
SPLITK_RE = re.compile(r"splitK|split_k|splitk|reduce_kernel.*cublas|cublasLt.*reduc", re.I)
COPY_RE = re.compile(r"copy|memcpy|CatArrayBatched|index_put|scatter", re.I)
ATTN_RE = re.compile(r"flash|attn|attention", re.I)
VLLM_EAGER_NAMES = ("fused_add_rms_norm_kernel", "act_and_mul_kernel", "rotary_embedding_kernel")


def author(name):
    if name.startswith("triton_"):
        return "compiler (Triton, generated)"
    if GEMM_RE.search(name):
        return "cuBLAS / CUTLASS"
    return "hand-written"


def step_report(steps, method, layers=LAYERS):
    """Counts and names for one decode step, medians across the captured steps."""
    if not steps:
        return {"method": method, "steps_found": 0}
    sizes = [len(s["graph"]) + len(s["tail"]) for s in steps]
    first = steps[len(steps) // 2]                    # a middle step, past any warm edge
    body_names = [k["short"] for k in first["graph"]]
    L, off = find_layer_block(body_names, layers)
    layer = first["graph"][off:off + L] if L else []
    outside_graph = [k["short"] for k in first["graph"][:off or 0]] + \
                    [k["short"] for k in first["graph"][(off or 0) + (L or 0) * layers:]]
    tail_names = [k["short"] for k in first["tail"]]
    # The output head: the longest kernel in the tail, timed across steps.
    head = None
    if first["tail"]:
        durs = [k["end"] - k["start"] for k in first["tail"]]
        hi = durs.index(max(durs))
        head_name = first["tail"][hi]["short"]
        per_step = []
        for s in steps:
            ks = [k for k in s["tail"] if k["short"] == head_name]
            if ks:
                per_step.append(max(k["end"] - k["start"] for k in ks) / 1e6)
        head = {"name": head_name, "full_name": first["tail"][hi]["name"],
                "looks_like_gemm": bool(GEMM_RE.search(first["tail"][hi]["name"])),
                "ms_median": statistics.median(per_step) if per_step else None,
                "ms_spread": (max(per_step) - min(per_step)) if per_step else None,
                "steps": len(per_step), "index_in_tail": hi,
                "after_head": tail_names[hi + 1:], "before_head": tail_names[:hi]}
    step_ms = [(s["tail"][-1]["end"] if s["tail"] else s["graph"][-1]["end"]) / 1e6 -
               (s["graph"][0]["start"] if s["graph"] else s["tail"][0]["start"]) / 1e6
               for s in steps]
    gpu_busy_ms = [sum(k["end"] - k["start"] for k in s["graph"] + s["tail"]) / 1e6 for s in steps]
    return {
        "method": method,
        "steps_found": len(steps),
        "kernels_per_step": statistics.median(sizes),
        "kernels_per_step_all": sizes,
        "kernels_per_layer": L,
        "layer_kernels_total": (L or 0) * layers,
        "kernels_outside_layers": statistics.median(sizes) - (L or 0) * layers,
        "layer_sequence": [{"name": k["short"], "full_name": k["name"], "author": author(k["name"]),
                            "us": (k["end"] - k["start"]) / 1e3} for k in layer],
        "outside_layers_in_graph": outside_graph,
        "after_graph": tail_names,
        "output_head": head,
        "step_span_ms_median": statistics.median(step_ms),
        "gpu_busy_ms_median": statistics.median(gpu_busy_ms),
        "cublas_kernels_in_layer": sorted({k["name"] for k in layer if GEMM_RE.search(k["name"])}),
        "splitk_reduction_kernels_in_layer": [k["short"] for k in layer if SPLITK_RE.search(k["name"])],
        "copy_like_kernels_in_step": [n for n in body_names + tail_names if COPY_RE.search(n)],
        "triton_kernels_in_layer": [k["short"] for k in layer if k["short"].startswith("triton_")],
        "vllm_custom_kernels_in_layer": [k["short"] for k in layer
                                         if any(v in k["name"] for v in VLLM_EAGER_NAMES)],
        "attention_kernels_in_step": sorted({n for n in body_names + tail_names if ATTN_RE.search(n)}),
    }


def rope_candidates(layer_sequence):
    """Kernels between the first GEMM (QKV) and the KV-cache write: where the rotary
    embedding runs. A heuristic; the names are printed so the reader checks it."""
    names = [k["name"] for k in layer_sequence]
    try:
        g = next(i for i, n in enumerate(names) if GEMM_RE.search(n))
        c = next(i for i, n in enumerate(names) if "reshape_and_cache" in n)
    except StopIteration:
        return None
    between = [layer_sequence[i]["name"] for i in range(g + 1, c)]
    return {"count": len(between), "names": between}


def prefill_kernels(kernels, steps):
    """Kernels before the first decode step: the prefill pass in a one-generate capture."""
    if not steps:
        return []
    return kernels[:steps[0]["first_index"]]


def memops_in(memops, t0, t1):
    return [m for m in memops if t0 <= m["start"] < t1]
