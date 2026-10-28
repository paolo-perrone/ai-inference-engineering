#!/usr/bin/env python3
"""Tests for chapter 3's measurement code. No GPU, no vLLM, no torch needed.

  python3 test_run.py

Parser fixtures are written in the formats the real sources emit, each named beside it:
vLLM 0.30.0's streamed completion chunks and startup log lines, Nsight Systems 2026.3.2's
sqlite export tables and cuda_gpu_trace CSV columns. The numbers inside the fixtures are
made up for the parsers only; no test asserts a measured value.
"""
import contextlib
import io
import json
import os
import pathlib
import re
import sqlite3
import subprocess
import sys
import tempfile
import types
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import hf_worker  # noqa: E402
import logparse  # noqa: E402
import nsys_tools  # noqa: E402
import run  # noqa: E402
import stream_client  # noqa: E402
import vllm_session  # noqa: E402

DRAFT = pathlib.Path(os.environ.get(
    "CH3_DRAFT",
    "/Users/paoloap/Desktop/🕹️/3 - Cash 🪙/1 - Earn 💰/1 - Syntax Studio 💎/🎯 Projects/"
    "📕 Manning Book/📚 Chapters/ch03/ch3 - The forward pass and the kernels that run it.md"))


# ---------------------------------------------------------------- streaming chunks

def chunk(text, ids, running, prompt_ids=None, finish=None):
    """A chunk as vLLM 0.30.0 writes it: CompletionStreamResponse.model_dump_json(
    exclude_unset=True) with return_token_ids and continuous_usage_stats on
    (entrypoints/openai/completion/serving.py)."""
    choice = {"index": 0, "text": text, "logprobs": None, "finish_reason": finish,
              "stop_reason": None, "token_ids": ids}
    if prompt_ids is not None:
        choice["prompt_token_ids"] = prompt_ids
    body = {"id": "cmpl-abc", "object": "text_completion", "created": 1, "model": common.MODEL,
            "choices": [choice],
            "usage": {"prompt_tokens": 4, "total_tokens": 4 + running, "completion_tokens": running}}
    return f"data: {json.dumps(body)}\n\n".encode()


def usage_chunk(n):
    body = {"id": "cmpl-abc", "object": "text_completion", "created": 1, "model": common.MODEL,
            "choices": [], "usage": {"prompt_tokens": 4, "total_tokens": 4 + n, "completion_tokens": n}}
    return f"data: {json.dumps(body)}\n\n".encode()


class FakeClock:
    def __init__(self, times):
        self.times = list(times)

    def __call__(self):
        return self.times.pop(0)


class StreamTests(unittest.TestCase):
    def test_parse_sse_line(self):
        self.assertEqual(stream_client.parse_sse_line(b"data: [DONE]\n"), "DONE")
        self.assertIsNone(stream_client.parse_sse_line(b"\n"))
        self.assertIsNone(stream_client.parse_sse_line(b": keep-alive\n"))
        msg = stream_client.parse_sse_line(chunk("x", [7], 1))
        self.assertEqual(msg["choices"][0]["token_ids"], [7])

    def test_one_token_per_chunk(self):
        lines = [chunk("a", [1], 1, prompt_ids=[9, 9, 9, 9]), chunk("b", [2], 2),
                 chunk("c", [3], 3, finish="length"), usage_chunk(3), b"data: [DONE]\n\n"]
        # clock: one reading per data line (4 before DONE), then the end reading
        tl = stream_client.timeline_from_lines(lines, 0.0, FakeClock([0.10, 0.15, 0.20, 0.21, 0.22]))
        s = stream_client.summarize_timeline(tl)
        self.assertAlmostEqual(s["ttft_ms"], 100.0)
        self.assertAlmostEqual(s["e2e_ms"], 200.0)
        self.assertAlmostEqual(s["tpot_ms"], 50.0)
        self.assertEqual(s["tokens_per_chunk"], [1, 1, 1])
        self.assertEqual(s["completion_tokens"], 3)
        self.assertEqual(s["completion_tokens_usage"], 3)
        self.assertEqual(s["prompt_token_ids_len"], 4)
        self.assertEqual([round(g, 6) for g in s["itl_ms"]], [50.0, 50.0])

    def test_vllm_docs_case_itl_differs_from_tpot(self):
        """The vLLM docs' case the draft quotes: TTFT 100 ms, E2E 180 ms, 5 tokens, the middle
        3 in one chunk. Mean ITL 40 ms, TPOT 20 ms."""
        lines = [chunk("a", [1], 1), chunk("bcd", [2, 3, 4], 4), chunk("e", [5], 5),
                 usage_chunk(5), b"data: [DONE]\n\n"]
        tl = stream_client.timeline_from_lines(lines, 0.0, FakeClock([0.100, 0.140, 0.180, 0.19, 0.2]))
        s = stream_client.summarize_timeline(tl)
        self.assertAlmostEqual(s["tpot_ms"], 20.0)
        self.assertAlmostEqual(sum(s["itl_ms"]) / len(s["itl_ms"]), 40.0)
        self.assertEqual(s["tokens_per_chunk"], [1, 3, 1])

    def test_usage_difference_when_no_token_ids(self):
        msg = {"choices": [{"text": "ab"}], "usage": {"completion_tokens": 6}}
        self.assertEqual(stream_client.chunk_tokens(msg, 4), (None, 2, 6))

    def test_empty_chunk_is_not_a_token(self):
        lines = [chunk("", [], 0), chunk("a", [1], 1), chunk("b", [2], 2), b"data: [DONE]\n\n"]
        tl = stream_client.timeline_from_lines(lines, 0.0, FakeClock([0.05, 0.1, 0.2, 0.3]))
        self.assertEqual(len(tl["chunks"]), 2)

    def test_aggregate_pools_gaps(self):
        r1 = {"ttft_ms": 1.0, "decode_ms": 2.0, "e2e_ms": 3.0, "tpot_ms": 1.0, "itl_ms": [1.0, 1.0],
              "tokens_per_chunk": [1, 1, 1], "completion_tokens": 3, "prompt_tokens": 4}
        r2 = dict(r1, ttft_ms=3.0, tpot_ms=2.0, itl_ms=[4.0], tokens_per_chunk=[1, 2])
        a = stream_client.aggregate([r1, r2])
        self.assertEqual(a["ttft_ms"]["median"], 2.0)
        self.assertEqual(a["ttft_ms"]["spread"], 2.0)
        self.assertAlmostEqual(a["mean_itl_ms"], 2.0)
        self.assertEqual(a["tokens_per_chunk"]["max"], 2)

    def test_request_body(self):
        b = stream_client.request_body(common.MODEL, "hi", 180)
        self.assertEqual(b["temperature"], 0)
        self.assertTrue(b["ignore_eos"] and b["return_token_ids"] and b["cache_salt"])
        q = stream_client.request_body(common.MODEL, "hi", 8, greedy=False, ignore_eos=False)
        for k in ("temperature", "top_p", "top_k", "repetition_penalty", "ignore_eos"):
            self.assertNotIn(k, q)


# ---------------------------------------------------------------- vLLM log lines

# vllm/logger.py _FORMAT: "%(levelname)s %(asctime)s [%(fileinfo)s:%(lineno)d] %(message)s",
# messages from platforms/cuda.py:539 and config/model.py:1772, arguments filled the way the
# source fills them.
LOG = """\
INFO 10-27 12:00:01 [utils.py:300] non-default args: {'model': 'Qwen/Qwen2.5-7B-Instruct'}
(EngineCore pid=4242) INFO 10-27 12:00:09 [cuda.py:539] Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION', 'TURBOQUANT'].
WARNING 10-27 12:01:30 [model.py:1772] Default vLLM sampling parameters have been overridden by the model's `generation_config.json`: `{'repetition_penalty': 1.05, 'temperature': 0.7, 'top_k': 20, 'top_p': 0.8}`. If this is not intended, please relaunch vLLM instance with `--generation-config vllm`.
WARNING 10-27 12:01:31 [vllm.py:1548] Enforce eager set, disabling torch.compile and CUDAGraphs. This is equivalent to setting -cc.mode=none -cc.cudagraph_mode=none
"""


class LogTests(unittest.TestCase):
    def test_backend(self):
        b = logparse.attention_backend(LOG)
        self.assertEqual(b["backend"], "FLASH_ATTN")
        self.assertEqual(b["candidates"][1], "FLASHINFER")
        self.assertIn("[cuda.py:539]", b["line"])

    def test_override(self):
        o = logparse.sampling_override(LOG)
        self.assertEqual(o["source"], "the model's `generation_config.json`")
        self.assertEqual(o["params"]["top_k"], 20)
        self.assertIn("overridden", o["line"])

    def test_absent(self):
        self.assertIsNone(logparse.sampling_override("INFO nothing here"))
        self.assertIsNone(logparse.attention_backend(""))
        self.assertTrue(logparse.enforce_eager(LOG))
        self.assertFalse(logparse.enforce_eager("INFO started"))


# ---------------------------------------------------------------- Nsight traces

LAYER = ["triton_red_fused_add_rsqrt_0", "sm89_xmma_gemm_bf16_qkv", "triton_poi_fused_rope_1",
         "reshape_and_cache_flash_kernel", "flash_fwd_splitkv_kernel",
         "flash_fwd_splitkv_combine_kernel", "sm89_xmma_gemm_bf16_o",
         "triton_red_fused_add_rsqrt_2", "sm89_xmma_gemm_bf16_gateup",
         "triton_poi_fused_mul_silu_3", "sm89_xmma_gemm_bf16_down"]
HEAD = "sm89_xmma_gemm_bf16_head"
SAMPLE = ["argmax_kernel", "index_copy_kernel", "elementwise_copy_kernel"]


def synthetic_capture(steps=4, graph=True):
    """One prefill pass (no graph) and `steps` decode steps: a graph replay of embedding,
    28 layers and the final norm, then the head and sampling launched after it."""
    seq, t = [], 0

    def add(name, dur, node):
        nonlocal t
        seq.append((t, t + dur, node, name))
        t += dur + 10
    add("embedding_kernel", 50, None)
    for _ in range(28):
        for n in LAYER:
            if "combine" not in n:          # prefill runs no combine kernel in this fixture
                add(n, 100, None)
    add(HEAD, 3000, None)
    for n in SAMPLE:
        add(n, 20, None)
    node = 1
    for _ in range(steps):
        add("embedding_kernel", 50, node if graph else None)
        node += 1
        for _ in range(28):
            for n in LAYER:
                add(n, 300 if "gemm" in n else 20, node if graph else None)
                node += 1
        add("triton_red_fused_rsqrt_final", 20, node if graph else None)
        add(HEAD, 3600, None)
        for n in SAMPLE:
            add(n, 20, None)
    return seq


def write_nsys_sqlite(path, seq, with_graph_column=True):
    """The two tables reports/kernel_helper.py reads: CUPTI_ACTIVITY_KIND_KERNEL (start, end,
    demangledName, shortName as StringIds keys, graphNodeId) and StringIds (id, value)."""
    con = sqlite3.connect(path)
    cols = "start INTEGER, end INTEGER, deviceId INTEGER, streamId INTEGER, demangledName INTEGER, shortName INTEGER"
    if with_graph_column:
        cols += ", graphNodeId INTEGER"
    con.execute(f"CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL ({cols})")
    con.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
    ids = {}
    for s, e, node, name in seq:
        for v in (name, name + "(args)"):
            if v not in ids:
                ids[v] = len(ids) + 1
                con.execute("INSERT INTO StringIds VALUES (?, ?)", (ids[v], v))
        row = [s, e, 0, 7, ids[name + "(args)"], ids[name]]
        if with_graph_column:
            row.append(node)
        con.execute(f"INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES ({','.join('?' * len(row))})", row)
    con.commit()
    con.close()


class NsysTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def read(self, seq, graph_col=True):
        db = pathlib.Path(self.tmp.name) / "t.sqlite"
        if db.exists():
            db.unlink()
        write_nsys_sqlite(str(db), seq, graph_col)
        return nsys_tools.read_sqlite(db)

    def test_graph_capture(self):
        tr = self.read(synthetic_capture(4, graph=True))
        self.assertEqual(tr["kernels"][1]["short"], LAYER[0])
        self.assertTrue(tr["kernels"][1]["name"].endswith("(args)"))
        steps, method = nsys_tools.decode_steps(tr["kernels"])
        self.assertEqual(method, "cuda-graph runs")
        self.assertEqual(len(steps), 4)
        rep = nsys_tools.step_report(steps, method)
        self.assertEqual(rep["kernels_per_layer"], 11)
        self.assertEqual(rep["kernels_per_step"], 1 + 28 * 11 + 1 + 1 + 3)
        self.assertEqual(rep["kernels_outside_layers"], 6)
        self.assertEqual(rep["output_head"]["name"], HEAD)
        self.assertAlmostEqual(rep["output_head"]["ms_median"], 0.0036)
        self.assertEqual(rep["output_head"]["after_head"], SAMPLE)
        self.assertIn("flash_fwd_splitkv_combine_kernel", rep["attention_kernels_in_step"])
        rope = nsys_tools.rope_candidates(rep["layer_sequence"])
        self.assertEqual(rope["count"], 1)
        pre = nsys_tools.prefill_kernels(tr["kernels"], steps)
        self.assertNotIn("flash_fwd_splitkv_combine_kernel", {k["short"] for k in pre})

    def test_no_graph_column_uses_the_period(self):
        tr = self.read(synthetic_capture(5, graph=False), graph_col=False)
        self.assertTrue(all(k["graph_node"] is None for k in tr["kernels"]))
        steps, method = nsys_tools.decode_steps(tr["kernels"])
        self.assertIn("period", method)
        rep = nsys_tools.step_report(steps, method)
        self.assertEqual(rep["kernels_per_layer"], 11)
        self.assertEqual(rep["kernels_per_step"], 1 + 28 * 11 + 1 + 1 + 3)

    def test_find_period_prefers_coverage(self):
        names = ["p"] * 5 + (["a", "b", "c", "c", "c"] * 6)
        self.assertEqual(nsys_tools.find_period(names), (5, 6))
        self.assertEqual(nsys_tools.find_period(["x", "y"]), (None, 0))

    def test_layer_block(self):
        seq = ["emb"] + ["n", "g", "a"] * 28 + ["final"]
        self.assertEqual(nsys_tools.find_layer_block(seq), (3, 1))

    def test_gpu_trace_csv_both_header_styles(self):
        # reports/cuda_gpu_trace.py column order; the nsys binary renders units into names.
        for start, dur in (("Start (ns)", "Duration (ns)"), ("Start:ts_ns", "Duration:dur_ns")):
            text = (f'Processing [report.sqlite] with [cuda_gpu_trace.py]... \n\n'
                    f'"{start}","{dur}","CorrId","GrdX","GrdY","GrdZ","BlkX","BlkY","BlkZ",'
                    f'"Reg/Trd","StcSMem (MB)","DymSMem (MB)","Bytes (MB)","Throughput (MBps)",'
                    f'"SrcMemKd","DstMemKd","Device","Ctx","GreenCtx","Strm","Name"\n'
                    '1000,250,11,1,1,1,128,1,1,32,0.0,0.0,,,,,"NVIDIA L4 (0)",1,,7,"triton_poi_fused_mul_silu_0"\n'
                    '1300,40,12,,,,,,,,,,0.001,25.0,Device,Pageable,"NVIDIA L4 (0)",1,,7,"[CUDA memcpy Device-to-Host]"\n'
                    '1400,90,13,4,1,1,256,1,1,64,0.0,0.03,,,,,"NVIDIA L4 (0)",1,,7,"void flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128>>(Flash_fwd_params)"\n')
            tr = nsys_tools.read_gpu_trace_csv(text)
            self.assertEqual([k["name"] for k in tr["kernels"]],
                             ["triton_poi_fused_mul_silu_0",
                              "void flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128>>(Flash_fwd_params)"])
            self.assertEqual(tr["kernels"][0]["end"], 1250)
            self.assertEqual(len(tr["memops"]), 1)

    def test_commands(self):
        cmd = nsys_tools.profile_cmd("out/x", ["vllm", "bench", "latency"], "stop")
        self.assertIn("--cuda-graph-trace=node", cmd)
        self.assertIn("--capture-range=cudaProfilerApi", cmd)
        self.assertEqual(cmd[-3:], ["vllm", "bench", "latency"])
        whole = nsys_tools.profile_cmd("out/x", ["vllm"])
        self.assertFalse(any(c.startswith("--capture-range") for c in whole))


# ---------------------------------------------------------------- transformers profiler rows

def fe(name, start, kernels=(), parent=None):
    return types.SimpleNamespace(name=name, kernels=[types.SimpleNamespace(name=k) for k in kernels],
                                 cpu_parent=parent, time_range=types.SimpleNamespace(start=start))


class HFTests(unittest.TestCase):
    def test_kernel_rows_and_roles(self):
        layer = fe("ch3.layer.0", 0)
        norm = fe("ch3.input_norm", 1, parent=layer)
        attn = fe("ch3.attn", 5, parent=layer)
        events = [layer, norm, attn,
                  fe("aten::pow", 2, ["pow_kernel"], norm), fe("aten::mean", 3, ["reduce_kernel"], norm),
                  fe("aten::mm", 6, ["gemm"], attn), fe("aten::add", 9, ["add_kernel"], layer),
                  fe("aten::embedding", -1, ["gather"], None)]
        rows = hf_worker.kernel_rows(events)
        self.assertEqual([r["op"] for r in rows][0], "aten::embedding")
        per = hf_worker.per_layer(rows)
        self.assertEqual([p["role"] for p in per],
                         ["input_norm", "input_norm", "attn", "residual/other (layer body)"])
        self.assertEqual(per[-1]["kernel"], "add_kernel")

    def test_oom_search_brackets_the_length_to_256(self):
        real = hf_worker.fits
        try:
            for cap, ran, failed, want in [(7000, 4096, 8192, (6912, 7168)),
                                           (6144, 4096, 8192, (6144, 6400)),
                                           (20000, 8192, None, (19968, 20224)),
                                           (40000, 8192, None, (32768, None))]:
                hf_worker.fits = lambda m, t, n, cap=cap: n <= cap
                r = hf_worker.oom_search(None, None, ran, failed, 256)
                self.assertEqual((r["longest_ran"], r["oom_length"]), want)
        finally:
            hf_worker.fits = real


# ---------------------------------------------------------------- run.py and its contracts

class FakeTok:
    """Splits on characters: enough to check padded_ids' arithmetic."""
    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [ord(c) for c in text]}


class RunTests(unittest.TestCase):
    def test_list(self):
        p = subprocess.run([sys.executable, str(HERE / "run.py"), "--list"],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0)
        ids = p.stdout.split()
        self.assertEqual(sorted(ids), [f"m{i:02d}" for i in range(1, 17)])
        self.assertEqual(len(ids), len(set(ids)))

    def test_dry_run_all(self):
        p = subprocess.run([sys.executable, str(HERE / "run.py"), "all", "--dry-run"],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn('--compilation-config \'{"cudagraph_mode": "NONE"}\'', p.stdout)
        self.assertIn("--enforce-eager", p.stdout)
        self.assertIn("--cuda-graph-trace=node", p.stdout)
        self.assertNotIn("DRY-RUN FAILED", p.stdout)

    def test_configs_validate(self):
        self.assertEqual(vllm_session.validate_configs(), [])
        cmd = vllm_session.serve_cmd("qwen_defaults", 1)
        self.assertNotIn("--generation-config", cmd)
        self.assertIn("--generation-config", vllm_session.serve_cmd("default", 1))

    def test_padded_ids_exact(self):
        tok = FakeTok()
        n = len(common.mini_prompt())
        for L in (10, n, n + 5000):
            self.assertEqual(len(common.padded_ids(tok, L)), L)

    def test_a_failure_is_recorded_and_the_rest_still_run(self):
        with tempfile.TemporaryDirectory() as d:
            old = common.RESULTS
            common.RESULTS = pathlib.Path(d)
            saved = run.BY_ID["m03"]
            try:
                def boom(ctx):
                    raise RuntimeError("engine fell over")
                run.BY_ID["m03"] = (saved[0], boom) + saved[2:]
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = run.run_one("m03", dry=False, fresh=False)
                self.assertEqual(rc, 1)
                data = json.loads((pathlib.Path(d) / "m03.json").read_text())
                self.assertIn("engine fell over", data["error"])
                self.assertIn("run_log", data)
            finally:
                run.BY_ID["m03"] = saved
                common.RESULTS = old

    def test_book_floor(self):
        self.assertAlmostEqual(common.BOOK["decode_floor_ms"], 47.0)


# ---------------------------------------------------------------- fixtures.md covers the draft

FIX_ROW = re.compile(r"^\| *(\d+) *\| *(\d+) *\| *`\[FIXTURE: (.+?)\]` *\| *`(m\d\d)` *\| *`([^`]+)` *\|")


def fixture_rows():
    rows = []
    for line in (HERE / "fixtures.md").read_text(encoding="utf-8").splitlines():
        m = FIX_ROW.match(line)
        if m:
            rows.append({"line": int(m.group(2)), "text": m.group(3), "mid": m.group(4),
                         "fields": [f.strip() for f in m.group(5).split(", ")]})  # commas inside [] carry no space
    return rows


class FixtureTests(unittest.TestCase):
    def test_rows_name_real_subcommands_and_fields(self):
        rows = fixture_rows()
        self.assertGreater(len(rows), 0)
        for r in rows:
            self.assertIn(r["mid"], run.BY_ID, r)
            declared = run.BY_ID[r["mid"]][5]
            for f in r["fields"]:
                base = re.split(r"[\[.]", f, maxsplit=1)[0]
                self.assertIn(base, declared, f"{r['mid']} does not declare {base} ({r['text']})")

    @unittest.skipUnless(DRAFT.exists(), f"draft not found at {DRAFT}; set CH3_DRAFT")
    def test_every_placeholder_in_the_draft_is_mapped(self):
        found = []
        for i, line in enumerate(DRAFT.read_text(encoding="utf-8").splitlines(), 1):
            for m in re.finditer(r"\[FIXTURE: ([^\]]+)\]", line):
                found.append((i, m.group(1)))
        rows = fixture_rows()
        mapped = {}
        for r in rows:
            mapped[r["text"]] = mapped.get(r["text"], 0) + 1
        needed = {}
        for _, t in found:
            needed[t] = needed.get(t, 0) + 1
        unmapped = {t: n - mapped.get(t, 0) for t, n in needed.items() if mapped.get(t, 0) < n}
        self.assertEqual(unmapped, {}, "placeholders in the draft with no row in fixtures.md")
        moved = sorted({(i, t) for i, t in found} - {(r["line"], r["text"]) for r in rows})
        if moved:
            print(f"\n  note: {len(moved)} placeholder(s) moved lines since fixtures.md was written",
                  file=sys.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=1)
