#!/usr/bin/env python3
"""Chapter 7's measurements on one NVIDIA L4: every [FIXTURE] number in the chapter.

One subcommand per run or sweep. Each starts its own `vllm serve`, measures, stops the
engine and writes results/<id>.json: the numbers under named fields (median of the runs,
the spread, every run's value), the exact server and client commands, the card, driver,
CUDA, vLLM and PyTorch versions, and every input the draft leaves undefined.

  python run.py --list                 every subcommand id, in the order to run them
  python run.py runA                   one measurement
  python run.py runA --dry-run         commands, schedule and checks; no GPU, no network
  python run.py all                    every subcommand in order; a failure does not stop the rest
  python run.py summary                the computed numbers (cost, operating point, spread)

fixtures.md maps every [FIXTURE] placeholder in the draft to a subcommand and a field.
"""
import argparse
import datetime
import json
import os
import pathlib
import shlex
import shutil
import statistics
import subprocess
import sys
import threading
import time
import uuid

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
MINI = REPO_ROOT / "examples" / "mini-session"
sys.path.insert(0, str(MINI))
sys.path.insert(0, str(HERE))
import measure as ms  # noqa: E402  chapter 1's fixture: the prompt, the engine helpers
import ch7lib as lib  # noqa: E402

RESULTS = HERE / "results"
LOGS = HERE / "runs"
TRAFFIC = json.loads((HERE / "traffic.json").read_text(encoding="utf-8"))

ORDER = ["launches", "cold32k", "static", "runA", "runB", "runC", "runD", "admit128k",
         "sweep4k", "sweep32k", "sweepshared", "summary"]

# Every server flag this file passes, with where vLLM 0.30.0 defines it. --dry-run refuses
# a flag that is not here, so a flag typed from memory fails on the laptop, not the card.
FLAGS_FROM_SOURCE = {
    "--kv-cache-dtype": "vllm/config/cache.py:111 cache_dtype",
    "--max-model-len": "vllm/engine/arg_utils.py, ModelConfig.max_model_len",
    "--hf-overrides": "vllm/engine/arg_utils.py:940",
    "--max-num-seqs": "vllm/engine/arg_utils.py:1595",
    "--max-num-batched-tokens": "vllm/engine/arg_utils.py, SchedulerConfig.max_num_batched_tokens",
    "--long-prefill-token-threshold": "vllm/engine/arg_utils.py:1609, default 0 (scheduler.py:70)",
    "--scheduling-policy": "vllm/engine/arg_utils.py:1615, Literal['fcfs', 'priority'] (scheduler.py:22)",
    "--no-async-scheduling": "vllm/engine/arg_utils.py:1644, BooleanOptionalAction (:361)",
    "--enable-per-request-metrics": "vllm/entrypoints/launchers/cli_args.py:134",
    "--host": "vllm serve", "--port": "vllm serve",
}


def val(key):
    return TRAFFIC[key]["value"]


MODEL = val("model")
RUNS_DEFAULT = int(val("runs"))
BASE = ["--kv-cache-dtype", val("kv_cache_dtype")]
LEN4K = ["--max-model-len", "4096"]
LEN32K = ["--max-model-len", "32768"]
LEN128K = ["--max-model-len", "131072", "--hf-overrides", json.dumps(val("yarn_override"))]
RUN_FLAGS = {"A": BASE + LEN128K}
RUN_FLAGS["B"] = RUN_FLAGS["A"] + ["--max-num-batched-tokens", "8192",
                                   "--long-prefill-token-threshold", "2048"]
RUN_FLAGS["C"] = RUN_FLAGS["A"] + ["--scheduling-policy", "priority"]
RUN_FLAGS["D"] = RUN_FLAGS["B"] + ["--no-async-scheduling"]

# The traffic.json keys each subcommand reads, so its result names only its own inputs.
REPLAY_KEYS = ["seed", "sessions", "prefix_tokens", "tail_tokens", "followup_new_tokens",
               "cold_paste_tokens", "cold_paste_max_tokens", "followup_mean_gap_s",
               "replay_duration_s", "paste_at_s", "followups_carry_history", "completion_tokens"]
USES = {
    "launches": [], "cold32k": ["cold_paste_tokens"],
    "static": ["static_answer_lengths", "static_batch", "static_context",
               "continuous_max_num_seqs", "followup_new_tokens"],
    "runA": REPLAY_KEYS, "runB": REPLAY_KEYS, "runC": REPLAY_KEYS + ["runC_priorities"],
    "runD": REPLAY_KEYS,
    "admit128k": ["admit128k_prompt_tokens", "admit128k_stagger_s", "completion_tokens"],
    "sweep4k": ["sweep4k_points", "sweep4k_prompt_tokens", "sweep_requests_per_point",
                "tpot_p99_definition", "completion_tokens"],
    "sweep32k": ["sweep32k_points", "sweep32k_prompt_tokens", "sweep_requests_per_point",
                 "tpot_p99_definition", "completion_tokens"],
    "sweepshared": ["sweepshared_points", "prefix_tokens", "tail_tokens", "followup_new_tokens",
                    "sweep_requests_per_point", "tpot_p99_definition", "completion_tokens"],
    "summary": ["price_per_hour_usd", "tpot_target_p99_ms"],
}


def provisional(keys):
    return {k: TRAFFIC[k] for k in keys if str(TRAFFIC[k]["source"]).startswith("UNDEFINED")}


def rel(path):
    """A path relative to this folder when it is inside it, else absolute."""
    try:
        return str(path.relative_to(HERE))
    except ValueError:
        return str(path)


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ---------------------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------------------

class LaunchFailed(Exception):
    def __init__(self, info):
        super().__init__(info.get("reason", "launch failed"))
        self.info = info


class Engine:
    """`vllm serve` with prefix caching left at its default (on), per-request metrics on.

    Chapter 1's start_engine turns prefix caching off, which chapter 7 needs on, so this
    keeps its port and stop helpers and records a failed launch instead of exiting: an
    out-of-memory launch the draft quotes is a result.
    """

    def __init__(self, name, flags, env_extra=None, timeout=2400):
        self.name, self.flags, self.env_extra, self.timeout = name, list(flags), env_extra or {}, timeout
        self.exe = shutil.which("vllm") or "vllm"
        self.proc = self.log = self.url = None
        self.log_path = LOGS / f"{stamp()}-{name}-server.log"

    def command(self, port="<port>"):
        return [self.exe, "serve", MODEL, "--host", "127.0.0.1", "--port", str(port),
                "--enable-per-request-metrics"] + self.flags

    def log_text(self):
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def __enter__(self):
        import urllib.request
        if not shutil.which("vllm"):
            raise SystemExit("vllm is not on PATH; pip install vllm==0.30.0")
        LOGS.mkdir(exist_ok=True)
        port = ms._free_port()
        self.cmd = self.command(port)
        self.log = open(self.log_path, "w", encoding="utf-8")
        env = dict(os.environ, **self.env_extra)
        self.proc = subprocess.Popen(self.cmd, stdout=self.log, stderr=subprocess.STDOUT,
                                     start_new_session=True, env=env)
        self.url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                self.log.close()
                raise LaunchFailed(self._failure(f"vllm serve exited with code {self.proc.returncode}"))
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=5) as r:
                    if r.status == 200:
                        return self
            except OSError:
                pass
            time.sleep(2)
        ms.stop_engine(self.proc, self.log)
        raise LaunchFailed(self._failure(f"no /health within {self.timeout} s"))

    def _failure(self, reason):
        text = self.log_text()
        return {"reason": reason, "server_command": self.cmd, "server_log": str(self.log_path),
                "parsed": lib.parse_startup_log(text), "log_tail": text.splitlines()[-60:]}

    def __exit__(self, *a):
        ms.stop_engine(self.proc, self.log)


def environment(engine):
    """Chapter 1's run log (card, driver, CUDA, vLLM) plus PyTorch, read in vLLM's interpreter."""
    env = ms.environment(engine.url, engine.exe)
    py = ms._interpreter(engine.exe) or sys.executable
    out = ms._output([py, "-c", "import torch; print(torch.__version__)"]).splitlines()
    env["torch"] = out[-1] if out else None
    env["sources"]["torch"] = "torch.__version__ in vLLM's interpreter"
    return env


PROBE = r"""
import json, sys
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.usage.usage_lib import UsageContext
a = AsyncEngineArgs.add_cli_args(FlexibleArgumentParser()).parse_args(sys.argv[1:])
cfg = AsyncEngineArgs.from_cli_args(a).create_engine_config(UsageContext.OPENAI_API_SERVER)
s = cfg.scheduler_config
print("CH7PROBE " + json.dumps({"max_num_seqs": s.max_num_seqs,
    "max_num_batched_tokens": s.max_num_batched_tokens, "async_scheduling": s.async_scheduling,
    "enable_chunked_prefill": s.enable_chunked_prefill,
    "long_prefill_token_threshold": s.long_prefill_token_threshold, "policy": s.policy,
    "scheduler_reserve_full_isl": s.scheduler_reserve_full_isl,
    "kv_cache_dtype": cfg.cache_config.cache_dtype,
    "enable_prefix_caching": cfg.cache_config.enable_prefix_caching,
    "disable_cascade_attn": cfg.model_config.disable_cascade_attn}))
"""


def probe_config(flags):
    """The scheduler values vLLM resolves for these flags, read from its own config objects.

    vLLM 0.30.0 logs max_num_batched_tokens at INFO (scheduler.py:288), max_num_seqs only at
    DEBUG (arg_utils.py:2989) and async_scheduling never when it turns it on by default
    (vllm.py:1438), so the resolved values come from create_engine_config() with the usage
    context `vllm serve` uses, run in vLLM's own interpreter before the server starts.
    """
    exe = shutil.which("vllm")
    py = (ms._interpreter(exe) if exe else None) or sys.executable
    cmd = [py, "-c", PROBE, "--model", MODEL] + flags
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except (OSError, subprocess.SubprocessError) as e:
        return {"error": str(e), "command": cmd}
    for line in (p.stdout or "").splitlines():
        if line.startswith("CH7PROBE "):
            return {**json.loads(line[len("CH7PROBE "):]),
                    "source": "create_engine_config(UsageContext.OPENAI_API_SERVER)"}
    return {"error": (p.stderr or "")[-2000:], "returncode": p.returncode}


class Tokens:
    """Exact-length prompts as token ids, built from the mini-session's own text.

    The server's /tokenize turns chapter 1's prompt (the repo plus the task) into ids once;
    every prompt here is that text tiled to an exact length behind a short header. Headers
    make prompts differ from their first block, so prefix caching shares exactly what it
    should and nothing more.
    """

    def __init__(self, url):
        self.url = url
        self.base = self.ids(ms.build_prompt())
        self._h = {}

    def ids(self, text):
        r = lib.post_json(self.url + "/tokenize", {"model": MODEL, "prompt": text,
                                                   "add_special_tokens": False})
        return r["tokens"]

    def h(self, text):
        if text not in self._h:
            self._h[text] = self.ids(text)
        return self._h[text]

    def make(self, n, header, rotate=0):
        return lib.tile(self.base, n, rotate=rotate, header=self.h(header))


def scrape(url):
    return lib.parse_prometheus(lib.get_text(url + "/metrics"))


def wait_idle(url, timeout=900):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        p = scrape(url)
        if not lib.metric_value(p, lib.M_RUNNING) and not lib.metric_value(p, lib.M_WAITING):
            return True
        time.sleep(0.5)
    return False


def counter_delta(before, after, name):
    a, b = lib.metric_value(after, name), lib.metric_value(before, name)
    return None if a is None or b is None else a - b


def concurrent(jobs):
    """Run (callable) jobs in threads at once; return their results in order."""
    out = [None] * len(jobs)

    def go(i, f):
        out[i] = f()
    ts = [threading.Thread(target=go, args=(i, f)) for i, f in enumerate(jobs)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return out


def slim(rec):
    """A request as the result file keeps it: numbers, not every chunk's timestamp."""
    n = lib.request_numbers(rec)
    sm = rec.get("server_metrics") or {}
    return {"tag": rec.get("tag"), "prompt_tokens_sent": rec["prompt_tokens_sent"],
            "engine_prompt_tokens": (rec.get("usage") or {}).get("prompt_tokens"),
            "tokens": rec["tokens"], "ttft_s": n["ttft"], "e2e_s": n["e2e"],
            "tpot_ms": None if n["tpot"] is None else n["tpot"] * 1000,
            "server_queue_time_ms": sm.get("queue_time_ms"),
            "server_ttft_after_scheduling_ms": sm.get("time_to_first_token_ms"),
            "error": rec.get("error")}


def server_queue_s(rec):
    """The server's own queue time for one request, in seconds, or None if it sent none."""
    q = (rec.get("server_metrics") or {}).get("queue_time_ms")
    return None if q is None else q / 1000


def ms_(x):
    return None if x is None else x * 1000


# ---------------------------------------------------------------------------------------
# Result files
# ---------------------------------------------------------------------------------------

def new_result(name):
    return {"subcommand": name, "status": "running", "started_utc": stamp(),
            "client_command": [sys.executable, str(pathlib.Path(__file__).name)] + sys.argv[1:],
            "server_commands": [], "server_logs": [], "environment": None,
            "provisional_inputs": provisional(USES.get(name, [])),
            "fields": {}, "per_run": [], "failures": [], "notes": []}


def write_result(res):
    res["finished_utc"] = stamp()
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{res['subcommand']}.json"
    path.write_text(json.dumps(res, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"wrote {rel(path)} ({res['status']})")
    return 0 if res["status"] == "ok" else 1


def attach(res, eng):
    res["server_commands"].append(eng.cmd)
    res["server_logs"].append(rel(eng.log_path))
    if res["environment"] is None:
        res["environment"] = environment(eng)


def banner(name):
    p = provisional(USES.get(name, []))
    if p:
        print(f"[{name}] PROVISIONAL INPUTS, not defined by the draft (see traffic.json):")
        for k, v in p.items():
            print(f"  {k} = {json.dumps(v['value'])}  ({v['source'][len('UNDEFINED: '):]})")


# ---------------------------------------------------------------------------------------
# launches: resolved defaults, attention backend, pool size, maximum concurrency
# ---------------------------------------------------------------------------------------

LAUNCHES = [("4k", BASE + LEN4K), ("32k", BASE + LEN32K), ("128k", RUN_FLAGS["A"])]


def cmd_launches(dry):
    if dry:
        for tag, flags in LAUNCHES:
            print(f"[launch {tag}] VLLM_LOGGING_LEVEL=DEBUG " + shlex.join(Engine(tag, flags).command()))
            print(f"[launch {tag}] probe: create_engine_config with the same engine flags")
        return 0
    res = new_result("launches")
    ok = True
    for tag, flags in LAUNCHES:
        entry = {"config_probe": probe_config(flags)}
        eng = Engine(f"launch-{tag}", flags, env_extra={"VLLM_LOGGING_LEVEL": "DEBUG"})
        try:
            with eng:
                attach(res, eng)
                entry.update(lib.parse_startup_log(eng.log_text()))
                entry["launched"] = True
        except LaunchFailed as e:
            entry.update({"launched": False, **e.info})
            res["server_commands"].append(e.info["server_command"])
            ok = ok and tag == "128k"   # a refused 128k launch is a result the draft can quote
        res["fields"][tag] = entry
    res["notes"].append("max_concurrency_x is the 'Maximum concurrency for N tokens per request' "
                        "line; available_kv_cache_gib is the pool in GiB as vLLM prints it.")
    res["status"] = "ok" if ok else "failed"
    return write_result(res)


# ---------------------------------------------------------------------------------------
# cold32k: one cold 32k prompt alone on an idle server
# ---------------------------------------------------------------------------------------

def cmd_cold32k(dry, runs):
    n = int(val("cold_paste_tokens"))
    if dry:
        print("[cold32k] " + shlex.join(Engine("A", RUN_FLAGS["A"]).command()))
        print(f"[cold32k] 1 discarded warm-up, then {runs} x one {n:,}-token prompt with a fresh "
              "cache_salt, max_tokens 1, one at a time on an idle server; TTFT each")
        return 0
    res = new_result("cold32k")
    try:
        with Engine("cold32k", RUN_FLAGS["A"]) as eng:
            attach(res, eng)
            tok = Tokens(eng.url)
            lib.stream_request(eng.url, MODEL, tok.base, 1)
            for r in range(runs):
                wait_idle(eng.url)
                rec = lib.stream_request(eng.url, MODEL, tok.make(n, f"Pasted files, cold run {r}:\n", 11 * r + 3),
                                         1, cache_salt=uuid.uuid4().hex, tag=f"cold-{r}")
                res["per_run"].append(slim(rec))
        res["fields"]["cold_32k_ttft_s"] = lib.across_runs([p["ttft_s"] for p in res["per_run"]])
        res["status"] = "ok" if all(not p["error"] for p in res["per_run"]) else "failed"
    except LaunchFailed as e:
        res["failures"].append(e.info)
        res["status"] = "failed"
    return write_result(res)


# ---------------------------------------------------------------------------------------
# static: client-side static batches of four against vLLM's continuous batching
# ---------------------------------------------------------------------------------------

def cmd_static(dry, runs):
    lengths = [int(x) for x in val("static_answer_lengths")]
    b = int(val("static_batch"))
    flags = BASE + LEN4K + ["--max-num-seqs", str(val("continuous_max_num_seqs"))]
    if dry:
        print("[static] " + shlex.join(Engine("static", flags).command()))
        groups = [lengths[i:i + b] for i in range(0, len(lengths), b)]
        print(f"[static] static: batches {groups}, each sent together, the next only after its slowest ends")
        print(f"[static] continuous: all {len(lengths)} sent together, answers {lengths}")
        print(f"[static] {runs} runs of each, order alternating; prompt = mini-session prompt + "
              f"{val('followup_new_tokens')} new tokens")
        return 0
    res = new_result("static")
    try:
        with Engine("static", flags) as eng:
            attach(res, eng)
            tok = Tokens(eng.url)
            lib.stream_request(eng.url, MODEL, tok.base, 1)
            nf = int(val("followup_new_tokens"))

            def prompts(r, mode):
                return [tok.base + tok.make(nf, f"Follow-up {mode} {r}.{i}:\n", 7 * i + r)
                        for i in range(len(lengths))]

            def one(mode, r):
                wait_idle(eng.url)
                ps = prompts(r, mode)
                jobs = [lambda p=p, n=n, i=i: lib.stream_request(eng.url, MODEL, p, n, tag=f"{mode}-{i}")
                        for i, (p, n) in enumerate(zip(ps, lengths))]
                t0 = time.monotonic()
                recs = []
                if mode == "static":
                    for i in range(0, len(jobs), b):
                        recs += concurrent(jobs[i:i + b])
                else:
                    recs = concurrent(jobs)
                end = max(x["chunk_times"][-1] for x in recs if x["chunk_times"])
                toks = sum(x["tokens"] for x in recs)
                return {"mode": mode, "run": r, "completion_time_s": end - t0,
                        "tokens": toks, "tokens_per_s": toks / (end - t0),
                        "errors": [x["error"] for x in recs if x["error"]],
                        "requests": [slim(x) for x in recs]}

            for r in range(runs):
                order = ("static", "continuous") if r % 2 == 0 else ("continuous", "static")
                for mode in order:
                    res["per_run"].append(one(mode, r))
        names = {"static": ("static_completion_time_s", "static_tokens_per_s"),
                 "continuous": ("continuous_completion_time_s", "continuous_tokens_per_s")}
        for mode, (t_key, s_key) in names.items():
            rows = [p for p in res["per_run"] if p["mode"] == mode]
            res["fields"][t_key] = lib.across_runs([p["completion_time_s"] for p in rows])
            res["fields"][s_key] = lib.across_runs([p["tokens_per_s"] for p in rows])
        res["status"] = "ok" if not any(p["errors"] for p in res["per_run"]) else "failed"
    except LaunchFailed as e:
        res["failures"].append(e.info)
        res["status"] = "failed"
    return write_result(res)


# ---------------------------------------------------------------------------------------
# runA to runD: the team replay
# ---------------------------------------------------------------------------------------

def replay_plan():
    sched = lib.arrival_schedule(TRAFFIC, int(val("seed")))
    P, T, F = int(val("prefix_tokens")), int(val("tail_tokens")), int(val("followup_new_tokens"))
    return sched, P, T, F


def check_lengths():
    """Every prompt plus its answer has to fit the launch's --max-model-len."""
    P, T, F, C = (int(val(k)) for k in ("prefix_tokens", "tail_tokens", "followup_new_tokens",
                                        "completion_tokens"))
    problems = []
    for what, n, cap in [("follow-up", P + T + F + C, 131072),
                         ("cold paste", int(val("cold_paste_tokens")) + int(val("cold_paste_max_tokens")), 131072),
                         ("128k admission", int(val("admit128k_prompt_tokens")) + C, 131072),
                         ("32k sweep", int(val("sweep32k_prompt_tokens")) + C, 32768),
                         ("4k sweep", int(val("sweep4k_prompt_tokens")) + C, 4096)]:
        if n > cap:
            problems.append(f"{what}: {n:,} tokens with its answer exceeds --max-model-len {cap:,}")
    return problems


def cmd_replay(label, dry, runs):
    name = f"run{label}"
    sched, P, T, F = replay_plan()
    prio = val("runC_priorities") if label == "C" else None
    if dry:
        print(f"[{name}] " + shlex.join(Engine(name, RUN_FLAGS[label]).command()))
        print(f"[{name}] warm: {val('sessions')} sessions x (prefix {P:,} + tail {T:,}) tokens, max_tokens 1")
        print(f"[{name}] each follow-up: prefix + its session's tail + {F} new tokens, "
              f"{val('completion_tokens')}-token answer; cold paste: "
              f"{int(val('cold_paste_tokens')):,} tokens, fresh cache_salt, {val('cold_paste_max_tokens')}-token answer"
              + (f"; priorities {prio}" if prio else ""))
        print(f"[{name}] arrival schedule (seed {val('seed')}), same in each of {runs} runs:")
        for t, kind, s, i in sched:
            print(f"    t={t:7.3f}s  {kind:<8} " + ("" if s is None else f"session {s} #{i}"))
        nf = sum(1 for e in sched if e[1] == "followup")
        print(f"[{name}] {nf} follow-ups and 1 paste per run, {nf * runs} follow-ups in all")
        return 0
    res = new_result(name)
    res["fields"]["arrival_schedule"] = [list(e) for e in sched]
    try:
        if label == "A":
            res["fields"]["resolved_scheduler"] = probe_config(RUN_FLAGS["A"])
        with Engine(name, RUN_FLAGS[label]) as eng:
            attach(res, eng)
            if label == "A":
                info = lib.parse_startup_log(eng.log_text())
                res["fields"]["resolved_scheduler"]["log_chunked_prefill_max_num_batched_tokens"] = \
                    info["chunked_prefill_max_num_batched_tokens"]
            tok = Tokens(eng.url)
            prefix = tok.make(P, "Shared repository context:\n", 0)
            tails = [tok.make(T, f"Session {s} working notes:\n", 97 * (s + 1))
                     for s in range(int(val("sessions")))]
            res["fields"]["warm_ttft_s"] = [slim(lib.stream_request(eng.url, MODEL, prefix + tails[s], 1,
                                                                    tag=f"warm-{s}"))["ttft_s"]
                                            for s in range(len(tails))]
            for r in range(runs):
                res["per_run"].append(replay_once(eng, tok, sched, prefix, tails, r, prio))
        per = res["per_run"]
        for key in ("followup_ttft_p50_s", "followup_ttft_p99_s", "followup_queue_time_p50_s",
                    "followup_queue_time_p50_s_from_histogram", "total_output_tokens_per_s",
                    "followup_itl_p50_ms", "followup_itl_p99_ms", "followup_tpot_p50_ms",
                    "paste_ttft_s", "prefix_cache_hit_rate", "preemptions", "max_waiting",
                    "max_kv_cache_usage"):
            res["fields"][key] = lib.across_runs([p[key] for p in per])
        res["status"] = "ok" if not any(p["errors"] for p in per) else "failed"
    except LaunchFailed as e:
        res["failures"].append(e.info)
        res["status"] = "failed"
    return write_result(res)


def replay_once(eng, tok, sched, prefix, tails, r, prio):
    wait_idle(eng.url)
    F = int(val("followup_new_tokens"))
    jobs = []
    for t, kind, s, i in sched:
        if kind == "paste":
            ids = tok.make(int(val("cold_paste_tokens")), f"Pasted files, run {r}:\n", 13 * r + 5)
            jobs.append((t, dict(prompt_ids=ids, max_tokens=int(val("cold_paste_max_tokens")),
                                 cache_salt=uuid.uuid4().hex, tag="paste",
                                 priority=prio["paste"] if prio else None)))
        else:
            ids = prefix + tails[s] + tok.make(F, f"Follow-up {r}.{s}.{i}:\n", 31 * i + 7 * s + r)
            jobs.append((t, dict(prompt_ids=ids, max_tokens=int(val("completion_tokens")),
                                 tag=f"followup-{s}-{i}",
                                 priority=prio["followup"] if prio else None)))
    before = scrape(eng.url)
    recs, threads = [None] * len(jobs), []

    def go(k, kw):
        recs[k] = lib.stream_request(eng.url, MODEL, **kw)
    with lib.Poller(eng.url) as poll:
        t0 = time.monotonic() + 0.5
        for k, (t, kw) in enumerate(jobs):
            time.sleep(max(0.0, t0 + t - time.monotonic()))
            th = threading.Thread(target=go, args=(k, kw))
            th.start()
            threads.append(th)
        for th in threads:
            th.join()
    after = scrape(eng.url)
    fus = [x for x in recs if x["tag"].startswith("followup") and not x["error"]]
    paste = next(x for x in recs if x["tag"] == "paste")
    nums = [lib.request_numbers(x) for x in fus]
    itl = [g for n in nums for g in n["itl"]]
    qs = [server_queue_s(x) for x in fus]
    live = [x for x in recs if x["chunk_times"]]
    span = max(x["chunk_times"][-1] for x in live) - min(x["send_t"] for x in live) if live else None
    q, h = counter_delta(before, after, lib.M_PREFIX_Q), counter_delta(before, after, lib.M_PREFIX_H)
    t_start, t_end = t0, time.monotonic()
    return {
        "run": r,
        "followup_ttft_p50_s": lib.percentile([n["ttft"] for n in nums], 50),
        "followup_ttft_p99_s": lib.percentile([n["ttft"] for n in nums], 99),
        "followup_queue_time_p50_s": lib.percentile(qs, 50),
        "followup_queue_time_p50_s_from_histogram": lib.histogram_quantile(
            0.5, lib.histogram_delta(lib.histogram(before, lib.M_QUEUE), lib.histogram(after, lib.M_QUEUE))),
        "total_output_tokens_per_s": sum(x["tokens"] for x in live) / span if span else None,
        "followup_itl_p50_ms": ms_(lib.percentile(itl, 50)),
        "followup_itl_p99_ms": ms_(lib.percentile(itl, 99)),
        "followup_tpot_p50_ms": ms_(lib.percentile([n["tpot"] for n in nums], 50)),
        "paste_ttft_s": lib.request_numbers(paste)["ttft"],
        "prefix_cache_hit_rate": (h / q) if q else None,
        "preemptions": counter_delta(before, after, lib.M_PREEMPT),
        "max_waiting": max(poll.between(t_start, t_end, "waiting"), default=None),
        "max_kv_cache_usage": max(poll.between(t_start, t_end, "kv_usage"), default=None),
        "errors": [x["error"] for x in recs if x["error"]],
        "requests": [slim(x) for x in recs],
    }


# ---------------------------------------------------------------------------------------
# admit128k: two 128k requests sent together
# ---------------------------------------------------------------------------------------

def cmd_admit128k(dry, runs):
    n, gap = int(val("admit128k_prompt_tokens")), float(val("admit128k_stagger_s"))
    if dry:
        print("[admit128k] " + shlex.join(Engine("A", RUN_FLAGS["A"]).command()))
        print(f"[admit128k] {runs} runs: two {n:,}-token prompts, fresh cache_salt each, the second "
              f"{gap} s after the first; /metrics polled every 0.25 s")
        return 0
    res = new_result("admit128k")
    try:
        with Engine("admit128k", RUN_FLAGS["A"]) as eng:
            attach(res, eng)
            tok = Tokens(eng.url)
            lib.stream_request(eng.url, MODEL, tok.base, 1)
            C = int(val("completion_tokens"))
            for r in range(runs):
                wait_idle(eng.url)
                p1 = tok.make(n, f"Session A, run {r}:\n", 17 * r + 1)
                p2 = tok.make(n, f"Session B, run {r}:\n", 17 * r + 9)
                out = [None, None]

                def go(k, p):
                    out[k] = lib.stream_request(eng.url, MODEL, p, C, cache_salt=uuid.uuid4().hex,
                                                tag=f"128k-{k + 1}")
                with lib.Poller(eng.url) as poll:
                    a = threading.Thread(target=go, args=(0, p1))
                    a.start()
                    time.sleep(gap)
                    b = threading.Thread(target=go, args=(1, p2))
                    b.start()
                    a.join()
                    b.join()
                first, second = out
                t_first_token = first["chunk_times"][0] if first["chunk_times"] else time.monotonic()
                res["per_run"].append({
                    "run": r, "first": slim(first), "second": slim(second),
                    "second_queue_time_s": server_queue_s(second),
                    "max_waiting_while_first_prefills": max(
                        poll.between(first["send_t"], t_first_token, "waiting"), default=None),
                    "max_running_while_first_prefills": max(
                        poll.between(first["send_t"], t_first_token, "running"), default=None),
                    "errors": [x["error"] for x in out if x["error"]]})
        per = res["per_run"]
        for key in ("second_queue_time_s", "max_waiting_while_first_prefills",
                    "max_running_while_first_prefills"):
            res["fields"][key] = lib.across_runs([p[key] for p in per])
        res["fields"]["first_ttft_s"] = lib.across_runs([p["first"]["ttft_s"] for p in per])
        res["fields"]["second_ttft_s"] = lib.across_runs([p["second"]["ttft_s"] for p in per])
        res["status"] = "ok" if not any(p["errors"] for p in per) else "failed"
    except LaunchFailed as e:
        res["failures"].append(e.info)
        res["status"] = "failed"
    return write_result(res)


# ---------------------------------------------------------------------------------------
# Sweeps over --max-num-seqs
# ---------------------------------------------------------------------------------------

SWEEPS = {
    "sweep4k": {"points": "sweep4k_points", "flags": BASE + LEN4K},
    "sweep32k": {"points": "sweep32k_points", "flags": BASE + LEN32K},
    "sweepshared": {"points": "sweepshared_points", "flags": BASE + LEN128K},
}


def sweep_prompts(name, tok, n, r):
    """The n prompts one run of a sweep point sends. r < 0 is the warm-up run."""
    if name == "sweepshared":
        P, T, F = int(val("prefix_tokens")), int(val("tail_tokens")), int(val("followup_new_tokens"))
        prefix = tok.make(P, "Shared repository context:\n", 0)
        return [prefix + tok.make(T, f"Session {s} working notes:\n", 97 * (s + 1))
                + tok.make(F, f"Follow-up {r}.{s}:\n", 31 * s + r + 2) for s in range(n)]
    L = int(val("sweep4k_prompt_tokens" if name == "sweep4k" else "sweep32k_prompt_tokens"))
    return [tok.make(L, f"Request {i}:\n", 101 * i) for i in range(n)]


def warm_prompts(name, tok, n):
    if name == "sweepshared":
        P, T = int(val("prefix_tokens")), int(val("tail_tokens"))
        prefix = tok.make(P, "Shared repository context:\n", 0)
        return [prefix + tok.make(T, f"Session {s} working notes:\n", 97 * (s + 1)) for s in range(n)]
    return sweep_prompts(name, tok, n, 0)


def cmd_sweep(name, dry, runs):
    points = [int(x) for x in val(SWEEPS[name]["points"])]
    flags = SWEEPS[name]["flags"]
    C = int(val("completion_tokens"))
    if dry:
        for n in points:
            print(f"[{name} {n}] " + shlex.join(Engine(name, flags + ["--max-num-seqs", str(n)]).command()))
        print(f"[{name}] per point: warm each prompt (max_tokens 1), 1 discarded run, then {runs} runs "
              f"of max-num-seqs prompts sent together, {C}-token answers")
        return 0
    res = new_result(name)
    price = float(val("price_per_hour_usd"))
    ok = True
    for n in points:
        point = {"max_num_seqs": n, "runs": []}
        try:
            with Engine(f"{name}-{n}", flags + ["--max-num-seqs", str(n)]) as eng:
                attach(res, eng)
                point["startup"] = lib.parse_startup_log(eng.log_text())
                tok = Tokens(eng.url)
                for p in warm_prompts(name, tok, n):
                    lib.stream_request(eng.url, MODEL, p, 1, tag="warm")
                for r in [-1] + list(range(runs)):
                    row = sweep_once(eng, sweep_prompts(name, tok, n, r), C, r)
                    if r >= 0:
                        point["runs"].append(row)
        except LaunchFailed as e:
            point["launch_failure"] = e.info
            res["failures"].append(e.info)
            ok = False
        rs = point["runs"]
        agg = {k: lib.across_runs([x[k] for x in rs]) for k in
               ("tpot_p50_ms", "tpot_p99_ms", "tokens_per_s", "answers_per_s", "ttft_p50_s",
                "running_batch_max", "running_batch_median", "max_kv_cache_usage",
                "preemptions", "prefix_cache_hit_rate")}
        agg["tpot_p99_ms_pooled"] = ms_(lib.percentile(
            [q["tpot_ms"] / 1000 for x in rs for q in x["requests"] if q["tpot_ms"] is not None], 99))
        tps = agg["tokens_per_s"]["median"]
        agg["dollars_per_million_output_tokens"] = lib.dollars_per_million(price, tps)
        agg["throughput_methods"] = sorted({x["throughput_method"] for x in rs})
        point["fields"] = agg
        res["fields"][str(n)] = agg
        res["per_run"].append(point)
        if any(x["errors"] for x in rs):
            ok = False
    res["status"] = "ok" if ok else "failed"
    return write_result(res)


def sweep_once(eng, prompts, C, r):
    wait_idle(eng.url)
    before = scrape(eng.url)
    with lib.Poller(eng.url) as poll:
        t0 = time.monotonic()
        recs = concurrent([lambda p=p, i=i: lib.stream_request(eng.url, MODEL, p, C, tag=f"s{i}")
                           for i, p in enumerate(prompts)])
        t1 = time.monotonic()
    after = scrape(eng.url)
    ok = [x for x in recs if not x["error"] and x["chunk_times"]]
    nums = [lib.request_numbers(x) for x in ok]
    st = lib.steady_window_throughput(ok)
    running = poll.between(t0, t1, "running")
    q, h = counter_delta(before, after, lib.M_PREFIX_Q), counter_delta(before, after, lib.M_PREFIX_H)
    last = max((x["chunk_times"][-1] for x in ok), default=None)
    return {"run": r,
            "tpot_p50_ms": ms_(lib.percentile([n["tpot"] for n in nums], 50)),
            "tpot_p99_ms": ms_(lib.percentile([n["tpot"] for n in nums], 99)),
            "ttft_p50_s": lib.percentile([n["ttft"] for n in nums], 50),
            "tokens_per_s": st["tokens_per_s"], "throughput_method": st["method"],
            "answers_per_s": len(ok) / (last - t0) if last else None,
            "running_batch_max": max(running, default=None),
            "running_batch_median": statistics.median(running) if running else None,
            "max_kv_cache_usage": max(poll.between(t0, t1, "kv_usage"), default=None),
            "preemptions": counter_delta(before, after, lib.M_PREEMPT),
            "prefix_cache_hit_rate": (h / q) if q else None,
            "errors": [x["error"] for x in recs if x["error"]],
            "requests": [slim(x) for x in recs]}


# ---------------------------------------------------------------------------------------
# summary: the numbers computed from the runs, not measured separately
# ---------------------------------------------------------------------------------------

def load(name):
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def median_of(res, *path):
    x = res
    for k in path:
        if not isinstance(x, dict) or k not in x:
            return None
        x = x[k]
    return x.get("median") if isinstance(x, dict) else x


def operating_point(sweep_fields, target_ms):
    """The largest --max-num-seqs whose TPOT p99 (median of the runs) holds the target."""
    best = None
    for k in sorted(sweep_fields, key=int):
        p99 = median_of(sweep_fields, k, "tpot_p99_ms")
        if p99 is not None and p99 <= target_ms:
            best = int(k)
    return best


def largest_spread(results):
    """The largest run-to-run spread in any across_runs field, absolute and as a share of its median."""
    best = None

    def walk(x, path):
        nonlocal best
        if isinstance(x, dict):
            if {"median", "spread", "values"} <= set(x) and x["median"] and x["spread"] is not None:
                rel = x["spread"] / abs(x["median"])
                if best is None or rel > best["relative"]:
                    best = {"field": "/".join(path), "median": x["median"], "spread": x["spread"],
                            "relative": round(rel, 4)}
                return
            for k, v in x.items():
                walk(v, path + [k])
    for name, res in results.items():
        walk(res.get("fields", {}), [name])
    return best


def cmd_summary(dry):
    names = [n for n in ORDER if n != "summary"]
    if dry:
        print("[summary] reads " + ", ".join(f"results/{n}.json" for n in names) + "; no GPU")
        return 0
    results = {n: r for n in names if (r := load(n))}
    price, target = float(val("price_per_hour_usd")), float(val("tpot_target_p99_ms"))
    res = new_result("summary")
    res["environment"] = {n: r.get("environment") for n, r in results.items()}
    f = res["fields"]
    shared = (results.get("sweepshared") or {}).get("fields", {})
    op = operating_point(shared, target) if shared else None
    f["operating_point_max_num_seqs"] = op
    f["tpot_target_p99_ms"] = target
    f["dollars_per_million_by_point"] = {k: v.get("dollars_per_million_output_tokens")
                                         for k, v in shared.items()}
    f["dollars_per_million_at_operating_point"] = (shared.get(str(op), {}).get(
        "dollars_per_million_output_tokens") if op else None)
    f["dollars_per_million_at_batch_1"] = shared.get("1", {}).get("dollars_per_million_output_tokens")
    f["answers_per_s_at_operating_point"] = median_of(shared, str(op), "answers_per_s") if op else None
    f["dollars_per_million_at_operating_point_card_idle_half_the_day"] = (
        2 * f["dollars_per_million_at_operating_point"]
        if f["dollars_per_million_at_operating_point"] else None)
    f["idle_half_day_method"] = "arithmetic: the operating point's figure times 2, not measured"
    s4 = (results.get("sweep4k") or {}).get("fields", {})
    f["dollars_per_million_4k_at_32"] = s4.get("32", {}).get("dollars_per_million_output_tokens")
    f["figure_7_11_tokens_per_s"] = {
        "shared_prefix_batch_1": median_of(shared, "1", "tokens_per_s"),
        "shared_prefix_operating_point": median_of(shared, str(op), "tokens_per_s") if op else None,
        "4k_at_32": median_of(s4, "32", "tokens_per_s")}
    f["largest_run_to_run_spread"] = largest_spread(results)
    f["baseline_fields"] = {
        "ch7_followup_ttft_p99_s_runA": median_of(results.get("runA") or {}, "fields", "followup_ttft_p99_s"),
        "ch7_followup_ttft_p99_s_runB": median_of(results.get("runB") or {}, "fields", "followup_ttft_p99_s"),
        "ch7_dollars_per_million_at_operating_point": f["dollars_per_million_at_operating_point"]}
    res["notes"].append("dollars per million = price_per_hour x 1e6 / (tokens_per_s x 3600), "
                        "tokens_per_s the median of the runs at each point")
    res["status"] = "ok" if results else "failed"
    missing = [n for n in names if n not in results]
    if missing:
        res["notes"].append("missing results: " + ", ".join(missing))
    return write_result(res)


# ---------------------------------------------------------------------------------------
# Validation and the entry point
# ---------------------------------------------------------------------------------------

def validate():
    """Config checks that need no GPU: lengths fit, flags exist in vLLM 0.30.0, points sorted."""
    problems = check_lengths()
    for flags in [*RUN_FLAGS.values(), BASE + LEN4K, BASE + LEN32K, ["--max-num-seqs", "1"]]:
        for fl in flags:
            if fl.startswith("--") and fl not in FLAGS_FROM_SOURCE:
                problems.append(f"flag {fl} is not in FLAGS_FROM_SOURCE")
    for k in ("sweep4k_points", "sweep32k_points", "sweepshared_points"):
        pts = val(k)
        if pts != sorted(pts) or not all(isinstance(x, int) and x > 0 for x in pts):
            problems.append(f"{k} must be increasing positive integers")
    if not 0 < float(val("paste_at_s")) < float(val("replay_duration_s")):
        problems.append("paste_at_s must fall inside the replay")
    if int(val("runs")) < 1:
        problems.append("runs must be at least 1")
    return problems


def dispatch(name, dry, runs):
    if name == "launches":
        return cmd_launches(dry)
    if name == "cold32k":
        return cmd_cold32k(dry, runs)
    if name == "static":
        return cmd_static(dry, runs)
    if name in ("runA", "runB", "runC", "runD"):
        return cmd_replay(name[-1], dry, runs)
    if name == "admit128k":
        return cmd_admit128k(dry, runs)
    if name in SWEEPS:
        return cmd_sweep(name, dry, runs)
    if name == "summary":
        return cmd_summary(dry)
    raise SystemExit(f"unknown subcommand {name}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--list" in argv:
        print("\n".join(ORDER))
        return 0
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("subcommand", choices=ORDER + ["all"])
    ap.add_argument("--dry-run", action="store_true", help="print commands and checks; no GPU")
    ap.add_argument("--runs", type=int, default=RUNS_DEFAULT, help="timed runs (default: traffic.json)")
    a = ap.parse_args(argv)
    problems = validate()
    if problems:
        for p in problems:
            print("CONFIG ERROR:", p)
        return 2
    names = ORDER if a.subcommand == "all" else [a.subcommand]
    worst = 0
    for name in names:
        banner(name)
        try:
            rc = dispatch(name, a.dry_run, a.runs)
        except Exception as e:  # noqa: BLE001 - one failed measurement must not stop the rest
            rc = 1
            if not a.dry_run:
                res = new_result(name)
                res["status"] = "failed"
                res["failures"].append({"reason": f"{type(e).__name__}: {e}"})
                write_result(res)
            else:
                print(f"[{name}] dry run failed: {type(e).__name__}: {e}")
        worst = max(worst, rc)
    if a.dry_run:
        print("config checks: OK (lengths fit, every flag is in vLLM 0.30.0's source)")
    return worst


if __name__ == "__main__":
    sys.exit(main())
