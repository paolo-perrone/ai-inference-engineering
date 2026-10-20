#!/usr/bin/env python3
"""measure.py, the one number every chapter moves.

Prints cost per completed task for the mini session. Chapter 1 runs it against a naive
serving loop and writes baseline.json; every chapter after that runs it again and compares.

The prompt is assembled here, not in the engine, so the token accounting is the same
whatever is serving. That is the whole point: the numbers in the book have to be
attributable to the layer the chapter is about.

Usage:
  python3 measure.py --dry-run              token accounting only, no model call
  python3 measure.py --engine naive         chapter 1's deliberately bad loop
  python3 measure.py --engine naive --table the same run, as the book prints it
  python3 measure.py --engine vllm --url ...
  python3 measure.py --url ... --steps      time to first token and decode step, 5 runs
  python3 measure.py --serve <model>        chapter 2's experiment: CUDA graphs off, then on
  python3 measure.py --self-test
"""
import argparse
import datetime
import json
import os
import pathlib
import platform
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
import uuid

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE / "repo"
BASELINE = HERE / "baseline.json"
RUNS = HERE / "runs"
COMPLETION_TOKENS = 180                    # the answer length every turn generates

# vLLM 0.30.0's switch for chapter 2's experiment. Every other engine setting stays put.
GRAPHS_OFF = ["--compilation-config", json.dumps({"cudagraph_mode": "NONE"})]

# Priced per million tokens. Recorded here so a number in the book is traceable to the
# rate that produced it; change these and every derived figure changes with them.
RATES = {"prompt": 0.20, "completion": 0.80}
RATES_DATED = "2026-09-03, self-hosted 7B on a 24GB card, amortised"


def repo_files():
    """Every source file the agent sees, in a stable order, so runs are comparable."""
    return sorted(p for p in REPO.rglob("*.py") if p.stat().st_size)


def build_prompt():
    """The whole repo plus the task. Deliberately naive: this is what chapter 1 fixes."""
    parts = [f"You are a coding agent working in this repository.\n"]
    for p in repo_files():
        parts.append(f"\n--- {p.relative_to(REPO)} ---\n{p.read_text(encoding='utf-8')}")
    parts.append("\n\nTask: " + (HERE / "task.txt").read_text(encoding="utf-8").strip())
    return "".join(parts)


def count_tokens(text):
    """Characters over four. Crude, stated, and identical across every run.

    A real tokenizer changes the absolute numbers and none of the comparisons, and pinning
    one here would add a dependency to a file whose only job is to be reproducible.
    """
    return len(text) // 4


def cost(prompt_tokens, completion_tokens):
    return (prompt_tokens * RATES["prompt"] + completion_tokens * RATES["completion"]) / 1e6


def run(engine, turns):
    """One session: `turns` requests over the same repo context.

    A naive loop re-sends the whole context every turn, history included, so each turn
    costs more than the last. That is the number chapter 6 kills. A cached run sends only
    each turn's new text, because everything before it already sits in the KV cache.
    """
    prompt = build_prompt()
    pt = count_tokens(prompt)
    per_turn = []
    history = 0                            # earlier turns' text and answers, in tokens
    for i in range(turns):
        new = 0 if i == 0 else count_tokens("Task: " + str(i))
        if engine == "naive":
            sent = pt + history + new      # the whole context, again, and it grows
        else:
            sent = pt if i == 0 else new   # only the new text; the rest is cached
        out = COMPLETION_TOKENS            # a patch of this size, measured once
        history += new + out
        per_turn.append({"turn": i + 1, "prompt_tokens": sent, "completion_tokens": out,
                         "cost_usd": round(cost(sent, out), 6)})
    total = round(sum(t["cost_usd"] for t in per_turn), 6)
    return {"engine": engine, "turns": turns, "repo_files": len(repo_files()),
            "context_tokens": pt, "per_turn": per_turn,
            "cost_per_completed_task_usd": total,
            "cost_by_stage": stage_split(per_turn),
            "rates_usd_per_million": RATES, "rates_dated": RATES_DATED}


def stage_split(per_turn):
    """Which serving stage owns each dollar of the task.

    In this pricing model only two stages carry token dollars: prefill is billed on prompt
    tokens and decode on completion tokens. Arrival, tokenization, queueing, scheduling and
    streaming cost latency rather than tokens, and the split says so explicitly instead of
    inventing a number for them. Figure 1.8 in the book is drawn from this block.
    """
    p = sum(t["prompt_tokens"] for t in per_turn) * RATES["prompt"] / 1e6
    c = sum(t["completion_tokens"] for t in per_turn) * RATES["completion"] / 1e6
    total = p + c
    return {
        "prefill_usd": round(p, 6), "decode_usd": round(c, 6),
        "prefill_share": round(p / total, 3), "decode_share": round(c / total, 3),
        "other_stages": "arrive, tokenize, queue, schedule, stream: latency, not tokens",
    }


def measure_live(url, model, turns, concurrency):
    """The same session, actually served, with wall-clock numbers.

    Sends the fixture's turns to an OpenAI-compatible /v1/completions endpoint and records
    time to first token, total latency and tokens per second, at each requested concurrency.
    This is what turns figure 1.4's frontier from a shape into data: run it at
    concurrency 1, then 4, then 8, against the engine on your own card. Standard library
    only, so the fixture stays dependency-free.
    """
    import concurrent.futures
    import urllib.request

    prompt = build_prompt()

    def one_request():
        body = json.dumps({"model": model, "prompt": prompt, "max_tokens": 180,
                           "temperature": 0, "stream": True}).encode()
        req = urllib.request.Request(url.rstrip("/") + "/v1/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = time.monotonic()
        first, tokens = None, 0
        with urllib.request.urlopen(req, timeout=300) as r:
            for line in r:
                if not line.startswith(b"data:"):
                    continue
                if line.strip() == b"data: [DONE]":
                    break
                if first is None:
                    first = time.monotonic() - t0
                tokens += 1
        return {"ttft_s": round(first or 0.0, 3),
                "latency_s": round(time.monotonic() - t0, 3), "completion_tokens": tokens}

    t0 = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        results = [f.result() for f in [ex.submit(one_request) for _ in range(concurrency * turns)]]
    wall = time.monotonic() - t0
    lat = sorted(r["latency_s"] for r in results)
    out_tokens = sum(r["completion_tokens"] for r in results)
    return {"url": url, "model": model, "concurrency": concurrency,
            "requests": len(results), "wall_s": round(wall, 3),
            "p50_latency_s": lat[len(lat) // 2], "p99_latency_s": lat[-1],
            "throughput_tokens_per_s": round(out_tokens / wall, 1),
            "per_request": results}


def one_timed_request(url, model, prompt):
    """One request at a batch of one, timed token by token.

    Returns the time to first token and the decode step: the time from the first token to
    the last, divided by the steps between them. The request carries its own cache salt, so
    vLLM's prefix cache cannot hand it an earlier run's prefill and time to first token
    measures a full prefill. It also ignores the end-of-sequence token, so every run decodes
    the same 180 tokens.
    """
    import urllib.request

    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": COMPLETION_TOKENS,
                       "temperature": 0, "ignore_eos": True, "stream": True,
                       "stream_options": {"include_usage": True},
                       "cache_salt": uuid.uuid4().hex}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    first = last = None
    chunks, usage = 0, {}
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[len(b"data:"):].strip()
            if data == b"[DONE]":
                break
            msg = json.loads(data)
            usage = msg.get("usage") or usage
            if msg.get("choices") and msg["choices"][0].get("text"):
                last = time.monotonic()
                first = last if first is None else first
                chunks += 1
    tokens = usage.get("completion_tokens") or chunks
    if first is None or tokens < 2:
        raise RuntimeError("the endpoint streamed fewer than 2 tokens, so there is no step to time")
    return {"ttft_s": round(first - t0, 4),
            "decode_step_ms": round((last - first) * 1000 / (tokens - 1), 3),
            "completion_tokens": tokens, "engine_prompt_tokens": usage.get("prompt_tokens")}


def summarize(timed):
    """Medians, because one slow run should not move the number, and the spread once."""
    steps = [r["decode_step_ms"] for r in timed]
    ttfts = [r["ttft_s"] for r in timed]
    return {"runs": len(timed),
            "decode_step_ms": round(statistics.median(steps), 2),
            "decode_step_spread_ms": round(max(steps) - min(steps), 2),
            "ttft_s": round(statistics.median(ttfts), 3),
            "ttft_spread_s": round(max(ttfts) - min(ttfts), 3)}


def measure_steps(url, model, runs, warmup=1):
    """`runs` timed requests after `warmup` discarded ones, sent one at a time.

    One at a time keeps the batch at one, the case chapter 2 predicts. The first request
    after an engine starts pays one-off costs that belong to no step, so it is thrown away.
    """
    prompt = build_prompt()
    for _ in range(warmup):
        one_timed_request(url, model, prompt)
    timed = [one_timed_request(url, model, prompt) for _ in range(runs)]
    return {"per_run": timed, **summarize(timed)}


def _output(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _interpreter(exe):
    """The Python a console script runs under, read from its first line."""
    try:
        line = pathlib.Path(exe).read_text(encoding="utf-8", errors="replace").splitlines()[0]
    except (OSError, IndexError):
        return None
    if not line.startswith("#!"):
        return None
    parts = line[2:].split()
    if parts and parts[0].endswith("/env") and len(parts) > 1:
        return shutil.which(parts[1])
    return parts[0] if parts else None


def environment(url, vllm_exe=None):
    """What the run log names: the card, its driver, and the CUDA and vLLM versions.

    Each field says where it was read, because a version read from the wrong place is how a
    number gets credited to software it never ran on.
    """
    import urllib.request

    rows = _output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    gpus = [dict(zip(("name", "driver"), (s.strip() for s in r.split(","))))
            for r in rows.splitlines() if r.strip()]
    header = _output(["nvidia-smi"])
    env = {"gpus": gpus,
           "cuda_supported_by_driver": (header.split("CUDA Version:")[1].split()[0]
                                        if "CUDA Version:" in header else None),
           "cuda_runtime": None, "vllm": None, "python": platform.python_version(),
           "sources": {"gpus": "nvidia-smi", "cuda_supported_by_driver": "nvidia-smi header",
                       "cuda_runtime": "torch.version.cuda in vLLM's interpreter",
                       "vllm": "the server's /version"}}
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/version", timeout=10) as r:
            env["vllm"] = json.loads(r.read()).get("version")
    except (OSError, ValueError):
        pass
    py = _interpreter(vllm_exe) if vllm_exe else None
    if py:
        out = _output([py, "-c", "import torch; print(torch.version.cuda)"]).splitlines()
        env["cuda_runtime"] = out[-1] if out else None
    return env


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_engine(model, flags, log_path, timeout=1800):
    """Start `vllm serve` with prefix caching off and wait until /health answers."""
    import urllib.request

    exe = shutil.which("vllm")
    if not exe:
        raise SystemExit("vllm is not on PATH; install the version chapter 2 pins, vllm==0.30.0")
    port = _free_port()
    cmd = [exe, "serve", model, "--host", "127.0.0.1", "--port", str(port),
           "--no-enable-prefix-caching"] + flags
    log = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            log.close()
            raise SystemExit(f"vllm serve exited with code {proc.returncode}; see {log_path}")
        try:
            with urllib.request.urlopen(url + "/health", timeout=5) as r:
                if r.status == 200:
                    return proc, url, cmd, log
        except OSError:
            pass
        time.sleep(2)
    stop_engine(proc, log)
    raise SystemExit(f"vllm serve did not answer /health within {timeout} s; see {log_path}")


def stop_engine(proc, log):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    log.close()


def _stamp():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_log(result, name):
    RUNS.mkdir(exist_ok=True)
    path = RUNS / name
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return path.relative_to(HERE).as_posix() if path.is_relative_to(HERE) else str(path)


def graphs_pair(model, runs, extra=()):
    """Chapter 2's one experiment: the same decode step with CUDA graphs off, then on.

    Each setting gets its own engine start, because vLLM captures its graphs when it starts.
    Prefix caching is off for both, so every run pays a full prefill.
    """
    stamp = _stamp()
    RUNS.mkdir(exist_ok=True)
    result = {"started_utc": stamp, "model": model, "settings": {}}
    for name, flags in (("graphs_off", GRAPHS_OFF), ("graphs_on", [])):
        proc, url, cmd, log = start_engine(model, flags + list(extra),
                                           RUNS / f"{stamp}-{name}-server.log")
        try:
            if "environment" not in result:
                result["environment"] = environment(url, cmd[0])
            result["settings"][name] = {"server_command": cmd, **measure_steps(url, model, runs)}
        finally:
            stop_engine(proc, log)
    off, on = result["settings"]["graphs_off"], result["settings"]["graphs_on"]
    result["graphs_off_over_on"] = round(off["decode_step_ms"] / on["decode_step_ms"], 3)
    result["run_log"] = _write_log(result, f"{stamp}-graphs-pair.json")
    return result


def steps_at(url, model, runs):
    """The same timing against an engine you started yourself, with whatever flags it has."""
    result = {"started_utc": _stamp(), "model": model, "url": url,
              "environment": environment(url),
              "settings": {"as_served": measure_steps(url, model, runs)}}
    result["run_log"] = _write_log(result, f"{result['started_utc']}-steps.json")
    return result


def print_steps(result):
    """The pair as the book prints it: every run, then each setting's median and spread."""
    print("setting      run  ttft_s  decode_step_ms")
    for name, m in result["settings"].items():
        for i, r in enumerate(m["per_run"], 1):
            print(f"{name:<11}{i:>5}  {r['ttft_s']:>6.3f}  {r['decode_step_ms']:>14.2f}")
    for name, m in result["settings"].items():
        print(f"{name}: decode step {m['decode_step_ms']:.2f} ms, median of {m['runs']} runs, "
              f"spread {m['decode_step_spread_ms']:.2f} ms; time to first token "
              f"{m['ttft_s']:.3f} s, spread {m['ttft_spread_s']:.3f} s")
    if "graphs_off_over_on" in result:
        print(f"graphs off over graphs on: {result['graphs_off_over_on']:.3f}x")
    env = result["environment"]
    cards = ", ".join(f"{g['name']} (driver {g['driver']})" for g in env["gpus"]) or "no GPU found"
    first = next(iter(result["settings"].values()))["per_run"][0]
    print(f"card: {cards}; CUDA runtime {env['cuda_runtime'] or 'unknown'}, "
          f"driver supports CUDA {env['cuda_supported_by_driver'] or 'unknown'}; "
          f"vLLM {env['vllm'] or 'unknown'}")
    print(f"the engine counted {first['engine_prompt_tokens']} prompt tokens; "
          f"the fixture estimates {count_tokens(build_prompt())}")
    print(f"run log: {result['run_log']}")


def write_baseline(updates, path=None):
    """Merge `updates` into baseline.json, keeping every key this run did not produce.

    Each chapter adds its own fields, chapter 2 the decode step and time to first token, so
    rewriting the file from one run would erase what the others recorded.
    """
    path = path or BASELINE
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data.update(updates)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    print(f"\nbaseline written to {path.name}", file=sys.stderr)


def self_test():
    fails = []
    if not repo_files():
        fails.append("the fixture repo has no python files")
    p = build_prompt()
    if "fetch_user" not in p or "Task:" not in p:
        fails.append("the prompt does not carry the repo and the task")
    a, b = run("naive", 4), run("naive", 4)
    if a != b:
        fails.append("two runs of the same fixture differ, so no number here is comparable")
    if run("naive", 4)["cost_per_completed_task_usd"] <= run("cached", 4)["cost_per_completed_task_usd"]:
        fails.append("the naive loop should cost more than a cached one; the fixture proves nothing")
    sn, sc = run("naive", 4)["cost_by_stage"], run("cached", 4)["cost_by_stage"]
    if abs(sn["prefill_usd"] + sn["decode_usd"] - run("naive", 4)["cost_per_completed_task_usd"]) > 1e-5:
        fails.append("the stage split does not sum to the task cost")
    if not (sn["prefill_share"] > 0.5 > sc["prefill_share"]):
        fails.append("naive should be prefill-heavy and cached should flip it; the split shows neither")
    if fails:
        for f in fails:
            print("  FAIL:", f)
        return 1
    print("  mini-session self-test PASS: fixture reads, prompt assembles, runs are "
          "deterministic, and the naive loop costs more than the cached one")
    return 0


def print_table(r, model):
    """The run as the book prints it: one line per turn, then the task's cost."""
    print(f"{r['engine']} run, prices for {model}: "
          f"${RATES['prompt']:.2f}/M prompt, ${RATES['completion']:.2f}/M completion")
    print("turn  prompt  completion  cost_usd")
    for t in r["per_turn"]:
        print(f"{t['turn']:>4}  {t['prompt_tokens']:>6}  {t['completion_tokens']:>10}  "
              f"{t['cost_usd']:.6f}")
    s = r["cost_by_stage"]
    print(f"cost per completed task: ${r['cost_per_completed_task_usd']:.6f} "
          f"(prefill {s['prefill_share']:.1%}, decode {s['decode_share']:.1%})")


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--engine", default="naive")
    ap.add_argument("--url", help="OpenAI-compatible endpoint; switches to live measurement")
    ap.add_argument("--model", default="qwen2.5-7b")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--steps", action="store_true",
                    help="with --url: time to first token and decode step at a batch of one")
    ap.add_argument("--serve", metavar="MODEL",
                    help="start vllm serve MODEL twice, CUDA graphs off then on, and time both")
    ap.add_argument("--serve-args", default="", help="extra vllm serve flags, as one string")
    ap.add_argument("--runs", type=int, default=5, help="timed runs per setting")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--write-baseline", action="store_true")
    ap.add_argument("--table", action="store_true", help="print a short table, not JSON")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.serve or (a.url and a.steps):
        result = (graphs_pair(a.serve, a.runs, shlex.split(a.serve_args)) if a.serve
                  else steps_at(a.url, a.model, a.runs))
        print_steps(result)
        if a.write_baseline:
            m = result["settings"].get("graphs_on") or result["settings"]["as_served"]
            write_baseline({"decode_step_ms": m["decode_step_ms"], "ttft_s": m["ttft_s"],
                            "steps_run_log": result["run_log"]})
        return 0
    if a.url:
        print(json.dumps(measure_live(a.url, a.model, a.turns, a.concurrency), indent=2))
        return 0
    r = run(a.engine, a.turns)
    if a.table:
        print_table(r, a.model)
    else:
        print(json.dumps(r, indent=2))
    if a.write_baseline:
        write_baseline(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
