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
  python3 measure.py --engine vllm --url ...
  python3 measure.py --self-test
"""
import argparse
import json
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE / "repo"
BASELINE = HERE / "baseline.json"

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

    A naive loop re-sends the whole prompt every turn. That is the number chapter 6 kills.
    """
    prompt = build_prompt()
    pt = count_tokens(prompt)
    per_turn = []
    for i in range(turns):
        if engine == "naive":
            sent = pt                      # the whole context, again
        else:
            sent = pt if i == 0 else count_tokens("Task: " + str(i))
        out = 180                          # a patch of this size, measured once
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


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--engine", default="naive")
    ap.add_argument("--url", help="OpenAI-compatible endpoint; switches to live measurement")
    ap.add_argument("--model", default="qwen2.5-7b")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--write-baseline", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.url:
        print(json.dumps(measure_live(a.url, a.model, a.turns, a.concurrency), indent=2))
        return 0
    r = run(a.engine, a.turns)
    print(json.dumps(r, indent=2))
    if a.write_baseline:
        tmp = BASELINE.with_suffix(".tmp")
        tmp.write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, BASELINE)
        print(f"\nbaseline written to {BASELINE.name}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
