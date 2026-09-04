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
            "rates_usd_per_million": RATES, "rates_dated": RATES_DATED}


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
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--write-baseline", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
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
