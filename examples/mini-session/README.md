# The mini session

One coding-agent request, small enough that its whole token sequence prints on a page, the
same shape as the 128,000-token version the book measures.

Every number in the book comes from here. Chapter 1 runs it naively and prints the cost per
completed task; every chapter after that moves one number against the same fixture, so the
improvement is measured rather than described.

## What it is

`repo/` is a 200-line Python project with one real bug: `fetch_user` has no retry, so a
flaky upstream drops requests. The task is the one an engineer would actually type.

```
task.txt      the request, verbatim
repo/         the codebase the agent reads
expected.md   what a correct patch does, so a run can be scored
baseline.json chapter 1's measured numbers, the thing everything is compared against
```

## Why this one

It passes Manning's two-person check. A non-developer understands the goal: make the coding
assistant cheaper without making it worse. An expert could write the patch from `task.txt`
alone.

It is also the smallest thing that still has the properties the book is about: the whole
repo goes in the prompt, so prefix reuse is the hero rather than a footnote; the session is
multi-turn, so the KV cache lives across requests; and one user action fires several model
calls, so cost per completed task is the unit that matters rather than cost per token.

## Running it

```bash
python3 measure.py --engine naive --table
```

The naive runner is deliberately bad. That is the point of chapter 1.

## Timing it on your card (chapter 2)

```bash
pip install vllm==0.30.0
python3 measure.py --serve Qwen/Qwen2.5-7B-Instruct --write-baseline
```

This starts the engine twice, with CUDA graphs off and then on, and prefix caching off both
times, so every run pays a full prefill. Each start gets one warm-up request and five timed
ones, sent one at a time. It prints every run's time to first token and decode step, each
setting's median and spread, and the ratio between the two settings. The run log in `runs/`
names the card, the driver, and the CUDA and vLLM versions. `--write-baseline` adds the
graphs-on `decode_step_ms` and `ttft_s` to `baseline.json` and keeps every other field.

Against an engine you started yourself, the same timing runs with whatever flags it has:

```bash
python3 measure.py --url http://localhost:8000 --model <served model name> --steps
```

`python3 test_measure.py` checks both modes against a fake engine, with no GPU.
