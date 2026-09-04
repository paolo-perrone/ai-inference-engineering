# The mini session

One coding-agent request, small enough that its whole token sequence prints on a page, the
same shape as the 200,000-token version the book measures.

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
uv sync
python -m examples.mini_session.run --engine naive
```

The naive runner is deliberately bad. That is the point of chapter 1.
