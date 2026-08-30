# AI Inference Engineering

Companion code for the book, published by [Manning](https://www.manning.com/).

Every listing in the book runs from this repository. The book teaches the durable
principles; this is where the version-pinned specifics live, so the printed pages
stay correct while the engines underneath keep moving.

Written by [Paolo Perrone](https://www.linkedin.com/in/paoloperrone/), who writes
[The AI Engineer](https://theaiengineer.substack.com) and edits
[Data Science Collective](https://medium.com/data-science-collective).

---

## What this book is

Chip Huyen's *AI Engineering* defined building applications on top of foundation
models, and stops at the framework boundary. This book starts at that boundary and
goes down to the silicon: why inference is slow, what PagedAttention does to memory,
when to drop beneath the framework, and how a serving system holds together across
thousands of GPUs.

It is written for the engineer who has put a model into production and now owns its
latency or its bill. No CUDA experience assumed.

---

## Table of contents

### Part 1. Foundations: understanding how inference works

| | Chapter | Code |
|---|---|---|
| 1 | The Inference Engineer | [`ch01`](ch01-inference-engineer/) |
| 2 | How GPUs run models | [`ch02`](ch02-how-gpus-run-models/) |
| 3 | The forward pass and the kernels that run it | [`ch03`](ch03-forward-pass-and-kernels/) |
| 4 | The accelerator landscape | [`ch04`](ch04-accelerator-landscape/) |

### Part 2. Serving a single model: making inference fast and cheap

| | Chapter | Code |
|---|---|---|
| 5 | Model-level optimization | [`ch05`](ch05-model-level-optimization/) |
| 6 | The KV cache | [`ch06`](ch06-kv-cache/) |
| 7 | Batching and scheduling | [`ch07`](ch07-batching-and-scheduling/) |
| 8 | Inference engines as systems | [`ch08`](ch08-inference-engines/) |

### Part 3. Scale: distributing inference across many GPUs

| | Chapter | Code |
|---|---|---|
| 9 | Multi-GPU parallelism | [`ch09`](ch09-multi-gpu-parallelism/) |
| 10 | Coordination and cluster scale | [`ch10`](ch10-coordination-and-cluster-scale/) |

### Part 4. Production: running inference reliably and economically

| | Chapter | Code |
|---|---|---|
| 11 | Operating inference in production | [`ch11`](ch11-operating-in-production/) |
| 12 | Designing an inference system end to end | [`ch12`](ch12-end-to-end-design/) |

---

## Three ways to read it

**The full path.** Chapters 1 to 12 in order. Part 1 builds the on-ramp from nothing,
so this works even if you have never written a GPU kernel.

**Serving first.** If you are willing to treat the forward pass as a black box and
care about routing, batching and cost, read 1, then 6, 7, 8, and 11. Come back to
2 and 3 when you want to know why the numbers are what they are.

**Hardware first.** If you want the mechanism before the machinery, read 1 to 4,
then jump to 9 and 10 for how it all coordinates at scale.

---

## What ages, and what does not

About 70% of the book is durable: memory hierarchies and the roofline, attention
math, why kernel fusion wins, KV-cache mechanics, batching theory, and the
parallelism strategies and their memory-versus-communication trade-offs.

The remaining 30% is deliberately perishable, quarantined into dated sidebars and
[`appendix/`](appendix/) rather than spread through the chapters. Engine APIs, current
GPU specs, and benchmark numbers live there, so they can be corrected here without
the rest of the book going stale.

---

## Running the code

Requires Python 3.11 or newer and, for most chapters, an NVIDIA GPU.
Chapters that run on CPU or on other accelerators say so in their own README.

```bash
git clone https://github.com/paoloperrone/ai-inference-engineering
cd ai-inference-engineering
uv sync
```

Each chapter pins its own environment where it needs something the base does not
provide. Where a listing depends on a specific engine build, the exact version is
recorded next to it and mirrored in the appendix.

---

## Errata and corrections

Open an issue. If a listing no longer runs because something moved beneath it, that
is a bug in this repository, not in your setup, and it gets fixed here.

## License

Code is MIT. The book text is not included here and remains © Manning Publications.
