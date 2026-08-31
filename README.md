# AI Inference Engineering

### Serving foundation models in production

Companion repository for the book, published by [Manning](https://www.manning.com/).

---

## The box nobody opened

In *AI Engineering*, Chip Huyen draws the architecture of a modern AI application, then leaves one component unexplained. Her caption says why: the KV cache and prompt cache "are typically implemented by model API providers, so they aren't shown in this image."

She drew the diagram and left one box closed. **This book is the inside of that box.**

It starts where the framework boundary ends and goes down to the silicon: why inference is slow, what PagedAttention actually does to memory, when it is worth dropping beneath the engine, and how a serving system holds together across thousands of GPUs.

---

## Who it is for

**The engineer who put a model into production and now owns its latency or its bill.** You have deployed something behind an API, you reason in p99 rather than averages, and you have changed a vLLM flag without knowing what it did underneath.

No CUDA experience assumed. Part 1 builds that on-ramp from nothing.

**What you will be able to do afterwards:** diagnose why a workload is slow or expensive and separate memory-bound from compute-bound; choose and configure an engine deliberately; predict what quantization, batching and speculative decoding each cost you; split a deployment across GPUs from the memory-versus-communication trade-off; and design a system that holds a latency SLO at a target cost per token.

**What you will not be able to do:** write a fused attention kernel, or build an engine. That is deliberate. You build here to understand, and you deploy vLLM or SGLang.

---

## What ages, and what does not

About 70% of the book is durable: memory hierarchies and the roofline, attention math, why kernel fusion wins, KV-cache mechanics, batching theory, and the parallelism strategies with their memory-versus-communication trade-offs.

The other 30% is deliberately perishable and quarantined into dated sidebars and [`appendix/`](appendix/) rather than spread through the chapters. Engine APIs, current GPU specs and benchmark numbers live there, so they get corrected here without the rest of the book going stale.

**That is a promise this repository has to keep, not the book.** If a listing stops running because something moved underneath it, that is a bug here.

---

## Start here

**[resources.md](resources.md)** is a curated path through the material worth reading, ordered the way the book is ordered, with what each source is good for and where it stops. Useful on its own, whether or not you read the book.

---

## Table of contents

### Part 1. Foundations: understanding how a model runs

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

## One workload, twelve chapters

Every chapter serves the same system one layer deeper: a coding agent, with whole-repo context in the prompt, multi-turn sessions, and one user action firing many model calls.

Chapter 1 serves it naively and prints the cost per completed task. Each chapter after that fixes one measured failure. Chapter 12 assembles what you have watched improve since page one.

---

## Three ways to read it

**The full path.** Chapters 1 to 12 in order. Part 1 builds the on-ramp, so this works even if you have never written a GPU kernel.

**Serving first.** Willing to treat the forward pass as a black box and care about routing, batching and cost? Read 1, then 6, 7, 8 and 11. Come back to 2 and 3 when you want to know why the numbers are what they are.

**Hardware first.** Want the mechanism before the machinery? Read 1 to 4, then jump to 9 and 10.

---

## Running the code

Python 3.11 or newer. Chapters 1 to 8 run on a single 24GB GPU. Chapters 9 and 10 need two or more.

```bash
git clone https://github.com/paolo-perrone/ai-inference-engineering
cd ai-inference-engineering
uv sync
```

Each chapter pins its own environment where it needs something the base does not provide. Where a listing depends on a specific engine build, the version is recorded next to it and mirrored in the appendix.

---

## Errata

Open an issue. If a listing no longer runs because something moved beneath it, that is a bug here, not in your setup.

## License

Code is MIT. The book text is not included and remains © Manning Publications.
