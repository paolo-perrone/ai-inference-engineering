# Learning inference engineering

A curated path through the material worth your time, ordered the way the book is ordered.

Most inference reading lists are a pile of links. This one says what each source is good for and where it stops, and it maps to the chapters of *AI Inference Engineering* so you can follow it on its own or alongside the book.

**Start at the top.** Most people jump to the optimization techniques and wonder why nothing clicks. The techniques only make sense once you can locate the bottleneck they exist to fix.

---

## 1. What inference actually is

*Maps to chapters 1 and 3.*

Every model call follows the same path: tokenize, prefill the prompt, then generate one token at a time. Prefill and decode have completely different performance profiles, and almost every confusion downstream comes from not separating them.

Concepts worth owning before anything else: tokenization, the forward pass, autoregressive generation, the prefill and decode split, the KV cache, TTFT and inter-token latency, and why throughput and latency pull against each other.

- **[Why is inference slow and expensive?](https://theaiengineer.substack.com/p/why-is-inference-slow-and-expensive)** The question the whole field answers. Read first.
- **[What does NVIDIA actually do?](https://theaiengineer.substack.com/p/what-does-nvidia-actually-do)** The full-stack view, from silicon to serving, in one piece.

---

## 2. Transformers, only as far as you need them

*Maps to chapter 3.*

You do not need to become a transformer researcher. You do need to understand the computations an inference engine is optimizing, because every optimization in section 4 targets one of them.

- **[LLM Visualization](https://bbycroft.net/llm)** by Brendan Bycroft. A 3D walkthrough of a running GPT. The best way to see the shapes of the tensors moving through a forward pass, and it takes twenty minutes.
- **[Build a Large Language Model From Scratch](https://www.manning.com/books/build-a-large-language-model-from-scratch)** by Sebastian Raschka. The mechanism in code rather than in pictures. Note that it teaches training, not serving, so it stops exactly where inference engineering begins.

---

## 3. The hardware underneath

*Maps to chapters 2 and 4.*

Most inference bottlenecks are hardware bottlenecks, and most engineers reading this have never had a reason to learn the hardware. This section is where people stall, and it is also the section that makes everything after it obvious.

What to come away with: the GPU execution model, the memory hierarchy from registers through SRAM to HBM, why memory bandwidth rather than FLOPs usually sets the ceiling, and how to tell a compute-bound workload from a memory-bound one.

- **[Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)** by Horace He. The canonical piece. Compute-bound, memory-bandwidth-bound, overhead-bound, and how to tell which one you are looking at. If you read one thing in this section, read this.
- **[Why does AI need a GPU?](https://theaiengineer.substack.com/p/why-does-ai-need-a-gpu)** The on-ramp if the above assumes more than you have.
- **[H100 vs H200 vs B200](https://theaiengineer.substack.com/p/h100-vs-h200-vs-b200)** What actually changes between generations, and which spec-sheet numbers matter for serving.
- **[I was ready to return my DGX Spark](https://medium.com/data-science-collective/i-was-ready-to-return-my-dgx-spark-then-nvidias-january-update-changed-everything-e67699155a45)** Local inference hardware, hands on rather than from a datasheet.

---

## 4. Making a single model fast

*Maps to chapters 5, 6 and 7.*

How modern inference gets fast: quantization, the KV cache and its paging, batching, and the tricks that trade cheap work for expensive work.

- **[Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)** The vLLM paper. Read it for the idea rather than the benchmarks: treat KV cache memory the way an operating system treats pages, and fragmentation stops eating your batch size.
- **[FlashAttention](https://arxiv.org/abs/2205.14135)** Tiling and kernel fusion so attention stops making round trips to HBM. The principle outlives the implementation, which is why the paper beats the release notes.
- **[What is quantization in AI?](https://theaiengineer.substack.com/p/what-is-quantization)** and **[GPTQ vs AWQ vs GGUF](https://theaiengineer.substack.com/p/quantization-in-practice-gptq-vs)** The concept, then the practice of choosing between schemes.

---

## 5. Inference engines

*Maps to chapter 8.*

Different engines optimize for different workloads, and the choice matters more than the tuning. vLLM for throughput, SGLang when requests share context, TensorRT-LLM for maximum performance on NVIDIA, llama.cpp for CPU and edge.

- **[Inside vLLM](https://aleksagordic.com/blog/vllm)** by Aleksa Gordić. The best available teardown of an engine's internals, and the closest thing to a systems chapter anyone has published for free.
- **[vLLM](https://github.com/vllm-project/vllm)** · **[SGLang](https://github.com/sgl-project/sglang)** · **[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)** The source is the documentation, particularly the scheduler.
- **[vLLM vs Ollama vs SGLang vs TensorRT-LLM](https://theaiengineer.substack.com/p/vllm-vs-ollama-vs-sglang-vs-tensorrt)** The comparison, with the workload each one is actually built for.

---

## 6. Scale, and operating it

*Maps to chapters 9, 10 and 11.*

Where one GPU stops being enough, and where a working system becomes an operated one.

- **[The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)** by the Hugging Face nanotron team. Built on thousands of scaling experiments across up to 512 GPUs. Training-focused rather than serving-focused, but the parallelism and collective-communication material transfers directly.
- **[Machine Learning Engineering Open Book](https://github.com/stas00/ml-engineering)** by Stas Bekman. First-hand operational guidance on running large GPU clusters: networking, storage, and debugging the failures that only appear at scale. A cookbook rather than a curriculum, and it assumes the fundamentals.
- **[The NVIDIA Nemotron stack for production agents](https://medium.com/data-science-collective/the-nvidia-nemotron-stack-for-production-agents-1483d7fb323d)** A production teardown rather than a tutorial.

---

## Communities and ongoing sources

- **[GPU MODE](https://www.gpumode.com)** 27,000+ members, lectures on YouTube, kernel leaderboards. The centre of gravity for people who do this work.
- **[NVIDIA GTC](https://www.nvidia.com/gtc/) talks** Authoritative and free, though scattered and tied to specific versions.
- **[The AI Engineer](https://theaiengineer.substack.com)** The order to do all of this in, twice a week.

---

## What is missing from this list, honestly

Nothing here teaches inference engineering as one discipline. Every source is excellent at its own layer and stops at the boundary of the next: the papers assume the hardware, the hardware writing assumes the transformer, the engine docs assume both, and none of them assume you are the person who has to hold an SLO at a budget on Monday.

That gap is why the book exists.

---

*Curated for AI Inference Engineering (Manning). Suggestions welcome: open an issue.*
