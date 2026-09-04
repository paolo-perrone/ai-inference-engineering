# The inference engineering glossary

Every term this book uses, defined once, with the arithmetic where arithmetic is the point.

**Why this exists in a repo rather than a book appendix.** A definition you can link to is worth more than a definition you can cite. This is the same material as the book's Appendix A, kept public so an engineer who never buys the book can still get the answer, and so a reader can send a colleague a URL rather than a page number.

**The rule for every entry: if a term has a formula, the formula is here and it is worked on a real model.** A glossary that says "the KV cache stores keys and values for previous tokens" has told you nothing you could act on. One that shows 84 GB for Llama 70B at 128k context, against 140 GB of weights, has told you why your batch size is what it is.

---

## Status

**This file is the term list, not yet the glossary.** Terms marked ⬜ have no entry written. The three columns after each term record what the field currently does with it, which is what decides the order they get written in.

**Gap column key.** `absent` means the term does not appear in the closest competing text, Philip Kiely's *Inference Engineering* (Baseten Books, 2026) or its 98-page companion at inferenceengineering.tech, harvested 2026-07-30 at 40,840 words. `thin` means named without a mechanism. `wrong` means stated incorrectly, with the correction in the entry. `covered` means he handles it well and the entry has to earn its place another way.

**Priority key.** 🔴 writes first: the term is absent from the competing text and load-bearing in this book. 🟡 second: thin or wrong elsewhere, so the entry wins by being correct. ⚪ third: well covered, include for completeness.

---

## Part 1. Foundations

### Chapter 1. Why inference is its own discipline

| | Term | Elsewhere | Why it matters here |
|---|---|---|---|
| 🔴 ⬜ | **Goodput** | absent from his entire site | The only metric that makes latency and throughput commensurable. Without it the central tradeoff stays a tension you feel rather than a quantity you optimise. |
| 🔴 ⬜ | **Cost per completed task** | absent | The unit for an agent workload, where one user action fires many model calls. Cost per token flatters a system that retries. |
| 🔴 ⬜ | **Little's Law** | stated as a learning objective, never delivered | Concurrency equals throughput times latency. It is how you size replicas, and without it the percentile discussion has no mechanism. |
| 🟡 ⬜ | **Cost per token** | covered as a formula, never worked; his chapter carries no dollar figure at all | GPU-hour price divided by achieved tokens per second times 3600. The entry shows the division on a named GPU. |
| 🟡 ⬜ | **Time to first token (TTFT)** | covered | Prefill-bound. Pairs with TPOT and the entry has to say which one a given product actually feels. |
| 🟡 ⬜ | **Time per output token (TPOT)** | covered as ITL | Decode-bound. The reciprocal relationship to perceived tokens per second is where readers get confused. |
| 🟡 ⬜ | **ISL and OSL** | covered well, and it is one of his best artifacts | Input and output sequence length. The ratio decides prefill-heavy against decode-heavy, which decides the silicon. |
| ⚪ ⬜ | **Inference engineering** | covered, as an outcome rather than a boundary | His: making a model faster, cheaper and more reliable. That says nothing about what the role owns or where it hands off. |

### Chapter 2. How GPUs run models

| | Term | Elsewhere | Why it matters here |
|---|---|---|---|
| 🔴 ⬜ | **Overhead-bound** | **absent as a word**, across the book and all 98 companion pages | The third regime. A 3B model at batch 4 sits below both roofs and the reader concludes the measurement is broken. |
| 🔴 ⬜ | **Kernel launch overhead** | absent | The mechanism behind overhead-bound, and the reason CUDA graphs exist. |
| 🔴 ⬜ | **CUDA graph** | absent | The fix for launch overhead. Named nowhere in the competing text. |
| 🔴 ⬜ | **Occupancy** | absent | How much of an SM's capacity a kernel actually uses. You cannot reason about a slow kernel without it. |
| 🔴 ⬜ | **Warp** | appears once, inside "Warp Matrix Multiply" | The scheduling unit. His hardware chapter teaches the GPU as a spec sheet to compare; this book teaches it as an execution model. |
| 🔴 ⬜ | **Register file** | absent | The fastest and smallest level. Its absence is why his roofline has one memory number. |
| 🟡 ⬜ | **SRAM** | defined in a DRAM-versus-SRAM box, never connected to attention | The tier FlashAttention tiles into. Defining it without that connection wastes it. |
| 🟡 ⬜ | **Arithmetic intensity** | formula given, balance point given, never computed on a workload | FLOPs over bytes accessed. The entry computes it for one decode step and one prefill, and plots both. |
| 🟡 ⬜ | **Roofline model** | described, never drawn with a workload on it | The entry carries a roofline with two dots and a machine-balance line. |
| 🟡 ⬜ | **Ops:byte ratio** | derived well, 989 TFLOPS over 3.35 TB/s equals about 295 on an H100 | He teaches the derivation, which is the right move. The entry extends it to the batch size where decode crosses it. |
| 🟡 ⬜ | **Memory-bound** | covered | The entry has to add the crossing, not restate the definition. |
| 🟡 ⬜ | **Compute-bound** | covered | Same. |
| ⚪ ⬜ | **HBM** | covered | Capacity and bandwidth per generation. |
| ⚪ ⬜ | **Streaming multiprocessor (SM)** | covered as a definition box | Contains CUDA cores, tensor cores and special function units. |
| ⚪ ⬜ | **Tensor core** | covered | |

### Chapter 3. The forward pass and the kernels that run it

| | Term | Elsewhere | Why it matters here |
|---|---|---|---|
| 🔴 ⬜ | **FlashAttention** | named three times, mechanism given nowhere | Tiling attention into SRAM so the N by N matrix is never materialised in HBM. The reader is told the most important kernel exists and never what it does. |
| 🔴 ⬜ | **Kernel fusion** | absent as a mechanism | Doing more work per trip to HBM. The general form of what FlashAttention does specifically. |
| 🟡 ⬜ | **Prefill** | covered, with the correct root cause: matrix-matrix | One of his best single sentences. The entry has to add the number. |
| 🟡 ⬜ | **Decode** | covered, vector-matrix | Same. |
| ⚪ ⬜ | **Logits** | covered | |
| ⚪ ⬜ | **Sampling, temperature, top-k, top-p** | covered | |

### Chapter 4. The accelerator landscape

| | Term | Elsewhere | Why it matters here |
|---|---|---|---|
| 🔴 ⬜ | **Thermal and power budget** | absent | A chip that throttles has a different effective roofline than its spec sheet. |
| 🟡 ⬜ | **Dense versus sparse FLOPS** | covered, and he flags the trap correctly | Read the dense number. Credit him and keep the warning. |
| 🟡 ⬜ | **Unified memory** | covered via Apple M4 Max | The DGX Spark case is the more interesting one: capacity without bandwidth. |
| 🟡 ⬜ | **MIG** | covered, with a worked 3/7 slice | The entry has to beat a worked example, so it needs the latency consequence of slicing. |
| ⚪ ⬜ | **NVLink, NVSwitch, InfiniBand** | covered as a spec table | Chapter 10 makes these mechanisms rather than numbers. |

---

## Part 2. Serving a single model

### Chapter 5. Model-level optimization

| | Term | Elsewhere | Why it matters here |
|---|---|---|---|
| 🔴 ⬜ | **Quantization-aware training versus post-training quantization** | compressed into his single chapter 5 | Different accuracy costs, different engineering effort. |
| 🟡 ⬜ | **Speculative decoding** | covered | Draft and verify. The entry adds the acceptance rate that decides whether it pays. |
| 🟡 ⬜ | **INT4 and INT8** | covered, and his own table contradicts his conclusion: he reads a 4x memory-access cut as a speed win while the table shows decode throughput plateauing after 8 bits | The correction is the entry. |
| ⚪ ⬜ | **Distillation** | covered well, defined against synthetic-data fine-tuning by the teacher's distributions | |

### Chapter 6. The KV cache

| | Term | Elsewhere | Why it matters here |
|---|---|---|---|
| 🔴 ⬜ | **KV cache size** | formula and its worked 84 GB result live in a marketing guide, not in the book | The calculation that decides how many concurrent users fit on a GPU. |
| 🔴 ⬜ | **Grouped-query attention (GQA)** | absent from his glossary | The single biggest structural lever on cache size, and the reason 56 KB per token rather than 224. |
| 🔴 ⬜ | **Multi-query attention (MQA)** | absent | |
| 🔴 ⬜ | **Sliding-window attention** | absent | Named by Cody Yu as one of two things making serving engines harder over the next two years. |
| 🟡 ⬜ | **Attention complexity** | **stated wrongly**: "the KV cache makes attention linear rather than quadratic" | The cache makes the per-token cost linear in context. Generating N tokens is still quadratic in total, and prefill of N tokens is quadratic. This is the foundation of every long-context cost model. |
| 🟡 ⬜ | **PagedAttention** | covered | Blocks rather than contiguous allocation. The entry adds the fragmentation number it removes. |
| 🟡 ⬜ | **Prefix caching** | covered | And measured elsewhere: prefix cache is 58.6% of cost across 665,453 agent steps in the TraceLab corpus. |
| 🔴 ⬜ | **KV cache eviction** | absent | What happens when the cache does not fit, which is the normal case at long context. |

### Chapter 7. Batching and scheduling

| | Term | Elsewhere | Why it matters here |
|---|---|---|---|
| 🔴 ⬜ | **Continuous batching** | absent from the companion | The mechanism behind every modern engine's throughput claim. |
| 🔴 ⬜ | **Chunked prefill** | absent | On by default in vLLM V1 and the fix for head-of-line blocking. |
| 🔴 ⬜ | **Head-of-line blocking** | absent | Why one 200k-token request stalls fifty short ones. |
| 🔴 ⬜ | **Admission control** | absent | A stage in every production scheduler and in no public teaching diagram. |
| 🔴 ⬜ | **Preemption** | absent | Same. |
| 🔴 ⬜ | **Queueing** | the word appears nowhere on his site | Every latency complaint the reader has is a queueing phenomenon. |
| 🔴 ⬜ | **Generation stall** | absent | Documented in Sarathi-Serve, OSDI'24: prefills scheduled ahead of resumed decodes, with 2.6x to 5.6x capacity gains from removing them. |

### Chapter 8. Inference engines as systems

| | Term | Elsewhere | Why it matters here |
|---|---|---|---|
| 🟡 ⬜ | **RadixAttention** | covered | SGLang's prefix-sharing structure. |
| 🔴 ⬜ | **Multi-LoRA serving** | absent | Many fine-tunes on one base, which is how most production fleets actually run. |
| 🔴 ⬜ | **The build-buy line** | structurally absent: he sells the layer above CUDA | Where a library stops paying and a custom kernel starts. He cannot write this honestly. |

---

## Part 3. Scale

### Chapter 9. Multi-GPU parallelism

| | Term | Elsewhere | Why it matters here |
|---|---|---|---|
| 🟡 ⬜ | **Tensor parallelism** | covered, and Baseten's own blog contradicts the book: the blog calls it all-to-all communication where the book counts an all-reduce per block | The correction is the entry. |
| 🔴 ⬜ | **Expert parallelism** | appears once, attached to a 72-GPU rack | How MoE actually serves, and the counterpart to his correct point that batching activates nearly every expert. |
| 🔴 ⬜ | **Context and sequence parallelism** | absent | |
| ⚪ ⬜ | **Pipeline parallelism** | covered | |

### Chapter 10. Coordination and cluster scale

| | Term | Elsewhere | Why it matters here |
|---|---|---|---|
| 🔴 ⬜ | **All-reduce, all-gather, all-to-all** | absent as mechanisms | The three collectives every parallelism strategy is priced in. |
| 🔴 ⬜ | **Ring and hierarchical collectives** | absent | Why interconnect topology decides whether scaling is linear. |
| 🔴 ⬜ | **Disaggregated prefill and decode** | compressed into his single chapter 5 | The textbook case for an agent workload, where prefill is enormous and decode is tiny. |
| 🔴 ⬜ | **KV cache transfer** | absent | The stage that exists in disaggregated setups and in no teaching diagram. |

---

## Part 4. Production

### Chapter 11. Operating inference in production

| | Term | Elsewhere | Why it matters here |
|---|---|---|---|
| 🔴 ⬜ | **Model FLOPs utilisation (MFU)** | absent | And the correction the reader needs: utilisation as nvidia-smi reports it means a kernel was resident, not that work was useful. |
| 🔴 ⬜ | **Determinism under batching** | absent | The same prompt returns different tokens depending on who else is in the batch. |
| 🔴 ⬜ | **Silent failure** | absent | Inference errors that return HTTP 200. Taxonomised in arXiv 2606.04594 as accuracy regression, inconsistent output and bogus output. |
| 🔴 ⬜ | **Straggler** | absent | |
| 🔴 ⬜ | **NCCL hang** | absent | |
| 🔴 ⬜ | **Cold start** | covered as a bullet | The entry needs the seconds, since that is what decides whether autoscaling is viable. |
| 🔴 ⬜ | **Service tier** | absent | One string field, four prices and four latency profiles. A buyer-side lever, not an engineering one. |
| 🔴 ⬜ | **Tokenizer fertility** | absent | Tokens per word, spanning 2.5x across 25 European languages. The same product costs 2.5x more per user in Athens than in London with no engineering change. |

### Chapter 12. Designing an inference system end to end

| | Term | Elsewhere | Why it matters here |
|---|---|---|---|
| 🟡 ⬜ | **SLO** | covered | The entry ties it to goodput, which he does not have. |
| 🔴 ⬜ | **Capacity planning** | absent as a calculation | Little's Law applied, which is the objective his chapter 1 states and never delivers. |

---

## Counting

**Seventy-four terms across all twelve chapters.** Forty-two are 🔴, meaning absent from the closest competing text and load-bearing here. Twenty-three are 🟡, where the entry wins by being correct or by carrying the arithmetic. Nine are ⚪.

**For comparison, the competing glossary carries 54 terms in 10,127 words**, harvested 2026-07-30. This list is longer before a single entry is written, and forty-two of its terms have no counterpart there at all.

**Three of the 🟡 exist because the competing text is wrong**, and the entry is the correction: attention complexity stated as linear, the INT4 speed claim contradicted by its own table, and tensor parallelism described two different ways by the same vendor.

**Write order: 🔴 first, chapter by chapter, since those are the pages that rank for a term nobody else has an entry for.**
