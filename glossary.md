# The inference engineering glossary

Every term used in *AI Inference Engineering: Serving Foundation Models in Production*
(Manning, in progress), defined once, in one place you can link to.

**The rule for every entry: if a term has a formula, the formula is here and it is worked
on a real model.** A glossary that says "the KV cache stores keys and values for previous
tokens" has told you nothing you could act on. One that shows 84 GB for Llama 70B at 128k
context, against 140 GB of weights, has told you why your batch size is what it is.

**Why this lives in a repo rather than only in the book's appendix.** A definition you can
link to is worth more than a definition you can cite. An engineer who never buys the book
can still get the answer, and a reader can send a colleague a URL rather than a page
number.

## Status

**The index below is complete. The entries are being written.**

Each term gets its entry as its chapter gets written, never in a separate pass at the end.
A term defined by someone who has just spent a week inside the mechanism reads differently
from one defined from memory, and the difference is the whole value of the thing.

180 terms, twelve chapters. If you want a particular entry written next, open an issue
naming the term.

---

## Part 1. Foundations

### Chapter 1. Why inference is its own discipline

Context window · Cost per completed task · Cost per token · Goodput · Inference · Inference engineering · ISL and OSL · Latency · Little's Law · Percentile (p50, p90, p99) · Prefill-heavy and decode-heavy · Prompt · Serving · Tail latency · Throughput · Time per output token (TPOT) · Time to first token (TTFT) · Token

### Chapter 2. How GPUs run models

Arithmetic intensity · Compute-bound · CUDA core · CUDA graph · FLOPs against FLOPS · GDDR against HBM · HBM · Kernel · Kernel launch overhead · L1 and L2 cache · Machine balance point · Memory bandwidth · Memory-bound · Occupancy · Ops:byte ratio · Overhead-bound · Register file · Roofline model · SRAM · Streaming multiprocessor (SM) · Tensor core · TFLOPS · Thread block · VRAM · Warp

### Chapter 3. The forward pass and the kernels that run it

Attention head · Autoregressive generation · Causal mask · Decode · Decoder-only · Embedding · Feed-forward network · FlashAttention · Forward pass · Hidden state · Kernel fusion · Layer normalisation · Logits · Matrix-matrix against vector-matrix · Memory round trip · Prefill · Query, key, value · Residual stream · Sampling, temperature, top-k, top-p · Softmax · Tiling · Tokenizer · Transformer block

### Chapter 4. The accelerator landscape

Compiler-first · CPU offload · CUDA moat · Dense versus sparse FLOPS · Inferentia and Trainium · MIG · NPU · NVLink, NVSwitch, InfiniBand · ROCm · Systolic array · Thermal and power budget · TPU · Unified memory

## Part 2. Serving a single model

### Chapter 5. Model-level optimization

Acceptance rate · Calibration set · Constrained generation · Distillation · Draft model · FP16 and BF16 · FP8 · INT4 and INT8 · Perplexity against task accuracy · Pruning · Quantization-aware training versus post-training quantization · Response caching · Speculative decoding · Weight-only quantization

### Chapter 6. The KV cache

Attention complexity · Block table · Cache hit rate · Fragmentation · Grouped-query attention (GQA) · KV cache eviction · KV cache size · KV head · KV quantization · Multi-head attention (MHA) · Multi-query attention (MQA) · PagedAttention · Prefix caching · Sliding-window attention · Streaming attention

### Chapter 7. Batching and scheduling

Admission control · Arrival rate · Batch size · Chunked prefill · Continuous batching · Dynamic batching · Generation stall · Head-of-line blocking · Iteration-level scheduling · Preemption · Queueing · Scheduler · Static batching · Token budget

### Chapter 8. Inference engines as systems

Cascade and routing · Engine · Kernel backend · KV manager · LoRA adapter · Multi-LoRA serving · Multi-tenancy · RadixAttention · Request lifecycle · SGLang · TensorRT-LLM · The build-buy line · vLLM

## Part 3. Scale

### Chapter 9. Multi-GPU parallelism

Activation memory · Capacity factor · Communication volume · Context and sequence parallelism · Data parallelism · Expert parallelism · Expert routing · Mixture of experts (MoE) · Pipeline parallelism · Sharding · Tensor parallelism

### Chapter 10. Coordination and cluster scale

All-reduce, all-gather, all-to-all · Bandwidth against latency in a fabric · Collective operation · Disaggregated prefill and decode · Interconnect topology · KV cache transfer · Load balancing · NCCL · Overlap of communication and compute · Prefill-decode disaggregation · Ring and hierarchical collectives

## Part 4. Production

### Chapter 11. Operating inference in production

Autoscaling · Cold start · Determinism under batching · Hot swap · Load generator · MLPerf Inference · Model deprecation · Model FLOPs utilisation (MFU) · NCCL hang · Nondeterminism · Service tier · Showback and chargeback · Silent failure · SLO and SLA · Straggler · Tokenizer fertility · Warm pool · Weight loading

### Chapter 12. Designing an inference system end to end

Capacity planning · Error budget · Headroom · SLO · Workload characterisation
