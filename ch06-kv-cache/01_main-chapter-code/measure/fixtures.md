# Chapter 6 placeholders and the fields that fill them

Every `[FIXTURE: ...]` in the chapter 6 draft (`ch6 - The KV cache.md`), quoted verbatim
with its line number, mapped to the subcommand that measures it and the field in
`results/<id>.json` that holds the value. `python3 test_run.py` re-reads the draft and fails
if a placeholder is missing from this table or its line number has moved.

Paths use dots for nesting: `bf16_4096.max_concurrency` is `result["bf16_4096"]["max_concurrency"]`.

| Line | Placeholder | Subcommand | Result field |
|---|---|---|---|
| 24 | [FIXTURE: first line of vLLM's startup error, the bf16 start at 131,072 tokens] | m02 | `error_first_line`; `error_first_line_as_printed` is what the draft's printed command gives (see README, "The 128k command") |
| 115 | [FIXTURE: KV cache needed in GiB, the bf16 start at 131,072] | m02 | `kv_cache_needed_gib` |
| 115 | [FIXTURE: available KV cache memory in GiB, same run] | m02 | `kv_cache_available_gib` |
| 115 | [FIXTURE: estimated maximum model length in tokens, same run] | m02 | `estimated_max_model_len` |
| 117 | [FIXTURE: total device memory in MiB, nvidia-smi on the L4] | m01 | `memory_total_mib` |
| 169 | [FIXTURE: available KV cache memory in GiB, bf16 start at 32,768] | m03 | `available_kv_cache_gib` |
| 169 | [FIXTURE: GPU KV cache size in tokens, same run] | m03 | `kv_cache_size_tokens` |
| 169 | [FIXTURE: maximum concurrency, same run] | m03 | `max_concurrency` |
| 179 | [FIXTURE: model weights loaded in GiB, startup log of the bf16 run] | m03 | `model_weights_gib` |
| 179 | [FIXTURE: activation peak in GiB, same log] | m03 | `activation_peak_gib` (logged "peak activation" minus the CUDA graph estimate; the logged value is `peak_activation_as_logged_gib`) |
| 179 | [FIXTURE: non-torch memory in GiB, same log] | m03 | `non_torch_gib` (derived: v0.30.0 logs weights + non-torch together) |
| 179 | [FIXTURE: CUDA graph memory in GiB, same log] | m03 | `cuda_graph_gib` (the estimate the pool used); `cuda_graph_actual_gib` is the captured size |
| 195 | [FIXTURE: maximum concurrency at 4,096 and 32,768 tokens, bf16 and fp8 starts] | m04, m05 | m04 `bf16_4096.max_concurrency`, `bf16_32768.max_concurrency`; m05 `fp8_4096.max_concurrency`, `fp8_32768.max_concurrency` |
| 203 | [FIXTURE: sessions running over one shared 120k prefix in fp8, shared-prefix run] | m10 | `sessions_running_when_admission_stalled` (`sessions_running_peak` if it never stalled) |
| 223 | [FIXTURE: KV block size in tokens, bf16 startup log] | m03 | `block_size_tokens` (from `/metrics`, since v0.30.0 does not log it) |
| 239 | [FIXTURE: GPU KV cache size divided by the block size, bf16 start at 32,768] | m03 | `blocks_size_over_block`; the engine's own count is `num_gpu_blocks` |
| 245 | [FIXTURE: turn one and turn two TTFT in ms, prefix caching off, median of five runs] | m07 | `off_pair_ms` |
| 255 | [FIXTURE: spread across five runs, turn one TTFT, caching off] | m07 | `turn_one_spread_off_ms` |
| 255 | [FIXTURE: GPU, driver and CUDA version, run log] | m07 | `run_log.environment.gpus.name`, `run_log.environment.gpus.driver`, `run_log.environment.cuda_supported_by_driver`, `run_log.environment.packages.torch_cuda` |
| 269 | [FIXTURE: turn one and turn two TTFT in ms, prefix caching on, median of five runs] | m07 | `on_pair_ms` |
| 269 | [FIXTURE: prefix-cache hit tokens on turn two, and the counter names on vLLM v0.30.0] | m08 | `turn_two_prefix_cache_hits_tokens`, `counter_names` |
| 277 | [FIXTURE: turn one and turn two TTFT in ms over the 32k repository prefix, caching off] | m09 | `off_pair_ms` |
| 277 | [FIXTURE: turn one and turn two TTFT in ms over the 32k repository prefix, caching on] | m09 | `on_pair_ms` |
| 289 | [FIXTURE: first agent's turn two TTFT in ms, second agent running, offloading off and on] | m11 | `pair_ms` |
| 305 | [FIXTURE: GPU KV cache size in tokens, fp8 start at 131,072] | m05 | `fp8_131072.kv_cache_size_tokens` |
| 305 | [FIXTURE: maximum concurrency for 131,072 tokens, same run] | m05 | `fp8_131072.max_concurrency` |
| 321 | [FIXTURE: attention backend line, bf16 and fp8 starts] | m06 | `bf16.line`, `fp8.line` |
| 323 | [FIXTURE: TPOT in ms, bf16 and fp8 cache, at 32k and at the longest length the bf16 start accepts] | m12 | `bf16_32k.tpot_ms.median`, `fp8_32k.tpot_ms.median`, `bf16_longest_bf16.tpot_ms.median`, `fp8_longest_bf16.tpot_ms.median` |
| 327 | [FIXTURE: retrieval recovery at 32k, 64k and 131,072, fp8 and bf16] | m13 | `recovery.fp8.32768.percent`, `recovery.fp8.65536.percent`, `recovery.fp8.131072.percent`, `recovery.bf16.32768.percent`, `recovery.bf16.65536.percent` |
| 351 | [FIXTURE: turn two TTFT in ms over the 32k prefix, caching off] | m09 | `turn_two_off_ms` |
| 351 | [FIXTURE: turn two TTFT in ms over the 32k prefix, caching on] | m09 | `turn_two_on_ms` |
| 352 | [FIXTURE: X GiB needed against Y GiB available] | m02 | `x_against_y` |
| 352 | [FIXTURE: maximum concurrency, fp8 start] | m05 | `fp8_131072.max_concurrency` |
| 352 | [FIXTURE: recovery at 131,072 in fp8] | m13 | `recovery.fp8.131072.percent` |

34 placeholders, 34 mapped.

## Notes the draft needs before its numbers go in

- Line 24 and 115: in v0.30.0 the draft's printed command (`"rope_scaling"`, no
  `max_position_embeddings`) is expected to stop at the max-length check, before the KV
  check, because a YaRN rope type no longer scales `max_position_embeddings`. m02 runs both
  forms and records each first error line; X, Y and Z come from whichever start reached the
  KV check.
- Lines 163 to 167: v0.30.0 prints the pool and the concurrency as one line,
  `GPU KV cache size: N tokens, Maximum concurrency for 32,768 tokens per request: C.CCx`,
  and "Available KV cache memory: Y GiB" on its own line. m03's `capacity_line` is the
  verbatim line.
- Line 179: "peak activation" in v0.30.0's summary line already contains the CUDA graph
  estimate, so subtracting both the logged peak and the graphs counts the graphs twice.
- Line 255: the run log is in every result; m07's is the one this section's numbers come from.
