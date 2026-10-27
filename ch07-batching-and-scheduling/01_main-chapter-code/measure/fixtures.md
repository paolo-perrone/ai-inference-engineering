# Chapter 7 placeholders, mapped to the runs that fill them

Every `[FIXTURE...]` in the chapter 7 draft, quoted verbatim with its line number, and the
subcommand and result field that produce it. `test_run.py` re-reads the draft (set
`CH7_DRAFT` to its path) and fails if a placeholder has no row here, or a row names a line
the draft no longer has. Values are medians of the runs unless the path says otherwise;
every median sits beside its `spread` and its per-run `values` in the same object.

Three placeholders are a bare `[FIXTURE]` with no label. Each sits in a figure caption that
says the measured values replace the figure's arithmetic heights: line 230 is figure 7.7
(table 7.3, runs A and B), line 272 is figure 7.9 (table 7.4, the shared-prefix sweep), and
line 358 is figure 7.11 (batch 1, the operating point, the same on a card idle half the day,
and 32 sessions at 4k).

Number formats: times in `_s` are seconds and in `_ms` milliseconds; `max_kv_cache_usage`
is vLLM's `vllm:kv_cache_usage_perc`, where 1 means 100 percent.

| # | line | placeholder, verbatim | subcommand | field key | result path (results/<subcommand>.json) |
|---|---|---|---|---|---|
| 1 | 24 | `[FIXTURE: follow-ups' p99 time to first token, run A]` | runA | `followup_ttft_p99_s` | fields.followup_ttft_p99_s.median |
| 2 | 42 | `[FIXTURE: TPOT p50 at batch 1, 32k sweep]` | sweep32k | `tpot_p50_ms` | fields.1.tpot_p50_ms.median |
| 3 | 48 | `[FIXTURE: NVIDIA driver and CUDA version, 32k sweep]` | sweep32k | `cuda_runtime` | environment.gpus[0].driver, environment.cuda_runtime, environment.cuda_supported_by_driver |
| 4 | 48 | `[FIXTURE: largest run-to-run spread across five runs in any table]` | summary | `largest_run_to_run_spread` | fields.largest_run_to_run_spread (field, median, spread, relative) |
| 5 | 54 | `[FIXTURE: TPOT p50, batch 1, 32k sweep]` | sweep32k | `tpot_p50_ms` | fields.1.tpot_p50_ms.median |
| 6 | 54 | `[FIXTURE: tokens/s, batch 1, 32k sweep]` | sweep32k | `tokens_per_s` | fields.1.tokens_per_s.median |
| 7 | 55 | `[FIXTURE: TPOT p50, batch 2, 32k sweep]` | sweep32k | `tpot_p50_ms` | fields.2.tpot_p50_ms.median |
| 8 | 55 | `[FIXTURE: tokens/s, batch 2, 32k sweep]` | sweep32k | `tokens_per_s` | fields.2.tokens_per_s.median |
| 9 | 56 | `[FIXTURE: TPOT p50, batch 4, 32k sweep]` | sweep32k | `tpot_p50_ms` | fields.4.tpot_p50_ms.median |
| 10 | 56 | `[FIXTURE: tokens/s, batch 4, 32k sweep]` | sweep32k | `tokens_per_s` | fields.4.tokens_per_s.median |
| 11 | 57 | `[FIXTURE: TPOT p50, batch 8, 32k sweep]` | sweep32k | `tpot_p50_ms` | fields.8.tpot_p50_ms.median |
| 12 | 57 | `[FIXTURE: running batch at max-num-seqs 8, 32k sweep]` | sweep32k | `running_batch_max` | fields.8.running_batch_max.median (also running_batch_median) |
| 13 | 78 | `[FIXTURE: 4k launch]` | launches | `max_concurrency_x` | fields.4k.max_concurrency_x (whole line: fields.4k.capacity_line; a refused launch: fields.4k.parsed.kv_refusal) |
| 14 | 79 | `[FIXTURE: 32k launch]` | launches | `max_concurrency_x` | fields.32k.max_concurrency_x (whole line: fields.32k.capacity_line; a refused launch: fields.32k.parsed.kv_refusal) |
| 15 | 80 | `[FIXTURE: 128k launch]` | launches | `max_concurrency_x` | fields.128k.max_concurrency_x (whole line: fields.128k.capacity_line; a refused launch: fields.128k.parsed.kv_refusal) |
| 16 | 93 | `[FIXTURE: TPOT p50, batch 4, shared 120k prefix, 8k tails]` | sweepshared | `tpot_p50_ms` | fields.4.tpot_p50_ms.median |
| 17 | 130 | `[FIXTURE: resolved value, run A log]` | runA | `max_num_seqs` | fields.resolved_scheduler.max_num_seqs (cross-check: launches fields.128k.default_max_num_seqs, DEBUG log) |
| 18 | 131 | `[FIXTURE: resolved value, run A log]` | runA | `max_num_batched_tokens` | fields.resolved_scheduler.max_num_batched_tokens (log line: fields.resolved_scheduler.log_chunked_prefill_max_num_batched_tokens) |
| 19 | 132 | `[FIXTURE: resolved value, run A log]` | runA | `async_scheduling` | fields.resolved_scheduler.async_scheduling (no log line prints it; config probe) |
| 20 | 162 | `[FIXTURE: completion time, static run]` | static | `static_completion_time_s` | fields.static_completion_time_s.median |
| 21 | 162 | `[FIXTURE: tokens/s, static run]` | static | `static_tokens_per_s` | fields.static_tokens_per_s.median |
| 22 | 163 | `[FIXTURE: completion time, continuous run]` | static | `continuous_completion_time_s` | fields.continuous_completion_time_s.median |
| 23 | 163 | `[FIXTURE: tokens/s, continuous run]` | static | `continuous_tokens_per_s` | fields.continuous_tokens_per_s.median |
| 24 | 191 | `[FIXTURE: follow-up TTFT p50, run A]` | runA | `followup_ttft_p50_s` | fields.followup_ttft_p50_s.median |
| 25 | 192 | `[FIXTURE: follow-up TTFT p99, run A]` | runA | `followup_ttft_p99_s` | fields.followup_ttft_p99_s.median |
| 26 | 193 | `[FIXTURE: vllm:request_queue_time_seconds p50, run A]` | runA | `followup_queue_time_p50_s` | fields.followup_queue_time_p50_s.median (histogram cross-check: fields.followup_queue_time_p50_s_from_histogram) |
| 27 | 194 | `[FIXTURE: total output tokens/s, run A]` | runA | `total_output_tokens_per_s` | fields.total_output_tokens_per_s.median |
| 28 | 195 | `[FIXTURE: GPU, driver and CUDA version, run A]` | runA | `cuda_runtime` | environment.gpus[0].name, .driver, environment.cuda_runtime, environment.cuda_supported_by_driver |
| 29 | 203 | `[FIXTURE: TTFT of one cold 32k prompt alone]` | cold32k | `cold_32k_ttft_s` | fields.cold_32k_ttft_s.median |
| 30 | 221 | `[FIXTURE: TTFT p50, run A]` | runA | `followup_ttft_p50_s` | fields.followup_ttft_p50_s.median |
| 31 | 221 | `[FIXTURE: TTFT p50, run B]` | runB | `followup_ttft_p50_s` | fields.followup_ttft_p50_s.median |
| 32 | 222 | `[FIXTURE: TTFT p99, run A]` | runA | `followup_ttft_p99_s` | fields.followup_ttft_p99_s.median |
| 33 | 222 | `[FIXTURE: TTFT p99, run B]` | runB | `followup_ttft_p99_s` | fields.followup_ttft_p99_s.median |
| 34 | 223 | `[FIXTURE: queue p50, run A]` | runA | `followup_queue_time_p50_s` | fields.followup_queue_time_p50_s.median |
| 35 | 223 | `[FIXTURE: queue p50, run B]` | runB | `followup_queue_time_p50_s` | fields.followup_queue_time_p50_s.median |
| 36 | 224 | `[FIXTURE: tokens/s, run A]` | runA | `total_output_tokens_per_s` | fields.total_output_tokens_per_s.median |
| 37 | 224 | `[FIXTURE: tokens/s, run B]` | runB | `total_output_tokens_per_s` | fields.total_output_tokens_per_s.median |
| 38 | 230 | `[FIXTURE]` | runA, runB | `followup_ttft_p99_s` | figure 7.7 bar heights: fields.followup_ttft_p50_s.median, fields.followup_ttft_p99_s.median, fields.total_output_tokens_per_s.median, both runs |
| 39 | 238 | `[FIXTURE: ITL p50, run A]` | runA | `followup_itl_p50_ms` | fields.followup_itl_p50_ms.median |
| 40 | 238 | `[FIXTURE: ITL p99, run A]` | runA | `followup_itl_p99_ms` | fields.followup_itl_p99_ms.median |
| 41 | 239 | `[FIXTURE: ITL p50, run B]` | runB | `followup_itl_p50_ms` | fields.followup_itl_p50_ms.median |
| 42 | 239 | `[FIXTURE: ITL p99, run B]` | runB | `followup_itl_p99_ms` | fields.followup_itl_p99_ms.median |
| 43 | 241 | `[FIXTURE: paste TTFT, run A]` | runA | `paste_ttft_s` | fields.paste_ttft_s.median |
| 44 | 241 | `[FIXTURE: paste TTFT, run B]` | runB | `paste_ttft_s` | fields.paste_ttft_s.median |
| 45 | 254 | `[FIXTURE: queue time of the second 128k request, and waiting count while the first prefills]` | admit128k | `second_queue_time_s` | fields.second_queue_time_s.median and fields.max_waiting_while_first_prefills.median |
| 46 | 272 | `[FIXTURE]` | sweepshared | `tokens_per_s` | figure 7.9 points: fields.1..6.tokens_per_s.median against fields.1..6.tpot_p99_ms.median |
| 47 | 274 | `[FIXTURE: NVIDIA driver and CUDA version, shared-prefix sweep]` | sweepshared | `cuda_runtime` | environment.gpus[0].driver, environment.cuda_runtime, environment.cuda_supported_by_driver |
| 48 | 286 | `[FIXTURE: tokens/s, shared-prefix sweep at 1]` | sweepshared | `tokens_per_s` | fields.1.tokens_per_s.median |
| 49 | 286 | `[FIXTURE: TPOT p50, shared-prefix sweep at 1]` | sweepshared | `tpot_p50_ms` | fields.1.tpot_p50_ms.median |
| 50 | 286 | `[FIXTURE: TPOT p99, shared-prefix sweep at 1]` | sweepshared | `tpot_p99_ms` | fields.1.tpot_p99_ms.median (pooled over runs: fields.1.tpot_p99_ms_pooled) |
| 51 | 287 | `[FIXTURE: tokens/s, shared-prefix sweep at 2]` | sweepshared | `tokens_per_s` | fields.2.tokens_per_s.median |
| 52 | 287 | `[FIXTURE: TPOT p50, shared-prefix sweep at 2]` | sweepshared | `tpot_p50_ms` | fields.2.tpot_p50_ms.median |
| 53 | 287 | `[FIXTURE: TPOT p99, shared-prefix sweep at 2]` | sweepshared | `tpot_p99_ms` | fields.2.tpot_p99_ms.median (pooled over runs: fields.2.tpot_p99_ms_pooled) |
| 54 | 288 | `[FIXTURE: tokens/s, shared-prefix sweep at 3]` | sweepshared | `tokens_per_s` | fields.3.tokens_per_s.median |
| 55 | 288 | `[FIXTURE: TPOT p50, shared-prefix sweep at 3]` | sweepshared | `tpot_p50_ms` | fields.3.tpot_p50_ms.median |
| 56 | 288 | `[FIXTURE: TPOT p99, shared-prefix sweep at 3]` | sweepshared | `tpot_p99_ms` | fields.3.tpot_p99_ms.median (pooled over runs: fields.3.tpot_p99_ms_pooled) |
| 57 | 289 | `[FIXTURE: tokens/s, shared-prefix sweep at 4]` | sweepshared | `tokens_per_s` | fields.4.tokens_per_s.median |
| 58 | 289 | `[FIXTURE: TPOT p50, shared-prefix sweep at 4]` | sweepshared | `tpot_p50_ms` | fields.4.tpot_p50_ms.median |
| 59 | 289 | `[FIXTURE: TPOT p99, shared-prefix sweep at 4]` | sweepshared | `tpot_p99_ms` | fields.4.tpot_p99_ms.median (pooled over runs: fields.4.tpot_p99_ms_pooled) |
| 60 | 290 | `[FIXTURE: tokens/s, shared-prefix sweep at 5]` | sweepshared | `tokens_per_s` | fields.5.tokens_per_s.median |
| 61 | 290 | `[FIXTURE: TPOT p50, shared-prefix sweep at 5]` | sweepshared | `tpot_p50_ms` | fields.5.tpot_p50_ms.median |
| 62 | 290 | `[FIXTURE: TPOT p99, shared-prefix sweep at 5]` | sweepshared | `tpot_p99_ms` | fields.5.tpot_p99_ms.median (pooled over runs: fields.5.tpot_p99_ms_pooled) |
| 63 | 291 | `[FIXTURE: tokens/s, shared-prefix sweep at 6]` | sweepshared | `tokens_per_s` | fields.6.tokens_per_s.median |
| 64 | 291 | `[FIXTURE: TPOT p50, shared-prefix sweep at 6]` | sweepshared | `tpot_p50_ms` | fields.6.tpot_p50_ms.median |
| 65 | 291 | `[FIXTURE: TPOT p99, shared-prefix sweep at 6]` | sweepshared | `tpot_p99_ms` | fields.6.tpot_p99_ms.median (pooled over runs: fields.6.tpot_p99_ms_pooled) |
| 66 | 303 | `[FIXTURE: pool use, shared-prefix, 6]` | sweepshared | `max_kv_cache_usage` | fields.6.max_kv_cache_usage.median (vllm:kv_cache_usage_perc, 1 = 100 percent) |
| 67 | 304 | `[FIXTURE: preemptions, shared-prefix, 6]` | sweepshared | `preemptions` | fields.6.preemptions.median (delta of vllm:num_preemptions_total over a run) |
| 68 | 313 | `[FIXTURE: operating point, shared-prefix sweep]` | summary | `operating_point_max_num_seqs` | fields.operating_point_max_num_seqs |
| 69 | 321 | `[FIXTURE: maximum concurrency at 4k]` | launches | `max_concurrency_x` | fields.4k.max_concurrency_x |
| 70 | 327 | `[FIXTURE: vllm:num_requests_running, 32k sweep]` | sweep32k | `running_batch_max` | fields.{1,2,4,8}.running_batch_max.median |
| 71 | 328 | `[FIXTURE: output tokens/s per point, 32k sweep]` | sweep32k | `tokens_per_s` | fields.{1,2,4,8}.tokens_per_s.median |
| 72 | 329 | `[FIXTURE: TPOT p99 per point, 32k sweep]` | sweep32k | `tpot_p99_ms` | fields.{1,2,4,8}.tpot_p99_ms.median |
| 73 | 344 | `[FIXTURE: answers per second at the operating point, shared-prefix sweep]` | summary | `answers_per_s_at_operating_point` | fields.answers_per_s_at_operating_point |
| 74 | 352 | `[FIXTURE: dollars per million output tokens at the operating point, shared-prefix sweep]` | summary | `dollars_per_million_at_operating_point` | fields.dollars_per_million_at_operating_point |
| 75 | 352 | `[FIXTURE: dollars per million output tokens at batch 1, shared-prefix sweep]` | summary | `dollars_per_million_at_batch_1` | fields.dollars_per_million_at_batch_1 |
| 76 | 358 | `[FIXTURE]` | summary | `figure_7_11_tokens_per_s` | figure 7.11 heights: fields.figure_7_11_tokens_per_s (batch 1, operating point, 4k at 32); idle half day = fields.dollars_per_million_at_operating_point_card_idle_half_the_day, arithmetic |
| 77 | 390 | `[FIXTURE: p50, run C]` | runC | `followup_ttft_p50_s` | fields.followup_ttft_p50_s.median |
| 78 | 390 | `[FIXTURE: p99, run C]` | runC | `followup_ttft_p99_s` | fields.followup_ttft_p99_s.median |
| 79 | 399 | `[FIXTURE: TTFT, run B]` | runB | `followup_ttft_p50_s` | fields.followup_ttft_p50_s.median |
| 80 | 399 | `[FIXTURE: TTFT, run D, async off]` | runD | `followup_ttft_p50_s` | fields.followup_ttft_p50_s.median |
| 81 | 400 | `[FIXTURE: TPOT, run B]` | runB | `followup_tpot_p50_ms` | fields.followup_tpot_p50_ms.median |
| 82 | 400 | `[FIXTURE: TPOT, run D, async off]` | runD | `followup_tpot_p50_ms` | fields.followup_tpot_p50_ms.median |
| 83 | 405 | `[FIXTURE: dollars per million output tokens, shared-prefix sweep]` | summary | `dollars_per_million_by_point` | fields.dollars_per_million_by_point |
