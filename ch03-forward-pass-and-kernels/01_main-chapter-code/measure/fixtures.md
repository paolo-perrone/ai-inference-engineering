# Fixtures: where every chapter 3 placeholder gets its number

Every `[FIXTURE: ...]` in the chapter 3 draft (`ch3 - The forward pass and the kernels that
run it.md`), quoted verbatim with the line it sat on when this table was written, mapped to the
subcommand that measures it and the field in `results/<id>.json` that holds the value. Fields
live under the file's `fields` key. `rows[impl=eager,label=mini].prefill_ms` means the entry of
the `rows` list with that `impl` and `label`; `sweep[*]` means every entry of `sweep`.

`python test_run.py` re-reads the draft and fails if any placeholder in it has no row here, or if
a row names a subcommand or field that `run.py` does not produce. Line numbers drift as the draft
is edited; the test prints a warning for a moved line and fails only on an unmapped placeholder.

46 placeholders, 46 rows.

| # | Line | Placeholder (verbatim) | Subcommand | Result field(s) | Note |
|---|---|---|---|---|---|
| 1 | 47 | `[FIXTURE: first token IDs of the mini-session prompt, measurement 16]` | `m16` | `first_ids` |  |
| 2 | 47 | `[FIXTURE: the tokenizer's count for the mini-session prompt, measurement 16]` | `m16` | `count` |  |
| 3 | 71 | `[FIXTURE: sampling defaults printed at startup, measurement 10]` | `m10` | `sampling_override_line, sampling_override_params` |  |
| 4 | 90 | `[FIXTURE: driver, measurement 1]` | `m01` | `driver` |  |
| 5 | 91 | `[FIXTURE: version, measurement 1]` | `m01` | `vllm_version` |  |
| 6 | 92 | `[FIXTURE: engine count, measurement 1]` | `m01` | `prompt_tokens_engine` |  |
| 7 | 94 | `[FIXTURE: prefill ms, median of 5, measurement 1]` | `m01` | `prefill_ms_median` |  |
| 8 | 95 | `[FIXTURE: decode ms, median of 5, measurement 1]` | `m01` | `decode_ms_median` |  |
| 9 | 98 | `[FIXTURE: spread across 5 runs, measurement 1]` | `m01` | `prefill_ms_spread, decode_ms_spread` |  |
| 10 | 104 | `[FIXTURE: the 0.16 s floor as a percent of prefill, measurement 4]` | `m04` | `floor_pct_of_prefill_draft_19T` | draft wording; the plan's 0.18 s / 22.1T variant is floor_pct_of_prefill_plan_22_1T |
| 11 | 104 | `[FIXTURE: the 47 ms floor as a percent of the step, measurement 3]` | `m03` | `floor_pct_of_step` | draft wording (floor / step); the plan's inverse is step_pct_of_floor |
| 12 | 120 | `[FIXTURE: TTFT and decode step time at 1,457, 4,096 and 8,192 tokens, measurement 5]` | `m05` | `sweep[*].ttft_ms_median, sweep[*].decode_step_ms_median` | rows labelled mini (the real prompt, about 1,457), 4096, 8192 |
| 13 | 134 | `[FIXTURE: TTFT, median of 5, measurement 2]` | `m02` | `ttft_ms_median` |  |
| 14 | 136 | `[FIXTURE: TPOT, median of 5, measurement 2]` | `m02` | `tpot_ms_median` |  |
| 15 | 171 | `[FIXTURE: TPOT, measurement 2]` | `m02` | `tpot_ms_median` |  |
| 16 | 171 | `[FIXTURE: mean ITL, measurement 2]` | `m02` | `mean_itl_ms` |  |
| 17 | 171 | `[FIXTURE: tokens per streamed chunk, measurement 2]` | `m02` | `tokens_per_chunk_min, tokens_per_chunk_max, tokens_per_chunk_mean` |  |
| 18 | 231 | `[FIXTURE: kernels per decode step and per layer, measurement 7]` | `m07` | `kernels_per_step, kernels_per_layer` |  |
| 19 | 233 | `[FIXTURE: output head kernel time, measurement 9]` | `m09` | `output_head_ms_median` | the longest kernel after the graph replay; output_head_kernel names it |
| 20 | 237 | `[FIXTURE: sampling kernels after the head under the bench, greedy and Qwen's defaults, measurement 9]` | `m09` | `sampling_kernels_bench_default, sampling_kernels_greedy, sampling_kernels_qwen_defaults` | greedy: `vllm serve` with `--generation-config vllm`, one request at temperature 0 |
| 21 | 266 | `[FIXTURE: kernel names under --enforce-eager, measurement 8]` | `m08` | `enforce_eager_layer_sequence, enforce_eager_vllm_kernels_found` |  |
| 22 | 272 | `[FIXTURE: kernels per layer in transformers eager, measurement 12]` | `m12` | `kernels_per_layer` |  |
| 23 | 278 | `[FIXTURE: kernel names for the add and RMSNorm in transformers eager, measurement 12]` | `m12` | `add_and_norm_kernel_names` |  |
| 24 | 332 | `[FIXTURE: peak memory for eager and flash_attention_2 at the mini-session length, measurement 14]` | `m14` | `peak_gb_eager_mini, peak_gb_fa2_mini` |  |
| 25 | 343 | `[FIXTURE: driver, m14]` | `m14` | `driver` |  |
| 26 | 344 | `[FIXTURE: torch and transformers versions, m14]` | `m14` | `torch_version, transformers_version` |  |
| 27 | 346 | `[FIXTURE: ms, m14]` | `m14` | `rows[impl=eager,label=mini].prefill_ms` |  |
| 28 | 346 | `[FIXTURE: GB, m14]` | `m14` | `rows[impl=eager,label=mini].peak_gb` |  |
| 29 | 347 | `[FIXTURE: ms, m14]` | `m14` | `rows[impl=eager,label=4096].prefill_ms` |  |
| 30 | 347 | `[FIXTURE: GB, m14]` | `m14` | `rows[impl=eager,label=4096].peak_gb` |  |
| 31 | 348 | `[FIXTURE: OOM or not, m14]` | `m14` | `rows[impl=eager,label=8192].oom` | true, or the row's prefill_ms and peak_gb when it ran |
| 32 | 349 | `[FIXTURE: ms, m14]` | `m14` | `rows[impl=flash_attention_2,label=mini].prefill_ms` |  |
| 33 | 349 | `[FIXTURE: GB, m14]` | `m14` | `rows[impl=flash_attention_2,label=mini].peak_gb` |  |
| 34 | 350 | `[FIXTURE: ms, m14]` | `m14` | `rows[impl=flash_attention_2,label=4096].prefill_ms` |  |
| 35 | 350 | `[FIXTURE: GB, m14]` | `m14` | `rows[impl=flash_attention_2,label=4096].peak_gb` |  |
| 36 | 351 | `[FIXTURE: ms, m14]` | `m14` | `rows[impl=flash_attention_2,label=8192].prefill_ms` |  |
| 37 | 351 | `[FIXTURE: GB, m14]` | `m14` | `rows[impl=flash_attention_2,label=8192].peak_gb` |  |
| 45 | 352 | `[FIXTURE: longest that ran, m14]` | `m14` | `longest_ran_eager` | from the eager out-of-memory search, `oom_search_eager` holds every probe |
| 46 | 353 | `[FIXTURE: OOM length to 256 tokens, m14]` | `m14` | `oom_length_eager` | null when 32,768 tokens ran; the chapter's proximal win |
| 38 | 354 | `[FIXTURE: free memory after loading the bf16 model, measurement 15]` | `m15` | `free_gb` |  |
| 39 | 356 | `[FIXTURE: prefill time gap between eager and flash_attention_2 at the mini-session length, measurement 14]` | `m14` | `prefill_gap_ms_eager_minus_fa2_mini` |  |
| 40 | 376 | `[FIXTURE: the FLASH_ATTN backend line, measurement 10]` | `m10` | `flash_attn_backend_line` |  |
| 41 | 376 | `[FIXTURE: attention kernel names in prefill and decode, measurement 11]` | `m11` | `attention_kernels_prefill, attention_kernels_decode` |  |
| 42 | 392 | `[FIXTURE: decode step time under Qwen's defaults against greedy, as a ratio, measurement 17]` | `m17` | `ratio_qwen_defaults_over_greedy` | exercise 1 since review 5; m06 still times the graphs pair chapter 2 prints |
| 43 | 391 | `[FIXTURE: decode step time and prefill time, transformers eager against torch.compile, measurement 13]` | `m13` | `hf_eager_decode_step_ms, hf_eager_prefill_ms, hf_compile_decode_step_ms, hf_compile_prefill_ms` |  |
| 44 | 391 | `[FIXTURE: prefill time, vLLM default against custom ops, measurement 13]` | `m13` | `vllm_default_prefill_ms, vllm_custom_ops_all_prefill_ms` |  |

## Where the writing plan and the draft disagree

The draft's placeholder text decides what is measured. Where the plan asks for something else,
the code measures both, so the author can choose.

| Topic | Draft (and chapter 2) | Writing plan | What is measured |
|---|---|---|---|
| Prefill operations and floor | 19 trillion operations, 0.16 s (ch2 says the same) | 22.1 trillion, 0.18 s | m04 computes both, plus the count rescaled to the engine's real prompt length (ch2 step 5) |
| Graphs-off switch | `--compilation-config '{"cudagraph_mode": "NONE"}'` (ch2's switch) | `--enforce-eager` | m06 times both against the default; m08 traces both |
| Measurement 3 direction | "the 47 ms floor as a percent of the step" (floor / step) | "decode step time as a percent of the 47 ms floor" (step / floor) | m03 records both |
| Measurement 4 direction | "the 0.16 s floor as a percent of prefill" (floor / prefill) | "prefill time as a percent of the 0.18 s floor" | m04 records both directions for both counts |
| Measurement 13 | transformers eager against `torch.compile` (flash_attention_2 in both), then vLLM default against `-cc.custom_ops=["all"]` | vLLM default against `-O0` and against `-cc.custom_ops=["all"]` | m13 runs all four vLLM and transformers arms |
| Measurement 14 implementations | eager and flash_attention_2 in the printout | eager, sdpa and flash_attention_2 | m14 runs all three |
| Measurement 5 lengths | 1,457, 4,096 and 8,192 | "several prompt lengths up to 8k" | m05 runs the real prompt, 4,096 and 8,192 |
| The split script | `measure.py --split` in the text | `measure.py --split` | `run.py m01`; chapter 2's `measure.py` has no `--split` flag and this task did not edit it |
| The ablation script | `attention_ablation.py` in the chapter 3 folder | the same | `run.py m14` (worker `hf_worker.py ablation`); no file of that name exists yet |
| Where results land | "Write the out-of-memory prompt length into the fixture's baseline.json" | `results/ch03.json` | `results/m01.json` ... `results/m16.json`; nothing writes baseline.json |
| Greedy setting | the printout says greedy | every run with `--generation-config vllm` | every engine passes `--generation-config vllm` and each request sets `temperature: 0`; chapter 2's `measure.py --serve` set only `temperature: 0`, so Qwen's repetition penalty of 1.05 stayed on in chapter 2's runs |
| The prompt length | "about 1,457" (characters over 4) | 1,457 | m16 and m01 record the tokenizer's and the engine's real counts; the `vllm bench latency` traces keep the draft's `--input-len 1457` of random token IDs |
