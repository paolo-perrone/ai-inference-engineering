# Chapter 7 measurements

Every measured number in chapter 7, produced on one NVIDIA L4 under vLLM 0.30.0 with
Qwen2.5-7B-Instruct and an fp8 KV cache. One entry point, one subcommand per run or sweep,
one result file per subcommand.

```
run.py         the entry point: python run.py <id>, --list, --dry-run, all
ch7lib.py      parsers, percentiles, the arrival schedule, the streaming client
traffic.json   every input, each with the draft line that defines it or UNDEFINED
fixtures.md    every [FIXTURE] placeholder in the draft -> subcommand and result field
test_run.py    parser, statistics and end-to-end tests against a fake vllm; no GPU
results/       <id>.json, written by each subcommand
runs/          the vllm serve log of every launch
```

It reuses chapter 1's fixture in `examples/mini-session/measure.py`: the prompt text, the
port and stop helpers, and the run log (card, driver, CUDA, vLLM). Prompts are that text,
tokenized by the server's own `/tokenize` and tiled to exact token counts.

## Run it on a fresh Colab L4 runtime

Runtime type: L4 GPU. Then, in cells:

```
!git clone https://github.com/paolo-perrone/ai-inference-engineering.git
%cd ai-inference-engineering/ch07-batching-and-scheduling/01_main-chapter-code/measure
!pip install -q vllm==0.30.0
!python test_run.py
!python run.py all --dry-run
```

The dry run prints every server command, the replay's arrival schedule and every
provisional input, and checks that each prompt fits its `--max-model-len` and that every
flag exists in vLLM 0.30.0's source. Then the measurements, one subcommand at a time, with
the results copied to Drive after each so a disconnect loses one run at most:

```python
from google.colab import drive; drive.mount("/content/drive")
import shutil, subprocess
ids = subprocess.run(["python", "run.py", "--list"], capture_output=True, text=True).stdout.split()
for i in ids:
    subprocess.run(["python", "run.py", i])
    shutil.copytree("results", "/content/drive/MyDrive/ch07-results", dirs_exist_ok=True)
    shutil.copytree("runs", "/content/drive/MyDrive/ch07-results/runs", dirs_exist_ok=True)
```

Each subcommand is independent: it starts its own engine, stops it, and writes its result
even when the launch fails (a refused or out-of-memory launch is recorded with its parsed
reason and the last 60 log lines, and the subcommand exits 1). `summary` runs last and
needs no GPU: it reads the other results and computes the operating point, dollars per
million output tokens and the largest run-to-run spread.

## The runs, and what they cost

GPU-minutes are estimates, not measurements. Each launch is counted at about 3 minutes
(weights load, torch.compile, CUDA graph capture, FlashInfer JIT); the first launch adds
about 4 for the 15 GB download. Prefill times use the draft's own arithmetic: about 10 s
for 32k (section 7.3), about 1 to 1.5 minutes for 120k to 128k by the same method.

| order | id | what it measures | launches | GPU-minutes |
|---|---|---|---|---|
| 1 | `launches` | resolved defaults, attention backend, KV pool, maximum concurrency at 4k, 32k, 128k | 3 | 15 |
| 2 | `cold32k` | one cold 32k prompt alone, TTFT, five times | 1 | 4 |
| 3 | `static` | eight answers, client batches of 4 against continuous batching | 1 | 8 |
| 4-7 | `runA` to `runD` | the team replay: defaults, step budget 8,192 + threshold 2,048, priority, async off | 4 | 52 |
| 8 | `admit128k` | two 128k requests together: the second's queue time, the waiting count | 1 | 19 |
| 9 | `sweep4k` | `--max-num-seqs` 1 to 32 at 4k | 6 | 26 |
| 10 | `sweep32k` | `--max-num-seqs` 1, 2, 4, 8 at 32k | 4 | 29 |
| 11 | `sweepshared` | `--max-num-seqs` 1 to 6 over a shared 120k prefix with 8k tails | 6 | 40 |
| 12 | `summary` | cost, operating point, spread | 0 | 0 |
| | **total** | | **26** | **about 193, 3.2 hours** |

The long tail is the 128k work. If the shared prefix is evicted between runs (at five or six
sessions the pool is close to full), every run pays a 120k prefill again, which can add 20
minutes to `sweepshared` and 10 to each replay. `prefix_cache_hit_rate` in the result says
whether that happened.

## What every result file holds

`fields` holds the numbers under the names fixtures.md cites. A measured field is an object:
`median` (of the runs), `spread` (largest minus smallest), `values` (every run) and `n`.
`environment` names the GPU, driver, CUDA runtime, the CUDA version the driver supports,
vLLM and PyTorch, and where each was read. `server_commands` and `client_command` are the
exact commands. `provisional_inputs` lists every input the draft does not define.

Definitions follow `vllm bench serve` (vllm/benchmarks/serve.py): TTFT is send to first
token, TPOT is (latency - TTFT) / (tokens - 1) per request, ITL is every gap between
streamed chunks, percentiles interpolate linearly as numpy does. Queue time is the server's
own per request (`--enable-per-request-metrics`, the interval
`vllm:request_queue_time_seconds` observes), with the histogram's p50 beside it as a
cross-check. Sweep throughput counts tokens only while every request decodes together; the
result says when that window never formed.

## Inputs the draft does not define

Each is in `traffic.json` with its source marked UNDEFINED, printed by every dry run and
copied into every result. Change the value there, not in code.

- the shared prefix's length: "120k" is 120,000 here, which matches the draft's 3.4 GB; the
  k = 1,024 reading (122,880) plus an 8k tail, a follow-up and its answer exceeds 131,072
- the follow-up arrival process: rate, replay length, when the paste lands; follow-ups do
  not carry earlier answers
- the cold paste's own answer length
- the static comparison's context, its eight answer lengths, and the continuous side's slot
  count (4 here, the same slots as the static side)
- what "128k request" and "32k context" and "4k context" include (here: the answer too)
- how many requests a sweep point sends (as many as its sequence cap) and how a p99 over one
  to six requests is reported

## Where the plan and the draft disagree (the draft decides)

- The 32k sweep runs 1, 2, 4, 8 (draft); the plan's item 14 says the same points as 4k, 1 to 32.
- The price comes from the shared-prefix sweep (draft); the plan priced the 4k sweep. The 4k
  sweep still runs, for figure 7.11's last bar and the 4k concurrency line.
- Section 7.2 shows `max_num_seqs=`, `max_num_batched_tokens=`, `async_scheduling=` read
  from the startup log. vLLM 0.30.0 prints only `Chunked prefill is enabled with
  max_num_batched_tokens=N.` at INFO; `Defaulting max_num_seqs to N` is DEBUG, and nothing
  prints async scheduling when it is on by default. The fixture reads all three from
  `create_engine_config()` and from a DEBUG launch, so the listing needs rewording.
- The draft's listings call `examples/mini-session/replay_mixed.py` and `sweep_seqs.py`
  with their own flags. Neither exists; the runs live here as `run.py runA` and the sweeps.
- The plan's attention backend (item 2) and pool size in GB (item 3) have no placeholder in
  the draft. `launches` records both; vLLM prints the pool in GiB.

## Risks on one Colab L4

- **The 128k launch.** fp8 plus YaRN should start (chapter 6). If it is refused, `launches`
  records the refusal and the estimated maximum length, and the 128k subcommands fail with
  the same record. Fallback: lower `prefix_tokens` and `tail_tokens` in traffic.json to fit
  the length vLLM estimates, and say so in the chapter.
- **The replay may not fit the pool.** Four sessions on 120k + 8k plus a cold 32k paste need
  about 5.3 GB of fp8 cache against a pool of roughly 5 GB. The paste can wait for admission
  or push the prefix out, and then the follow-ups measure a 120k recompute instead of the
  freeze. `max_waiting`, `preemptions` and `prefix_cache_hit_rate` show which; fallback:
  three sessions, or shorter tails.
- **Runtime limits.** About 3.2 hours in all; a disconnect costs only the running subcommand,
  because results go to Drive after each.
- **The CUDA stack.** vLLM 0.30.0 pins torch 2.13.0 and flashinfer 0.6.18.post1. If Colab's
  driver is older than the wheel's CUDA, the first launch fails; `launches` records it.
