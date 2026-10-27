# Chapter 5 measurements

Every measured number chapter 5 quotes, produced on one NVIDIA L4 under vLLM 0.30.0: the
precision table (bf16, FP8, 4-bit AWQ), the flip rate per quantized model on the agent's
recorded turns, and TPOT with suffix decoding off and on. One entry point, one subcommand
per measurement, one result file per subcommand.

```
run.py            the entry point: python run.py <id>, --list, --dry-run, all
common.py         pins, checkpoints, vLLM log parsers, servers, statistics, run log
turns.py          the recording format, flip and validity scoring, KL, the 128k needle probe
record_turns.py   the recorder: a proxy in front of the bf16 server that logs the agent's turns
test_run.py       parser and CLI tests, plus the check that fixtures.md maps the whole draft
fixtures.md       every [FIXTURE: ...] in the draft, with the subcommand and field that fill it
results/          <id>.json per measurement; results/raw holds server logs and bench files
recordings/       the agent's recorded turns (not committed: they hold your code and prompts)
```

It reuses `examples/mini-session/measure.py` (the mini-session prompt, the batch-1 timing,
the environment probe) rather than copying it.

## The measurements

`python run.py --list` prints the ids in the order to run them. Each one is independent: it
starts its own servers, writes `results/<id>.json`, and a failure in one never stops the
next. Exit status: 0 ok, 1 failed or partial, 2 blocked by a missing prerequisite.

| id | what it measures | recording? |
|---|---|---|
| m00 | preflight: versions, disk, how vLLM resolves each 128k config (CPU only) | no |
| m01 | kernel selected per precision: the AWQ and FP8 startup lines; does this build put FP8 on CUTLASS | no |
| m02 | weights loaded per precision, from vLLM's model-loading line | no |
| m04 | 128k sessions that fit, the cache size line, the activation and CUDA graph reserve | no |
| m03 | bytes read per decode step and the bandwidth they imply at batch 1 | no |
| m05 | prefill time at 8k, 32k and 128k per precision | no |
| m06 | decode TPOT at batch 1, 4, 8 and 16 per precision, plus a warm 128k session at batch 1 | no |
| m09 | the 128k multi-key needle probe per precision | no |
| m10 | Qwen2.5-Coder-1.5B refused at 128k; SWE-Dev-7B against stock Coder-7B step cost | no |
| m07 | flips per quantized model against bf16 twins | yes |
| m08 | mean per-token KL and tool-call validity | yes |
| m11 | speculation off, n-gram and suffix: TPOT, acceptance length, turn time, warm and cold, concurrency 1 and 4; greedy exact match; cache size line off and on | yes |
| m12 | prefill against decode seconds on one warm and one cold turn; suffix tree build time on a 128k prompt | yes |
| m13 | suffix decoding's speedup at concurrency 1 to 16, and where it falls to 1x | yes |
| m14 | strict tool calls off and on: first and second call TTFT, TPOT, parse failures | yes, with tools |
| m15 | the widest spread across five runs, over every result | no GPU |
| report | the precision table, flip blocks and TPOT pair, as text | no GPU |

Every timed cell is the median of five runs, stored with its spread (max minus min, as a
percent of the median) and all five values. Startup facts (loading, cache size, kernels) are
read once per server start. Each result's `run_log` names the GPU and driver (nvidia-smi),
the CUDA version the driver supports, torch's CUDA build, the vLLM, PyTorch, transformers,
XGrammar, llguidance and arctic-inference versions, every command issued with its flags, and
every checkpoint id with its revision.

## The prerequisite: the agent's recorded turns

m07, m08 and m11 to m14 replay "the agent's own recorded turns". No recording exists in this
repository yet, so those six measurements stop with status `blocked` until you make one:

1. Start the bf16 server with the tool parser:
   ```
   vllm serve Qwen/Qwen2.5-7B-Instruct --kv-cache-dtype fp8 --max-model-len 131072 \
       --hf-overrides "$YARN" --enable-auto-tool-choice --tool-call-parser hermes --port 8000
   ```
   (`$YARN` is the override in `common.py`, printed by `python run.py m01 --dry-run`.)
2. Start the recorder in front of it and point the agent's OpenAI base URL at the recorder:
   ```
   python record_turns.py --upstream http://127.0.0.1:8000 --port 9000 --repo /path/to/repo
   OPENAI_BASE_URL=http://127.0.0.1:9000/v1 <run the agent>
   ```
3. Run the agent on real tasks until `recordings/turns.jsonl` holds a few hundred turns, some
   of them with 64k to 128k tokens of repository context (the plan asks for both), and at
   least one editing session of several turns for m11 to m13.

A turn is scored on its action (the tool it called and the arguments, or its final text)
unless its record carries a `check`: a shell command that runs in a checkout of the recorded
commit with the replayed tool calls in `$TOOL_CALLS_JSON`, exit 0 meaning the outcome passed.
With checks, m07 also reports right-to-wrong and wrong-to-right flips. `turns.py` documents
the format.

## Running it on a fresh Colab L4 runtime

Pick an L4 runtime (pay-as-you-go). In a cell:

```bash
!nvidia-smi                       # check the driver's CUDA version before installing
!git clone <the companion repo> ai-inference-engineering
%cd ai-inference-engineering/ch05-model-level-optimization/01_main-chapter-code/measure
!pip install vllm==0.30.0 --extra-index-url https://download.pytorch.org/whl/cu129
!pip install --no-deps arctic-inference==0.1.1   # m11 to m14 only; see the risks below
!python test_run.py
!python run.py all --dry-run
```

The `--extra-index-url .../cu129` line is the install vLLM 0.30.0's own docs give
(`docs/getting_started/installation/gpu.cuda.inc.md:30`); it keeps PyTorch on CUDA 12.9.
Compare it with the "CUDA Version" nvidia-smi prints: the driver has to support at least
the build. If the driver supports CUDA 13, the plain `pip install vllm==0.30.0` also works.

Then, for a driver that runs each measurement and saves results after each one:

```python
import subprocess, shutil
from google.colab import drive
drive.mount('/content/drive')
ids = subprocess.run(["python", "run.py", "--list"], capture_output=True, text=True).stdout.split()
for i in ids:
    subprocess.run(["python", "run.py", i])
    shutil.copytree("results", "/content/drive/MyDrive/ch05-results", dirs_exist_ok=True)
```

### Downloads and disk

| checkpoint (exact HF id) | revision | download | used by |
|---|---|---|---|
| `Qwen/Qwen2.5-7B-Instruct` | a09a3545 | 15.24 GB | bf16 and FP8 (FP8 quantizes these weights as they load) |
| `Qwen/Qwen2.5-7B-Instruct-AWQ` | b2503754 | 5.58 GB | 4-bit |
| `Qwen/Qwen2.5-Coder-1.5B` | df3ce67c | 3.10 GB | m10 refusal (vLLM refuses at config time, so the weights are probably never fetched) |
| `Qwen/Qwen2.5-Coder-7B-Instruct` | c03e6d35 | 15.24 GB | m10 step cost |
| `zai-org/SWE-Dev-7B` | d04b618d | 15.25 GB | m10 step cost |

Total 54.4 GB of checkpoints (sizes from the Hugging Face API, 2026-10-27), plus vLLM,
PyTorch and the CUDA libraries (the vLLM 0.30.0 wheel alone is 315 MB), plus `results/raw` (bench files and server logs, and
the m07/m08 replays, which hold top-5 log probabilities for every replayed output token).
Budget 80 GB of free disk; m00 reports the free space it found. Qwen publishes no FP8
checkpoint of Qwen2.5-7B-Instruct, so FP8 needs no extra download. `THUDM/SWE-Dev-7B`
now redirects to `zai-org/SWE-Dev-7B`, the id used here.

### GPU minutes

These are planning estimates from roofline arithmetic, not measurements: a server start of
about 4 minutes (load, compile, CUDA graph capture; add download time on the first start of
each checkpoint), and a 128k prefill of about 90 seconds on the L4 (about 5 x 10^15
operations at roughly half of the 121 dense bf16 TFLOPS). Replace them with the
`wall_minutes` each result file records after the first run.

| id | estimate | what drives it |
|---|---|---|
| m00 | 2 min | CPU only, but the runtime is billed |
| m01, m02, m04 | 15 min together | three server starts, shared through `results/raw/startup-*.json` |
| m03 | 15 min | three starts, six short requests each |
| m05 | 45 min | three starts, six 128k prefills per precision |
| m06 | 35 min | three starts, 24 waves of short requests and two 128k prefills per precision |
| m09 | 35 min | three starts, five 128k probes per precision |
| m10 | 20 min | three refusals, two starts, plus about 31 GB of downloads |
| **without a recording** | **about 2.8 hours** | |
| m07 + m08 | 2 to 8 hours for 300 turns | three replays of every turn plus a teacher-forced pass; long turns cost about 90 s each cold, prefix cache hits between turns of one session cut that |
| m11 | 3 to 15 hours | three servers x four cells x five repeats of the session; cold concurrency 4 at 128k is most of it (`--session-turns 3` or `--no-cold-c4` cut it) |
| m12 | 25 min | one start, ten 128k prefills |
| m13 | 3 to 6 hours | two servers x five concurrencies x five repeats; 16 copies of a 128k session do not fit a pool of about three, so vLLM preempts and recomputes |
| m14 | 1 to 2 hours | two starts, 20 compile probes, TPOT on 20 turns x 10, parse pass over every tool turn twice |
| m15, report | 0 | reads results |
| **with a recording** | **about 12 to 34 hours** | dominated by context length and session length |

## What does not fit one Colab L4, and the fallback

- **The driver against the CUDA build.** vLLM 0.30.0's default CUDA is 13.0 in its source
  (`envs.py:90`) and 12.9 in its docs; CUDA 13 needs an R580 driver. Check `nvidia-smi` first
  and install from the cu129 index as above. m01 records torch's CUDA build; the CUTLASS FP8
  kernel needs 12.4 or newer on an L4 (`csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_entry.cu:145-154`), so cu129 is fine.
- **arctic-inference.** vLLM 0.30.0 asks for `arctic-inference==0.1.1`
  (`config/speculative.py:1581`), which ships only as source: pip compiles it, and its build
  pulls `torch==2.7.0` into an isolated build environment. If that build fails, try
  `pip install --no-deps arctic-inference` (0.3.0, also source only). m11 records the version
  it ran with. Its vLLM plugin stays off unless `ARCTIC_INFERENCE_ENABLED=1`; the servers here
  unset that variable, so the plugin's old-vLLM patches never load.
- **The 128k servers.** vLLM 0.30.0 does not scale `max_position_embeddings` for YaRN
  (`config/model.py:2468`) and refuses `--max-model-len` above it, so the override here also
  sets `max_position_embeddings` to 131072. m00 checks this resolution on the CPU before any
  GPU minute is spent. If the bf16 server cannot fit one 128k session next to its 15 GB of
  weights, its m04 line says so, and that refusal is the result: the draft predicts about 1.4
  sessions.
- **Memory under load.** An out-of-memory at a 128k prefill or at concurrency 16 lands in the
  cell's `error` field and the run moves on; it is a result, not a crash.
- **Session length.** Colab runtimes disconnect. The per-measurement result files and the
  Drive copy above make a disconnect cost one measurement, not the run. Recorded sessions
  near 128k make m11 and m13 the long poles; `--session-turns` caps them.
- **Coder-7B-Instruct and SWE-Dev at 128k.** Coder-7B-Instruct's tokenizer config caps it at
  32,768 tokens, which vLLM honours (`config/model.py:2418-2423`), so m10's step-cost pair runs
  both at 32,768. Per-step cost at batch 1 on the mini-session prompt does not need 128k.

## Where this code departs from the draft or the plan

Read from vLLM 0.30.0's source, the commands the draft prints need these changes. Each one is
also noted in the code:

1. `--tool-strict-level` does not exist in vLLM 0.30.0. Strict tool calling is per tool in the
   request (`"strict": true`), which turns on the hermes structural tag under automatic tool
   choice (`tool_parsers/structural_tag_registry.py:120`); `VLLM_ENFORCE_STRICT_TOOL_CALLING`
   (default on, `envs.py:1751`) then enforces the parameter schemas. There are no `function`
   and `parameter` levels to choose between.
2. The 128k prefill point is 131,008 input tokens: 131,072 plus 64 output tokens is longer than
   `--max-model-len 131072`, and vLLM rejects such a request.
3. `--num-prompts 5` at `--max-concurrency 16` never makes a batch of 16. m06 sends one wave of
   `c` prompts per run, five runs per concurrency.
4. The override is `rope_parameters` plus `max_position_embeddings`, not the plan's
   `rope_scaling` (see the risks above).
5. The scripts the draft names, `precision_table.py`, `flips.py` and `replay_session.py` under
   `examples/ch06/`, are `python run.py report`, `m07`/`m08`/`m09` and `m11` here.
6. The draft says the needle probe "is already in the companion repository"; it was not, so
   `turns.py` carries one: four keys at fixed depths in a filler document fitted to just under
   131,072 tokens, five seeds, pass when all five pass.

## Tests

```
python test_run.py
```

No GPU, no network, no vLLM. Every sample log line is rendered from the format string vLLM
0.30.0 prints it with. The fixture test reads the chapter draft from the author's vault, or
from `$CH5_DRAFT`, and is skipped where neither exists.
