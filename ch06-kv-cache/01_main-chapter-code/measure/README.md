# Chapter 6 measurements

Every measured number in chapter 6, "The KV cache", comes from this folder, run on one
NVIDIA L4 with vLLM 0.30.0 and Qwen/Qwen2.5-7B-Instruct. Each measurement is a subcommand
of `run.py` and writes `results/<id>.json`: the measured values under named fields, timings
as the median of five runs (after one discarded warm-up) with the spread, and a run log
(GPU name, driver, CUDA, vLLM and PyTorch versions, total device memory, every command and
flag). `fixtures.md` maps each `[FIXTURE: ...]` in the draft to its field.

```
run.py        the entry point: --list, one id, or all; --dry-run on every one
vlog.py       parsers for vLLM 0.30.0's startup lines and /metrics, each citing its source line
engine.py     starts vllm serve, keeps a refused start as a result, times streamed requests
prompts.py    the long prompts, built deterministically from the installed vLLM source
fixtures.md   every draft placeholder, its subcommand and its result field
test_run.py   parser, config, prompt and fake-engine tests; no GPU needed
```

It reuses the mini-session's `measure.py` for the turn-one prompt, the 180-token answer
length, the free-port and engine-stop helpers and the nvidia-smi reader.

## Run it on a fresh Colab L4 runtime

Runtime > Change runtime type > L4. Do not import torch in the notebook before the runs:
any memory the notebook holds on the GPU comes out of vLLM's 92 percent.

```bash
!git clone <companion repo> aie && cd aie && git checkout measure-first-ch06
!pip install uv && uv pip install --system vllm==0.30.0 --torch-backend=auto
%cd aie/ch06-kv-cache/01_main-chapter-code/measure
!python run.py m01            # preflight: read preflight_warnings in results/m01.json first
!python run.py all --dry-run  # every command, every prompt checked with the real tokenizer
```

Then run the ids in order, copying `results/` to Drive after each, so a disconnect loses at
most one measurement:

```python
import subprocess, shutil
ids = subprocess.run(["python", "run.py", "--list"], capture_output=True, text=True).stdout.split()
for i in ids:
    subprocess.run(["python", "run.py", i, "--skip-existing"])
    shutil.copytree("results", "/content/drive/MyDrive/ch06-results", dirs_exist_ok=True)
```

Each id is independent. Ids that only need a startup log (m02 to m06) share starts through
`results/logs/<name>.json`, so `all` starts each configuration once; `--fresh` ignores that
cache. m08 reuses m07's caching-on run, m12 and m13 read Z from m02, and each starts what it
needs when the other result is missing. `--skip-existing` skips ids whose result says ok.

## Downloads, disk, GPU time

- **Model:** Qwen2.5-7B-Instruct's bf16 safetensors, the 15.2 GB chapter 2 counts, downloaded
  once into the Hugging Face cache. The model is not gated.
- **Packages:** vLLM 0.30.0 with PyTorch 2.13.0, FlashInfer 0.6.18.post1 and their CUDA
  libraries (pins from vLLM's `requirements/cuda.txt`), several GB.
- **Disk:** keep at least 40 GB free for the model, the packages and torch.compile's cache.
  m01 records the free space.

GPU-minutes are planning estimates, not measurements: about 3 minutes per engine start,
prefill at half the L4's 121 TFLOPS bf16 peak (a 32k prefill about 12 s, a 131,072 one about
85 s, by section 6.5's arithmetic), decode near 50 ms a token. Replace them with the run
log's timestamps after the first run.

| Id | Measures | Engine starts | GPU-min |
|---|---|---|---|
| m01 | nvidia-smi total memory, environment, preflight | 0 | 0.5 |
| m14 | tokens per turn from baseline.json, against Qwen's tokenizer | 0 | 0 |
| m02 | bf16 at 131,072: X, Y, Z, the error's first line | 2 (both fail) | 6 |
| m03 | bf16 at 32,768: the carve-up, the pool, block size | 1 | 3 |
| m04 | concurrency, bf16 at 4,096 and 32,768 | 1 new | 3 |
| m05 | concurrency, fp8 at 4,096, 32,768, 131,072 | 3 | 9 |
| m06 | attention backend lines | 0 new | 0 |
| m07 | mini-session turn pair, caching off then on | 2 | 7 |
| m08 | turn two's prefix-cache hit tokens | 0 new | 0 |
| m09 | turn pair over the 32k repository prefix | 2 | 14 |
| m10 | sessions over one shared 120k prefix, fp8 | 1 | 15 |
| m11 | agent A's turn two with agent B, offloading off then on | 2 | 19 |
| m12 | TPOT, bf16 and fp8, at 32k and bf16's longest length | 4 | 27 |
| m13 | retrieval recovery, 20 needles per length | 4 | 40 |
| | **total** | | **143.5** |

## Inputs this folder had to define

The draft and the writing plan leave these open. Each is a constant in `run.py` or
`prompts.py`, and the draft should state the one it prints.

- **The 32k repository prefix (m09, m11).** The mini-session's own repo, then the installed
  vLLM 0.30.0 package's `vllm/v1/*.py` in path order, cut with Qwen's tokenizer so that turn
  two and its answer fill the 32,768-token context exactly: a prefix of 32,368 tokens with Qwen2.5's tokenizer. A
  prefix of 32,768 tokens cannot fit under the plain start the draft uses below 128k.
- **Turn two's question.** "now add a unit test for the retry in test_directory.py", sent
  after turn one's own 180 answer tokens, as token ids so the prefix matches to the token.
- **The second agent's repository (m11).** `vllm/model_executor/*.py`, the same length.
- **The pool that makes agent B evict agent A (m11).** At 32k in bf16 the default pool holds
  both agents, so nothing would be evicted and the CPU tier would never be read.
  `--kv-cache-memory-bytes 2818572288` (1.5 x 32,768 x 57,344 bytes) holds one agent and
  not two, the situation section 6.5 draws at 128k in fp8. The CPU tier is 8 GiB.
- **The 120k shared prefix and 8k tails (m10).** 120,000 tokens of `vllm/v1/`, then 8,000
  distinct tokens of `vllm/entrypoints/` per session; each session decodes 3,000 tokens so
  it stays resident. Up to 16 sessions; the count is the scheduler's running gauge when the
  waiting gauge first goes above zero.
- **The retrieval check (m13).** 20 trials per length, needle depths evenly spaced, deepest
  first. Each needle is `# NOTE: the rollback code for region <key> is <7 digits>.`, planted
  in `vllm/model_executor/` source inside Qwen's chat template, then asked for back;
  recovery is the share of answers that contain the digits. Every prompt is the length minus
  32 tokens, so prompt and answer fit the start. With 20 trials, 95 percent means 19 found.
- **The skip-layers rerun.** `--skip-layers 27` reruns fp8 with `--kv-cache-dtype-skip-layers`;
  the draft says "the outlier layers" without naming them, and layer 27 is the one its
  reproduction names.

## The 128k command

The draft prints `--hf-overrides '{"rope_scaling": {...}}'`. In vLLM 0.30.0 that form is
expected to stop before the KV check: vLLM's docs use `rope_parameters` (with `rope_theta`),
transformers 5 maps `rope_scaling` into `rope_parameters` with `rope_theta` empty, and a YaRN
rope type no longer multiplies `max_position_embeddings`, so 131,072 exceeds Qwen's 32,768
(`vllm/config/model.py`, `_get_and_verify_max_len`). m02 runs the printed command verbatim
and then the 0.30.0 form, `rope_parameters` plus `"max_position_embeddings": 131072`, and
records both first error lines. Every other start above 32,768 uses the 0.30.0 form;
`--yarn-form printed` switches them back.

## Colab risks and fallbacks

- **Driver against vLLM's CUDA build.** vLLM 0.30.0's default wheels are CUDA 12.9 and CUDA 13
  needs an R580 driver. If m01 warns that torch cannot see the GPU, reinstall with
  `--torch-backend=cu129` (or the variant the driver supports) and rerun m01.
- **bf16 at 131,072 refuses; fp8 at 131,072 may too.** Both are recorded as results. If fp8
  at 131,072 does not start, m05, m10 and m13's 131,072 rows record the first error line, and
  that is the chapter's finding.
- **Host memory for the CPU tier (m11).** 8 GiB of pinned host memory. m11 reads MemAvailable
  and lowers the tier to a third of it when short (one 32k agent needs about 1.9 GB).
- **Runtime limits.** The longest single id is m13, about 40 minutes. Copy `results/` after
  each id and rerun with `--skip-existing` after a disconnect. `--trials 10` halves m13.
- **bf16 at its longest length (m12).** The start can fail at exactly Z; m12 steps down 1,024
  tokens up to three times and records the length it used.
- **m10 never stalls.** If all 16 sessions run, the result says so; raise `MAX_SESSIONS`.

## Tests

```bash
python3 test_run.py
```

Runs on a Mac with no GPU. Parser inputs are vLLM 0.30.0's format strings filled with
synthetic values. The fixture check re-reads the chapter draft (path in `CH06_DRAFT`, which
defaults to the author's vault) and fails if any placeholder has no row in `fixtures.md`;
it skips when the draft is not on the machine.
