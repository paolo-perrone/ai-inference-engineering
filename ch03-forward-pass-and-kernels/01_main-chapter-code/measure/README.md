# Chapter 3 measurements

Every measured number in chapter 3, run on one NVIDIA L4. One entry point, one subcommand per
measurement, one result file each:

```
python run.py --list          # the ids, one per line, in the order to run them on the GPU
python run.py m01             # writes results/m01.json
python run.py all             # all of them; one failure is recorded and the rest still run
python run.py all --dry-run   # no GPU: prints every command and checks the config
python test_run.py            # parser and contract tests, no GPU
```

Each `results/<id>.json` holds the measured values under `fields`, medians of 5 runs with their
spread (max minus min) where the chapter asks for one, every command that ran, and a run log:
GPU name, driver, the CUDA version the driver supports and the one PyTorch was built with, and
the vLLM, PyTorch and transformers versions. A failed or incomplete measurement still writes its
file, with `error` or `missing_fields` saying what is missing. `fixtures.md` maps every
`[FIXTURE: ...]` placeholder in the draft to the subcommand and field that fills it.

Pins: vLLM 0.30.0, PyTorch 2.13.0 (the version vLLM 0.30.0 requires), transformers 5.17.0,
`Qwen/Qwen2.5-7B-Instruct` in bf16, greedy. Every vLLM engine starts with prefix caching off and
`--generation-config vllm`, and every timed request sets `temperature: 0` and `ignore_eos`, so
each run decodes the same 180 tokens. Flags, log lines and request fields were read from vLLM
0.30.0's own source; the module docstrings name the file for each.

## The measurements

Estimates are L4 minutes for a first run on a fresh runtime, assuming the model is already
downloaded. They are estimates, not measurements: engine start (load, `torch.compile`, CUDA
graph capture) dominates, and the first start pays the compile cache.

| Id | What it measures | Needs | Est. min |
|---|---|---|---|
| m16 | The prompt as token IDs: first IDs, count, against baseline.json's 1,457 and the engine's count | tokenizer only | 1 |
| m01 | Prefill (TTFT) and decode (E2E minus TTFT), median of 5 and spread, engine prompt count, driver, vLLM version | vLLM | 6 |
| m02 | TTFT, TPOT, mean ITL over every chunk gap, tokens per streamed chunk | m01's engine run | 0 |
| m03 | Decode step against the 47 ms floor, both directions | m01's engine run | 0 |
| m04 | Prefill against the 0.16 s (19T) and 0.18 s (22.1T) floors, and the count rescaled to the engine's prompt | m01's engine run | 0 |
| m05 | TTFT and decode step at the real prompt, 4,096 and 8,192 tokens (exact token-ID prompts) | vLLM | 8 |
| m06 | Decode step with `cudagraph_mode NONE`, with `--enforce-eager`, and default; both ratios | vLLM | 9 |
| m10 | The FLASH_ATTN backend line, the generation-config override line, Qwen's generation_config.json | vLLM | 4 |
| m13 | transformers eager against `torch.compile` (flash_attention_2 both); vLLM default, `-cc.custom_ops=["all"]`, `-O0` | transformers + vLLM | 16 |
| m15 | `torch.cuda.mem_get_info` after loading the bf16 model | transformers | 2 |
| m14 | The attention ablation: eager, sdpa, flash_attention_2 at the real prompt, 4,096, 8,192: time, peak memory, out-of-memory | transformers | 10 |
| m12 | transformers eager kernels per layer for one decode step, the add and RMSNorm kernels by name | transformers | 3 |
| m07 | Kernels in one vLLM decode step: per step, per layer, the layer's sequence, RoPE, split-K, copies, cuBLAS names | vLLM + Nsight Systems | 8 |
| m11 | Attention kernel names in prefill and in decode | m07's trace | 0 |
| m09 | Output head time, the kernels after it under bench sampling and under Qwen's defaults | m07's trace + one more capture | 8 |
| m08 | Kernel names under `--enforce-eager`, and whether `cudagraph_mode NONE` keeps the `triton_` kernels | vLLM + Nsight Systems | 14 |
| | **Total** | | **89**, plus about 15 for installs and the model download |

Shared work is cached, never assumed: m02 to m04 read the engine run m01 saved in
`results/raw/serve-default.json`, and m09 and m11 read m07's trace in `results/traces/`. Run any
of them alone and it does that work itself. `--fresh` ignores the cache.

## Running it on a fresh Colab L4 runtime

Runtime, then Change runtime type, then L4 GPU. Then, one cell at a time:

```python
!nvidia-smi --query-gpu=name,driver_version --format=csv
!df -h /content | tail -1          # needs about 40 GB free, see Disk below
!git clone https://github.com/paolo-perrone/ai-inference-engineering.git
%cd ai-inference-engineering/ch03-forward-pass-and-kernels/01_main-chapter-code/measure
```

```python
# vLLM 0.30.0 pulls in torch 2.13.0; accelerate is what transformers' device_map needs;
# kernels supplies flash_attention_2 to transformers when the flash_attn package is absent.
!pip install -q vllm==0.30.0 transformers==5.17.0 accelerate kernels
!python -c "import vllm, torch, transformers; print(vllm.__version__, torch.__version__, transformers.__version__)"
```

```python
# The model, about 15.2 GB of bf16 safetensors, into the Hugging Face cache. Not gated.
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2.5-7B-Instruct")
```

Nsight Systems, for m07, m08, m09 and m11 only:

```python
!wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(. /etc/os-release; echo $VERSION_ID | tr -d .)/x86_64/cuda-keyring_1.1-1_all.deb
!dpkg -i cuda-keyring_1.1-1_all.deb && apt-get update -q && apt-get install -y -q nsight-systems-2026.3.2
import os
os.environ["PATH"] = "/opt/nvidia/nsight-systems/2026.3.2/bin:" + os.environ["PATH"]
!nsys --version && nsys status --environment
```

Then run everything, saving each result to Drive as it lands, so a disconnect costs one
measurement at most:

```python
from google.colab import drive
drive.mount("/content/drive")
import subprocess, shutil
ids = subprocess.run(["python", "run.py", "--list"], capture_output=True, text=True, check=True).stdout.split()
for mid in ids:
    subprocess.run(["python", "run.py", mid])           # a failure is recorded in results/<id>.json
    shutil.copytree("results", "/content/drive/MyDrive/ch03-results", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("*.nsys-rep", "*.sqlite"))
```

The `--list` order runs the cheap, robust measurements first and the Nsight captures last. The
traces themselves (`results/traces/*.nsys-rep`, `*.sqlite`) are left out of the Drive copy above
because they are large; copy them too if you want to open a trace in the Nsight GUI.

## Disk

About 15.2 GB for the model, 10 to 12 GB for the Python packages (PyTorch, vLLM, FlashInfer),
about 1.5 GB for Nsight Systems, and up to a few GB of traces: the one-generate captures are
small, a whole-run fallback capture can pass 1 GB. Plan on 40 GB free.

## What can go wrong on Colab, and the fallback for each

- **Nsight cannot read the GPU counters in the container** (CUPTI permission, or a driver older
  than this nsys supports). `nsys status --environment` says so before you spend time. The
  capture writes a report with no kernels, and m07, m08, m09, m11 record the error. Fallback:
  rent the same L4 as a plain VM (GCP g2-standard-8) and run the same four subcommands there.
- **The one-generate capture finds nothing.** It relies on vLLM's CUDA profiler hook firing
  `cudaProfilerStart` in the engine's worker process. If it captures no kernels, `bench_trace`
  falls back on its own to the draft's exact command, which records the whole run, warm-ups
  included; the decode step is then found as the repeating unit at the end of the trace.
- **8,192 tokens in transformers eager runs out of memory.** That is the expected result, not
  a crash: m14 catches it per row and records `oom: true` and the first length that failed.
- **flash_attention_2 in transformers.** Without the `flash_attn` package, transformers 5.17
  fetches the `kernels-community/flash-attn2` kernel through `kernels`. If that has no build for
  torch 2.13 on the L4, the flash rows record the error and m13's transformers arms do too.
  Compiling `flash_attn` from source takes well over an hour on Colab; skip it unless the book
  needs those rows. Each row records `config_attn_implementation`, so the printout shows which
  kernel actually ran.
- **`torch.compile` on transformers fails** (m13). The error is recorded in
  `hf_compile_error`; the eager arm and the vLLM arms still run.
- **The runtime disconnects.** Idle or long sessions can drop. The loop above copies results to
  Drive after every measurement, and a re-run reuses whatever engine runs and traces survived in
  `results/`.
- **A different GPU.** Colab can hand out another card. Every result's run log names the GPU;
  check it is `NVIDIA L4` before using a number.

## Files

- `run.py`: the subcommands, the run order and each measurement's declared fields
- `vllm_session.py`: engine configurations and timed sessions, built on chapter 2's
  `examples/mini-session/measure.py` (prompt, engine stop, environment)
- `stream_client.py`: the streaming client that timestamps every chunk
- `hf_worker.py`: the transformers side, one configuration per process
- `nsys_tools.py`: Nsight Systems capture commands and trace analysis
- `logparse.py`: vLLM startup log lines
- `common.py`: pins, the book's constants, run logs, result files
- `fixtures.md`: every placeholder in the draft, mapped to its field
- `test_run.py`: the tests
