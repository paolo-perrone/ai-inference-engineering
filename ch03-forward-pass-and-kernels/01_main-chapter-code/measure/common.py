"""Shared pieces for chapter 3's measurements: pins, book constants, the run log, results.

Every measurement writes results/<id>.json. Each file carries the measured fields, the
medians and spreads, the exact commands, and a run log naming the GPU, driver, CUDA, vLLM,
PyTorch and transformers versions the numbers came from.
"""
import datetime
import importlib.metadata
import json
import os
import pathlib
import platform
import shlex
import statistics
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
MINI = REPO_ROOT / "examples" / "mini-session"
RESULTS = pathlib.Path(os.environ.get("CH3_RESULTS", HERE / "results"))
RAW = RESULTS / "raw"
TRACES = RESULTS / "traces"

sys.path.insert(0, str(MINI))
import measure as mini  # noqa: E402  chapter 2's harness: prompt, engine start/stop, env

MODEL = "Qwen/Qwen2.5-7B-Instruct"
DTYPE = "bfloat16"
PINS = {"vllm": "0.30.0", "torch": "2.13.0", "transformers": "5.17.0"}
RUNS = 5                      # timed runs per setting; the book prints the median of 5
MAX_CONTEXT = 32768           # Qwen2.5-7B's max_position_embeddings, config.json
WARMUP = 1                    # discarded runs after each engine start
COMPLETION_TOKENS = mini.COMPLETION_TOKENS   # 180, the mini-session's answer length

# Book constants, each with where the book takes it from. Nothing here is a measurement;
# the floors below are the arithmetic the chapter compares the measurements against.
BOOK = {
    "decode_bytes_gb": 14.1,           # ch2: Qwen2.5-7B bf16 weights a decode step streams
    "l4_hbm_gb_per_s": 300.0,          # NVIDIA L4 product page
    "l4_dense_bf16_tflops": 121.0,     # NVIDIA L4 product page
    "layer_weights": 6.5e9,            # ch2: weights in the 7 matrices x 28 layers
    "prefill_ops_draft": 19e12,        # ch3 draft and ch2 text: 2 x 6.5e9 x 1,457
    "prefill_ops_plan": 22.1e12,       # ch3 writing-plan.md, measurement 4
    "fixture_prompt_estimate": 1457,   # baseline.json context_tokens, characters over 4
    "paper_graphs_ratio_l4": 1.028,    # REPORTED, arXiv 2605.30571, transformers on L4
    "head_ms_derived": 3.6,            # ch3 draft: 1.09 GB / 300 GB/s
}
BOOK["decode_floor_ms"] = BOOK["decode_bytes_gb"] / BOOK["l4_hbm_gb_per_s"] * 1000


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def med_spread(values):
    """Median and spread (max minus min) of a list, or None for both when it is empty."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    return statistics.median(vals), max(vals) - min(vals)


def _out(cmd, timeout=60):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def pkg_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def run_log(command=None, extra=None):
    """What produced the numbers: card, driver, CUDA, package versions, and the command."""
    rows = _out(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    gpus = [dict(zip(("name", "driver"), (s.strip() for s in r.split(","))))
            for r in rows.splitlines() if r.strip()]
    header = _out(["nvidia-smi"])
    torch_cuda = _out([sys.executable, "-c", "import torch; print(torch.version.cuda)"]).splitlines()
    nsys = _out(["nsys", "--version"])
    log = {
        "utc": now(),
        "gpus": gpus,
        "gpu": gpus[0]["name"] if gpus else None,
        "driver": gpus[0]["driver"] if gpus else None,
        "cuda_supported_by_driver": (header.split("CUDA Version:")[1].split()[0]
                                     if "CUDA Version:" in header else None),
        "cuda_runtime_torch": torch_cuda[-1] if torch_cuda else None,
        "python": platform.python_version(),
        "versions": {p: pkg_version(p) for p in
                     ("vllm", "torch", "transformers", "kernels", "flash-attn",
                      "flashinfer-python", "huggingface-hub")},
        "pins": PINS,
        "nsys": nsys or None,
        "model": MODEL,
        "dtype": DTYPE,
        "argv": [sys.executable] + sys.argv,
    }
    if command is not None:
        log["command"] = command
    if extra:
        log.update(extra)
    off_pin = {p: v for p, v in PINS.items() if log["versions"].get(p) not in (None, v)}
    if off_pin:
        log["WARNING_off_pin"] = {p: {"pinned": v, "installed": log["versions"][p]}
                                  for p, v in off_pin.items()}
    return log


def shell(cmd):
    return " ".join(shlex.quote(str(c)) for c in cmd)


def write_result(mid, payload):
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{mid}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_json(path):
    path = pathlib.Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def mini_prompt():
    """The mini-session's first-turn prompt, assembled by chapter 2's harness."""
    return mini.build_prompt()


def padded_ids(tokenizer, length):
    """Token IDs of exactly `length`: the mini-session prompt, cut or extended with more of
    the same repo text. Extending with the fixture's own text keeps the content realistic."""
    base = tokenizer(mini_prompt(), add_special_tokens=False)["input_ids"]
    if length <= len(base):
        return base[:length]
    filler = tokenizer("\n" + "".join(p.read_text(encoding="utf-8") for p in mini.repo_files()),
                       add_special_tokens=False)["input_ids"]
    ids = list(base)
    while len(ids) < length:
        ids.extend(filler)
    return ids[:length]
