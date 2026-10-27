"""Parsers for the vLLM 0.30.0 startup log lines chapter 3 quotes.

Each pattern is copied from the format string in vLLM 0.30.0's source, named beside it.
The log prefix (vllm/logger.py _FORMAT: "%(levelname)s %(asctime)s [%(fileinfo)s:%(lineno)d]
%(message)s", plus any process tag) is not relied on: every pattern searches the message.
"""
import ast
import re

# platforms/cuda.py:539
#   "Using %s attention backend out of potential backends: %s."
BACKEND_RE = re.compile(
    r"Using (?P<backend>\S+) attention backend out of potential backends: (?P<cands>\[.*?\])\.")

# config/model.py:1772
#   "Default vLLM sampling parameters have been overridden by %s: `%s`. If this is not
#    intended, please relaunch vLLM instance with `--generation-config vllm`."
OVERRIDE_RE = re.compile(
    r"Default vLLM sampling parameters have been overridden by (?P<src>.+?): `(?P<params>\{.*?\})`\.")

# config/vllm.py:1548
#   "Enforce eager set, disabling torch.compile and CUDAGraphs. This is equivalent to
#    setting -cc.mode=none -cc.cudagraph_mode=none"
ENFORCE_EAGER_RE = re.compile(r"Enforce eager set, disabling torch\.compile and CUDAGraphs\.")


def _last_line_with(text, needle):
    for line in reversed(text.splitlines()):
        if needle in line:
            return line.strip()
    return None


def attention_backend(text):
    m = BACKEND_RE.search(text)
    if not m:
        return None
    return {"backend": m.group("backend"),
            "candidates": ast.literal_eval(m.group("cands")),
            "line": _last_line_with(text, "attention backend out of potential backends")}


def sampling_override(text):
    m = OVERRIDE_RE.search(text)
    if not m:
        return None
    return {"source": m.group("src"),
            "params": ast.literal_eval(m.group("params")),
            "line": _last_line_with(text, "Default vLLM sampling parameters have been overridden")}


def enforce_eager(text):
    return bool(ENFORCE_EAGER_RE.search(text))
