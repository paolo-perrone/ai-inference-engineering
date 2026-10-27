"""Every long prompt chapter 6 measures with, built the same way on every machine.

The mini-session's repo is about 1,500 tokens, so the long variants need more text. The
companion repo holds about 64 KB, far short of 131,072 tokens, so the extra text comes from
one real repository that every run already has on disk at a pinned version: the Python
source of the installed vLLM 0.30.0 package. Files go in sorted path order and are cut at
an exact token count with the model's own tokenizer, so a given vLLM version and model
always give the same token ids. The corpus fingerprint in each result names what was used.

Corpora (the chapter's draft does not define them; these are this fixture's choices):
  REPO_A   the 32k variant of the mini-session: its repo files first, then vllm/v1/
  REPO_B   the second agent's different repository: vllm/model_executor/
  SHARED   the 120,000-token shared repository of the shared-prefix run: vllm/v1/
  TAILS    each session's unique 8,000-token tail: vllm/entrypoints/, a disjoint slice each
  HAYSTACK the retrieval check's long context: vllm/model_executor/
"""
import hashlib
import importlib.util
import pathlib
import random
import sys

HERE = pathlib.Path(__file__).resolve().parent
MINI = HERE.parents[2] / "examples" / "mini-session"
sys.path.insert(0, str(MINI))
import measure  # noqa: E402  the mini-session's own prompt, reused verbatim

TURN_TWO_TASK = "now add a unit test for the retry in test_directory.py"
NEEDLE_ANSWER_TOKENS = 32


def vllm_source_root(override=None):
    """The installed vllm package directory, found without importing vllm (and torch)."""
    if override:
        return pathlib.Path(override)
    spec = importlib.util.find_spec("vllm")
    if spec and spec.submodule_search_locations:
        return pathlib.Path(list(spec.submodule_search_locations)[0])
    return None


def corpus_files(root, subdir):
    base = root / subdir if subdir else root
    return sorted(p for p in base.rglob("*.py") if p.is_file() and p.stat().st_size)


def corpus_text(root, subdir, max_chars=None):
    """Files as `--- path ---` sections, the same framing measure.build_prompt uses.

    Stops after `max_chars`, so a 32k prompt does not tokenize 13 MB of source.
    """
    parts, size = [], 0
    for p in corpus_files(root, subdir):
        part = (f"\n--- {p.relative_to(root.parent).as_posix()} ---\n"
                f"{p.read_text(encoding='utf-8', errors='replace')}")
        parts.append(part)
        size += len(part)
        if max_chars and size >= max_chars:
            break
    return "".join(parts)


_IDS = {}


def corpus_ids(tok, root, subdir, n):
    """At least `n` token ids of a corpus (all of it if shorter), cached per process.

    Source code runs about 3 to 4 characters per token, so 8 per token over-reads safely.
    """
    key = (str(root), subdir, tok.model, tok.exact)
    if key not in _IDS or len(_IDS[key]) < n:
        _IDS[key] = tok.encode(corpus_text(root, subdir, max_chars=8 * n + 4096))
    return _IDS[key]


def fingerprint(root, subdirs):
    h = hashlib.sha256()
    for sub in subdirs:
        for p in corpus_files(root, sub):
            h.update(p.relative_to(root).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


class Tokens:
    """The model's tokenizer when transformers is installed, else a stated estimate.

    The estimate is measure.py's characters over four. It exists so --dry-run can size
    prompts on a laptop; a real run refuses to measure with it.
    """

    def __init__(self, model, allow_estimate=False):
        self.model, self.exact, self.tok = model, False, None
        if model is None:
            return
        try:
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(model)
            self.exact = True
        except Exception as e:  # noqa: BLE001  no transformers, or no network
            if not allow_estimate:
                raise SystemExit(f"cannot load the tokenizer for {model}: {e}")
            self.tok = None

    def encode(self, text):
        if self.tok is not None:
            return self.tok.encode(text, add_special_tokens=False)
        return [0] * measure.count_tokens(text)          # stand-in ids of the right length

    def chat_wrap(self):
        """The ids the chat template puts before and after one user message."""
        if self.tok is None:
            return [0] * 20, [0] * 5
        mark = "\x00MESSAGE\x00"
        s = self.tok.apply_chat_template([{"role": "user", "content": mark}],
                                         tokenize=False, add_generation_prompt=True)
        before, after = s.split(mark)
        return self.encode(before), self.encode(after)


def take(ids, n, what):
    if len(ids) < n:
        raise SystemExit(f"{what}: the corpus has {len(ids)} tokens, {n} are needed")
    return ids[:n]


def mini_prompt_ids(tok):
    """The mini-session's turn one, exactly as measure.py builds it."""
    return tok.encode(measure.build_prompt())


def task_ids(tok):
    return tok.encode("\n\nTask: " + (MINI / "task.txt").read_text(encoding="utf-8").strip())


def repo_prefix_tokens(tok, context=32_768, margin=16):
    """The longest repository prefix whose two turns fit one 32,768-token context.

    The book's 32k is 32,768 tokens, and under the plain (no YaRN) start that is the whole
    context, so the repository itself cannot be 32,768 tokens: turn two must also hold turn
    one's task and answer, its own task, and room for its answer. The prefix is everything
    left, which keeps the run on the plain start the draft prescribes below 128k.
    """
    t2 = tok.encode("\n\nTask: " + TURN_TWO_TASK)
    return context - 2 * measure.COMPLETION_TOKENS - len(task_ids(tok)) - len(t2) - margin


def repo_turn_one(tok, root, prefix_tokens):
    """The 32k variant of the mini-session: header, mini repo, vllm/v1, cut, then the task.

    `prefix_tokens` counts everything before the task line, so the repository prefix
    that turn two can reuse is exactly that long.
    """
    head = tok.encode(measure.build_prompt().split("\n\nTask: ")[0])
    extra = corpus_ids(tok, root, "v1", prefix_tokens)
    prefix = head + take(extra, prefix_tokens - len(head), "REPO_A")
    return prefix + task_ids(tok)


def other_repo_turn_one(tok, root, prefix_tokens):
    """The second agent: a different repository of the same size, same framing."""
    head = tok.encode("You are a coding agent working in this repository.\n")
    body = corpus_ids(tok, root, "model_executor", prefix_tokens)
    task = tok.encode("\n\nTask: add a docstring to every public function in this module")
    return head + take(body, prefix_tokens - len(head), "REPO_B") + task


def turn_two(turn_one_ids, answer_ids, tok):
    """Turn two re-sends turn one and its answer, then the new task, as token ids.

    Sending ids keeps the shared part identical to the token, which is what the hash chain
    needs: re-tokenizing the joined text could merge tokens at the seam.
    """
    return list(turn_one_ids) + list(answer_ids) + tok.encode("\n\nTask: " + TURN_TWO_TASK)


def shared_prefix_sessions(tok, root, shared_tokens, tail_tokens, sessions):
    """`sessions` prompts that share one repository prefix and differ in their tails."""
    shared = take(corpus_ids(tok, root, "v1", shared_tokens), shared_tokens, "SHARED")
    tails_src = corpus_ids(tok, root, "entrypoints", tail_tokens * sessions)
    take(tails_src, tail_tokens * sessions, "TAILS")
    return [shared + tails_src[i * tail_tokens:(i + 1) * tail_tokens] for i in range(sessions)]


def needle_trials(tok, root, length, trials, seed=6):
    """The retrieval check's prompts at one context length, deepest needle first.

    Each trial plants one fact, a key and a 7-digit value drawn from a seeded generator,
    as a comment line at a depth evenly spaced through the context, then asks for it back.
    Every prompt is exactly `length - NEEDLE_ANSWER_TOKENS` tokens, so prompt plus answer
    fits a start whose max_model_len is `length`. Deepest-first order lets prefix caching
    reuse the haystack before each needle; cached keys and values are the ones a fresh
    prefill writes, so the order changes the time and not the answers.
    """
    rng = random.Random(seed * 1_000_003 + length)
    before, after = tok.chat_wrap()
    hay = corpus_ids(tok, root, "model_executor", length)
    total = length - NEEDLE_ANSWER_TOKENS
    out = []
    for k in range(trials):
        depth = 1 - (k + 0.5) / trials
        key = f"{rng.choice(['amber', 'basalt', 'cobalt', 'delta', 'ember', 'fjord'])}-{rng.randint(10, 99)}"
        value = str(rng.randint(1_000_000, 9_999_999))
        needle = tok.encode(f"\n# NOTE: the rollback code for region {key} is {value}.\n")
        question = tok.encode(f"\n\nWhat is the rollback code for region {key}? "
                              f"Answer with the number only.")
        room = total - len(before) - len(after) - len(needle) - len(question)
        body = take(hay, room, "HAYSTACK")
        at = int(room * depth)
        ids = before + body[:at] + needle + body[at:] + question + after
        out.append({"depth": round(depth, 4), "key": key, "value": value, "prompt": ids,
                    "prompt_tokens": len(ids)})
    return out


def recovered(answer_text, value):
    return value in answer_text


def filler(tok, root, n):
    """`n` tokens of plain repository text, for timing runs that need only a length."""
    return take(corpus_ids(tok, root, "model_executor", n), n, "FILLER")
