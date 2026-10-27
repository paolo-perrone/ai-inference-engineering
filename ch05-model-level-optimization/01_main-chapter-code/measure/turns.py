"""The agent's recorded turns: their format, how a replay is scored, and the needle probe.

A recording is a JSONL file, one turn per line, written by record_turns.py while the agent
runs against the bf16 server:

  {"turn_id": "s1-0003", "session_id": "s1", "index": 3, "utc": "...",
   "repo": "/path/to/repo", "commit": "<sha or null>",
   "request": {<the agent's /v1/chat/completions body: messages, tools, tool_choice, ...>},
   "response": {"text": "...", "tool_calls": [{"name": ..., "arguments": "<json>"}],
                "finish_reason": "...", "usage": {...}},
   "check": null | "<shell command, see below>"}

`check` is optional. When present, the replay writes the replayed turn's tool calls to a
JSON file, exports its path as TOOL_CALLS_JSON, checks out `commit` in a scratch copy of
`repo`, and runs the command there; exit status 0 means the turn's outcome passed. That is
how "did the test still pass" becomes a measured outcome. Turns without a check are scored
on their action: the tool called and its arguments, or the final text.
"""
import json
import math
import random
import re

TOOL_TAG = "<tool_call>"   # the Hermes markup Qwen2.5 writes (tool_parsers/hermes_tool_parser.py)


def load_turns(path):
    turns = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            for key in ("turn_id", "session_id", "request"):
                if key not in t:
                    raise ValueError(f"{path}:{n} has no {key}")
            if "messages" not in t["request"]:
                raise ValueError(f"{path}:{n} request has no messages")
            turns.append(t)
    return turns


def sessions(turns):
    out = {}
    for t in turns:
        out.setdefault(t["session_id"], []).append(t)
    for s in out.values():
        s.sort(key=lambda t: t.get("index", 0))
    return out


def canonical_args(arguments):
    """Arguments as canonical JSON when they parse, else the raw string (a parse failure)."""
    try:
        return json.dumps(json.loads(arguments), sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return "UNPARSED:" + (arguments or "")


def action(resp):
    """What the turn did: its tool calls, or its final text when it called nothing."""
    calls = resp.get("tool_calls") or []
    if calls:
        return ("tools", tuple((c.get("name"), canonical_args(c.get("arguments")))
                               for c in calls))
    return ("text", re.sub(r"\s+", " ", (resp.get("text") or "").strip()))


# ---------------------------------------------------------------- tool-call validity

def _type_ok(value, t):
    return {"object": isinstance(value, dict), "array": isinstance(value, list),
            "string": isinstance(value, str),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "number": isinstance(value, (int, float)) and not isinstance(value, bool),
            "boolean": isinstance(value, bool), "null": value is None}.get(t, True)


def schema_errors(value, schema, path="$"):
    """A small JSON Schema check: type, required, properties, additionalProperties, enum, items.

    It covers the keywords tool schemas use. A keyword it does not know is ignored, so it
    can only under-count invalid calls, never invent one.
    """
    errs = []
    if not isinstance(schema, dict):
        return errs
    t = schema.get("type")
    types = t if isinstance(t, list) else ([t] if t else [])
    if types and not any(_type_ok(value, x) for x in types):
        return [f"{path}: expected {t}"]
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: not in enum")
    if isinstance(value, dict):
        props = schema.get("properties") or {}
        for r in schema.get("required") or []:
            if r not in value:
                errs.append(f"{path}.{r}: required")
        for k, v in value.items():
            if k in props:
                errs += schema_errors(v, props[k], f"{path}.{k}")
            elif schema.get("additionalProperties") is False:
                errs.append(f"{path}.{k}: not allowed")
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for i, v in enumerate(value):
            errs += schema_errors(v, schema["items"], f"{path}[{i}]")
    return errs


def tool_schemas(request):
    out = {}
    for tool in request.get("tools") or []:
        fn = tool.get("function") or tool
        out[fn.get("name")] = fn.get("parameters") or {}
    return out


def validity(resp, request):
    """Count tool-call attempts and how many are valid against their schema.

    An attempt the parser could not turn into a call (Hermes markup left in the text with
    no parsed call) counts as one invalid attempt: that is the parse failure section 5.5
    is about.
    """
    schemas = tool_schemas(request)
    calls = resp.get("tool_calls") or []
    attempts, valid, failures = 0, 0, []
    for c in calls:
        attempts += 1
        try:
            args = json.loads(c.get("arguments") or "")
        except ValueError:
            failures.append(f"{c.get('name')}: arguments are not JSON")
            continue
        if c.get("name") not in schemas:
            failures.append(f"{c.get('name')}: unknown tool")
            continue
        errs = schema_errors(args, schemas[c["name"]])
        if errs:
            failures.append(f"{c['name']}: {errs[0]}")
        else:
            valid += 1
    if not calls and TOOL_TAG in (resp.get("text") or ""):
        attempts += 1
        failures.append("tool-call markup in text, nothing parsed")
    return {"attempts": attempts, "valid": valid, "failures": failures}


# ---------------------------------------------------------------- KL from prompt logprobs

def topk_at(entry):
    """{token_id: logprob} from one position of vLLM's prompt_logprobs.

    The completions API returns, per prompt position, None or a dict keyed by token id
    (as a string in JSON) whose values hold "logprob" (completion/protocol.py:645).
    """
    if not entry:
        return {}
    return {int(k): (v["logprob"] if isinstance(v, dict) else float(v)) for k, v in entry.items()}


def kl_position(ref, cand):
    """KL(ref || cand) at one position, over the reference's returned top-k tokens.

    vLLM returns only the top-k log probabilities (max_logprobs, config/model.py:254), so
    this is the truncated sum over the reference's top-k. A token missing from the
    candidate's list gets the candidate's smallest returned logprob, an upper bound on its
    probability. Both approximations are stated in the result, not hidden.
    """
    if not ref or not cand:
        return None
    floor = min(cand.values())
    total = 0.0
    for tok, lp in ref.items():
        p = math.exp(lp)
        total += p * (lp - cand.get(tok, floor))
    return total


def mean_kl(ref_positions, cand_positions):
    vals = [kl_position(topk_at(r), topk_at(c)) for r, c in zip(ref_positions, cand_positions)]
    vals = [v for v in vals if v is not None]
    return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)


# ---------------------------------------------------------------- the 128k needle probe

_WORDS = ("request cache token layer kernel batch step memory prefill decode server queue "
          "tensor weight scale block page session agent repository function module test "
          "patch commit branch retry timeout config handler parser schema field value").split()
_NAMES = ["ORCHID", "BASALT", "TUNDRA", "QUARTZ", "MAPLE", "CINDER", "HARBOR", "LANTERN"]


def haystack(n_lines, seed):
    rng = random.Random(seed)
    return [" ".join(rng.choice(_WORDS) for _ in range(12)) + "." for _ in range(n_lines)]


def needle_prompt(n_lines, seed, keys=4):
    """A long filler document with `keys` facts hidden at fixed depths, and the question.

    Returns the chat messages and the expected answers. Deterministic in (n_lines, seed).
    """
    rng = random.Random(10_000 + seed)
    names = rng.sample(_NAMES, keys)
    codes = [f"{rng.randrange(100000, 999999)}" for _ in names]
    lines = haystack(n_lines, seed)
    for i, (name, code) in enumerate(zip(names, codes)):
        depth = int(n_lines * (0.1 + 0.8 * i / max(1, keys - 1)))
        lines.insert(depth, f"The access code for project {name} is {code}.")
    question = ("List the access code for each of these projects: " + ", ".join(names)
                + ". Answer with one line per project, in the form NAME: CODE.")
    messages = [{"role": "user", "content": "\n".join(lines) + "\n\n" + question}]
    return messages, dict(zip(names, codes))


def needle_passed(answer, expected):
    """Pass when every code appears on the line that names its project."""
    lines = (answer or "").splitlines()
    for name, code in expected.items():
        if not any(name in l and code in l for l in lines):
            return False
    return True
