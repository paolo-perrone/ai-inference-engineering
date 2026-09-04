"""Configuration, read once at import. Env wins over the defaults."""
import os

DEFAULTS = {
    "directory_base": "https://directory.internal/v1",
    "timeout_seconds": 5.0,
    "cache_ttl_seconds": 60.0,
    "max_batch": 50,
}


def _coerce(default, raw):
    if isinstance(default, float):
        return float(raw)
    if isinstance(default, int):
        return int(raw)
    return raw


def load():
    """DEFAULTS overlaid with any DIRECTORY_* env var, coerced to the default's type."""
    out = dict(DEFAULTS)
    for key, default in DEFAULTS.items():
        raw = os.environ.get("DIRECTORY_" + key.upper())
        if raw is None:
            continue
        try:
            out[key] = _coerce(default, raw)
        except (TypeError, ValueError):
            raise ValueError(f"DIRECTORY_{key.upper()}={raw!r} is not a {type(default).__name__}")
    return out


CONFIG = load()
