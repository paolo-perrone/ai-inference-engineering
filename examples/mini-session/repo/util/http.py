"""Thin HTTP helpers. Deliberately has no retry: that is the task."""
import json
import urllib.error
import urllib.request


class UpstreamError(RuntimeError):
    """Raised when the upstream service answers with a non-2xx status."""


def get_json(url, timeout=5.0):
    """GET `url` and parse the body as JSON.

    Raises UpstreamError on a non-2xx status and lets urllib's own exceptions
    through on a transport failure, which is what makes the caller flaky.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            if not 200 <= r.status < 300:
                raise UpstreamError(f"{url} answered {r.status}")
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise UpstreamError(f"{url} answered {e.code}") from e
