"""A small TTL cache. The agent has to notice this exists before adding retry."""
import time


class TTLCache:
    """Keys expire after `ttl` seconds. Not thread-safe, deliberately.

    The directory service is read-heavy and its records change rarely, so most user
    lookups never reach the network at all. A retry added without checking here would
    retry lookups that the cache was about to answer.
    """

    def __init__(self, ttl=60.0, max_entries=1024):
        self.ttl = ttl
        self.max_entries = max_entries
        self._d = {}

    def get(self, key):
        hit = self._d.get(key)
        if hit is None:
            return None
        value, stamped = hit
        if time.monotonic() - stamped > self.ttl:
            del self._d[key]
            return None
        return value

    def put(self, key, value):
        if len(self._d) >= self.max_entries:
            oldest = min(self._d, key=lambda k: self._d[k][1])
            del self._d[oldest]
        self._d[key] = (value, time.monotonic())

    def clear(self):
        self._d.clear()

    def __len__(self):
        return len(self._d)
