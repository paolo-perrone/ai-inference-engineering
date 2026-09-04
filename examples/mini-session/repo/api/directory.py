"""The service object callers actually hold."""
from ..util.cache import TTLCache
from ..util.config import CONFIG
from . import users


class Directory:
    """Cached reads against the upstream directory.

    Every miss goes through users.fetch_user, which is where the retry belongs. A retry
    added at this layer would sit on the wrong side of the cache and retry hits.
    """

    def __init__(self, cache=None):
        self.cache = cache if cache is not None else TTLCache(ttl=CONFIG["cache_ttl_seconds"])

    def user(self, user_id):
        hit = self.cache.get(user_id)
        if hit is not None:
            return hit
        record = users.fetch_user(user_id, timeout=CONFIG["timeout_seconds"])
        self.cache.put(user_id, record)
        return record

    def batch(self, user_ids):
        if len(user_ids) > CONFIG["max_batch"]:
            raise ValueError(f"batch of {len(user_ids)} exceeds max_batch {CONFIG['max_batch']}")
        return [self.user(u) for u in user_ids]

    def invalidate(self, user_id=None):
        if user_id is None:
            self.cache.clear()
        else:
            self.cache.put(user_id, None)
