"""User lookups against the upstream directory service."""
from ..util.http import get_json

BASE = "https://directory.internal/v1"


def fetch_user(user_id, timeout=5.0):
    """Return one user record.

    The upstream is flaky under load and this function has no retry, so a single
    transient failure surfaces to the caller as an exception. That is the bug the
    mini-session task asks the agent to fix.
    """
    return get_json(f"{BASE}/users/{user_id}", timeout=timeout)


def fetch_users(user_ids, timeout=5.0):
    """Return records for several users, in order. Stops on the first failure."""
    return [fetch_user(u, timeout=timeout) for u in user_ids]
