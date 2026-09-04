"""The tests the agent's patch has to keep passing."""
from api.directory import Directory
from util.cache import TTLCache


class FakeCache(TTLCache):
    def __init__(self):
        super().__init__(ttl=999.0)
        self.gets = 0

    def get(self, key):
        self.gets += 1
        return super().get(key)


def test_cache_serves_the_second_read():
    c = FakeCache()
    c.put("u1", {"id": "u1"})
    d = Directory(cache=c)
    assert d.user("u1") == {"id": "u1"}
    assert c.gets == 1


def test_batch_refuses_an_oversized_request():
    d = Directory(cache=FakeCache())
    try:
        d.batch([str(i) for i in range(999)])
    except ValueError as e:
        assert "max_batch" in str(e)
    else:
        raise AssertionError("an oversized batch should raise")


def test_ttl_cache_evicts_when_full():
    c = TTLCache(ttl=999.0, max_entries=2)
    c.put("a", 1); c.put("b", 2); c.put("c", 3)
    assert len(c) == 2
