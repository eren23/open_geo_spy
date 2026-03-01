"""Tests for cache store and decorators."""

import asyncio
import time

import pytest

from src.cache.store import CacheStore, make_key


class TestMakeKey:
    def test_deterministic(self):
        k1 = make_key("serper", {"q": "paris", "num": 10})
        k2 = make_key("serper", {"q": "paris", "num": 10})
        assert k1 == k2

    def test_different_params(self):
        k1 = make_key("serper", {"q": "paris"})
        k2 = make_key("serper", {"q": "london"})
        assert k1 != k2

    def test_different_source(self):
        k1 = make_key("serper", {"q": "paris"})
        k2 = make_key("osm", {"q": "paris"})
        assert k1 != k2

    def test_key_length(self):
        k = make_key("test", {"a": 1})
        assert len(k) == 24


class TestCacheStore:
    @pytest.fixture
    def store(self):
        return CacheStore(max_memory_entries=10, default_ttl=60)

    @pytest.mark.asyncio
    async def test_get_miss(self, store):
        assert await store.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_set_and_get(self, store):
        await store.set("key1", {"data": "value"})
        result = await store.get("key1")
        assert result == {"data": "value"}

    @pytest.mark.asyncio
    async def test_ttl_expiry(self, store):
        await store.set("ephemeral", "gone", ttl=1)
        assert await store.get("ephemeral") == "gone"
        await asyncio.sleep(1.1)
        assert await store.get("ephemeral") is None

    @pytest.mark.asyncio
    async def test_eviction(self):
        store = CacheStore(max_memory_entries=3, default_ttl=60)
        for i in range(5):
            await store.set(f"k{i}", i)
        # Only last 3 should remain
        assert store.memory_size == 3
        assert await store.get("k0") is None
        assert await store.get("k4") == 4

    @pytest.mark.asyncio
    async def test_invalidate(self, store):
        await store.set("key", "value")
        store.invalidate("key")
        assert await store.get("key") is None

    @pytest.mark.asyncio
    async def test_clear(self, store):
        await store.set("a", 1)
        await store.set("b", 2)
        store.clear()
        assert store.memory_size == 0

    @pytest.mark.asyncio
    async def test_disk_backend(self, tmp_path):
        store = CacheStore(disk_path=str(tmp_path), default_ttl=60)
        await store.set("disk_key", {"hello": "world"})
        # Clear memory, should still find on disk
        store._memory.clear()
        result = await store.get("disk_key")
        assert result == {"hello": "world"}
