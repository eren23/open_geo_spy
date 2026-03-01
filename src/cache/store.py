"""Cache store with memory and disk backends, SHA-256 content-addressed keys."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import OrderedDict
from typing import Any, Optional

import aiofiles
from loguru import logger


def make_key(source: str, params: dict) -> str:
    """Deterministic cache key from source and params."""
    raw = f"{source}:{json.dumps(params, sort_keys=True, default=str)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


class CacheStore:
    """Two-tier cache: in-memory LRU + optional disk persistence."""

    def __init__(
        self,
        max_memory_entries: int = 1000,
        disk_path: Optional[str] = None,
        default_ttl: int = 3600,
    ):
        self._memory: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._max_memory = max_memory_entries
        self._disk_path = disk_path
        self._default_ttl = default_ttl

        if disk_path:
            os.makedirs(disk_path, exist_ok=True)

    async def get(self, key: str) -> Optional[Any]:
        """Retrieve from cache. Returns None on miss or expiry."""
        # Check memory first
        if key in self._memory:
            value, expires_at = self._memory[key]
            if time.time() < expires_at:
                self._memory.move_to_end(key)
                return value
            else:
                del self._memory[key]

        # Check disk
        if self._disk_path:
            disk_val = await self._disk_get(key)
            if disk_val is not None:
                # Promote to memory
                self._memory_set(key, disk_val, self._default_ttl)
                return disk_val

        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store in cache with TTL (seconds)."""
        ttl = ttl or self._default_ttl
        self._memory_set(key, value, ttl)

        if self._disk_path:
            await self._disk_set(key, value, ttl)

    def _memory_set(self, key: str, value: Any, ttl: int) -> None:
        """Set in memory LRU cache."""
        expires_at = time.time() + ttl
        self._memory[key] = (value, expires_at)
        self._memory.move_to_end(key)

        # Evict oldest if over limit
        while len(self._memory) > self._max_memory:
            self._memory.popitem(last=False)

    async def _disk_get(self, key: str) -> Optional[Any]:
        """Read from disk cache."""
        path = os.path.join(self._disk_path, f"{key}.json")
        if not os.path.exists(path):
            return None
        try:
            async with aiofiles.open(path, "r") as f:
                data = json.loads(await f.read())
            if time.time() < data["expires_at"]:
                return data["value"]
            else:
                os.unlink(path)
                return None
        except Exception as e:
            logger.debug("Disk cache read error for {}: {}", key, e)
            return None

    async def _disk_set(self, key: str, value: Any, ttl: int) -> None:
        """Write to disk cache."""
        path = os.path.join(self._disk_path, f"{key}.json")
        try:
            data = {"value": value, "expires_at": time.time() + ttl}
            async with aiofiles.open(path, "w") as f:
                await f.write(json.dumps(data, default=str))
        except Exception as e:
            logger.debug("Disk cache write error for {}: {}", key, e)

    def invalidate(self, key: str) -> None:
        """Remove a key from both tiers."""
        self._memory.pop(key, None)
        if self._disk_path:
            path = os.path.join(self._disk_path, f"{key}.json")
            if os.path.exists(path):
                os.unlink(path)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._memory.clear()

    @property
    def memory_size(self) -> int:
        return len(self._memory)
