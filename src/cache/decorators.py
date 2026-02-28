"""Caching decorator for async client methods."""

from __future__ import annotations

import functools
from typing import Any, Callable

from loguru import logger

from src.cache.store import CacheStore, make_key


def cached(source: str, ttl: int = 3600) -> Callable:
    """Decorator that caches async method results.

    Checks for a ``_cache`` attribute (CacheStore) on the instance.
    If absent or None, the call passes through uncached.

    Args:
        source: Cache namespace (e.g. "serper", "osm").
        ttl: Time-to-live in seconds.
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            cache: CacheStore | None = getattr(self, "_cache", None)
            if cache is None:
                return await fn(self, *args, **kwargs)

            # Build deterministic key from args
            params = {
                "args": list(args),
                "kwargs": kwargs,
            }
            key = make_key(source, params)

            hit = await cache.get(key)
            if hit is not None:
                logger.debug("Cache HIT [{}] key={}", source, key[:12])
                return hit

            result = await fn(self, *args, **kwargs)
            await cache.set(key, result, ttl)
            logger.debug("Cache MISS [{}] key={}", source, key[:12])
            return result

        return wrapper

    return decorator
