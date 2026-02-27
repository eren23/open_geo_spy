"""Retry utilities using tenacity with intelligent error classification."""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable

from loguru import logger
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random_exponential,
)


class ErrorType(Enum):
    NETWORK = "network"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    SERVER_ERROR = "server_error"
    AUTH_ERROR = "auth_error"
    NOT_FOUND = "not_found"
    UNKNOWN = "unknown"


# Error pattern -> classification
_ERROR_PATTERNS: dict[ErrorType, list[str]] = {
    ErrorType.NETWORK: ["connection", "network", "unreachable", "dns", "refused", "reset"],
    ErrorType.TIMEOUT: ["timeout", "timed out", "deadline"],
    ErrorType.RATE_LIMIT: ["rate limit", "429", "too many requests", "quota"],
    ErrorType.SERVER_ERROR: ["500", "502", "503", "504", "internal server error", "bad gateway"],
    ErrorType.AUTH_ERROR: ["401", "403", "unauthorized", "forbidden", "invalid key"],
    ErrorType.NOT_FOUND: ["404", "not found"],
}


def classify_error(error: Exception) -> ErrorType:
    """Classify an exception into an error type."""
    error_str = str(error).lower()
    for error_type, patterns in _ERROR_PATTERNS.items():
        if any(p in error_str for p in patterns):
            return error_type
    return ErrorType.UNKNOWN


def _log_retry(retry_state: RetryCallState) -> None:
    """Log retry attempts."""
    if retry_state.outcome and retry_state.outcome.failed:
        exc = retry_state.outcome.exception()
        error_type = classify_error(exc)
        logger.warning(
            "Retry attempt {attempt} for {fn} ({error_type}): {error}",
            attempt=retry_state.attempt_number,
            fn=retry_state.fn.__name__ if retry_state.fn else "unknown",
            error_type=error_type.value,
            error=str(exc)[:200],
        )


# --- Pre-built retry decorators ---


def retry_llm(max_attempts: int = 3) -> Callable:
    """Retry decorator for LLM API calls with exponential backoff."""
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_random_exponential(multiplier=1, max=30),
        retry=retry_if_exception_type(Exception),
        before_sleep=_log_retry,
        reraise=True,
    )


def retry_network(max_attempts: int = 3) -> Callable:
    """Retry decorator for network requests."""
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        before_sleep=_log_retry,
        reraise=True,
    )


def retry_browser(max_attempts: int = 2) -> Callable:
    """Retry decorator for browser operations (shorter, fewer retries)."""
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=2, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=_log_retry,
        reraise=True,
    )


async def execute_with_retry(
    func: Callable,
    *args: Any,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    **kwargs: Any,
) -> Any:
    """Execute an async function with retry logic.

    Use this for one-off retries where a decorator is inconvenient.
    """
    import asyncio

    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_error = e
            error_type = classify_error(e)

            if error_type in (ErrorType.AUTH_ERROR, ErrorType.NOT_FOUND):
                raise  # Don't retry auth or 404 errors

            if attempt < max_attempts:
                delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                logger.warning(
                    "Attempt {}/{} failed ({}): {}. Retrying in {:.1f}s",
                    attempt,
                    max_attempts,
                    error_type.value,
                    str(e)[:200],
                    delay,
                )
                await asyncio.sleep(delay)

    raise last_error
