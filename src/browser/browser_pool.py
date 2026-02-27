"""Async browser pool manager with stealth and session isolation.

Manages a pool of browser contexts for parallel web operations
with automatic stealth injection and request throttling.
"""

from __future__ import annotations

import asyncio
import random
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from loguru import logger

from src.browser.stealth import StealthConfig
from src.config.settings import BrowserSettings


class BrowserPool:
    """Manages a pool of stealth browser contexts."""

    def __init__(self, settings: BrowserSettings):
        self.settings = settings
        self.pool_size = settings.pool_size
        self._playwright = None
        self._browser = None
        self._semaphore = asyncio.Semaphore(self.pool_size)
        self._initialized = False

    async def initialize(self):
        """Launch browser. Call once at startup."""
        if self._initialized:
            return

        try:
            from playwright.async_api import async_playwright

            self._playwright = await async_playwright().start()

            launch_args = StealthConfig.get_launch_args()

            self._browser = await self._playwright.chromium.launch(
                headless=True,
                args=launch_args,
            )
            self._initialized = True
            logger.info("Browser pool initialized (pool_size={})", self.pool_size)

        except Exception as e:
            logger.error("Browser initialization failed: {}", e)
            raise

    @asynccontextmanager
    async def get_page(self) -> AsyncGenerator:
        """Get a stealth browser page from the pool.

        Usage:
            async with pool.get_page() as page:
                await page.goto("https://example.com")
        """
        if not self._initialized:
            await self.initialize()

        async with self._semaphore:
            context = await self._browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/131.0.0.0 Safari/537.36"
                ),
                locale="en-US",
                timezone_id="America/New_York",
            )

            # Inject stealth scripts
            stealth_js = StealthConfig.get_stealth_init_script(
                enable_canvas_noise=self.settings.enable_canvas_noise,
                enable_webgl_spoof=self.settings.enable_webgl_spoof,
            )
            await context.add_init_script(stealth_js)

            page = await context.new_page()

            try:
                yield page
            finally:
                await page.close()
                await context.close()

    async def throttle(self):
        """Random delay between requests to avoid detection."""
        delay = random.uniform(
            self.settings.request_delay_min,
            self.settings.request_delay_max,
        )
        await asyncio.sleep(delay)

    async def close(self):
        """Shut down the browser pool."""
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        self._initialized = False
        logger.info("Browser pool closed")
