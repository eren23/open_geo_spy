"""Stealth web search operations using browser pool.

Provides Google Maps search, Street View verification, and generic
URL scraping with stealth browser automation.
"""

from __future__ import annotations

import re
from typing import Any

from loguru import logger

from src.browser.browser_pool import BrowserPool
from src.evidence.chain import Evidence, EvidenceSource


class BrowserSearch:
    """Stealth browser-based search operations (Tier 2)."""

    def __init__(self, pool: BrowserPool):
        self.pool = pool

    async def search_google_maps(self, query: str) -> list[dict[str, Any]]:
        """Search Google Maps with stealth browser.

        Returns list of place results with coordinates.
        """
        try:
            query = query[:120]  # Google Maps can't handle paragraph-length queries
            async with self.pool.get_page() as page:
                url = f"https://www.google.com/maps/search/{query.replace(' ', '+')}"
                await page.goto(url, wait_until="networkidle", timeout=20000)
                await self.pool.throttle()

                # Wait for results to load
                await page.wait_for_timeout(3000)

                # Extract coordinates from URL (Google Maps encodes lat/lon in URL)
                current_url = page.url
                coords = _extract_coords_from_url(current_url)

                # Try to extract place info from the page
                results = []
                try:
                    # Get place name from title
                    title = await page.title()
                    if title and "Google Maps" not in title:
                        results.append({
                            "name": title.split(" - ")[0].strip(),
                            "lat": coords[0] if coords else None,
                            "lon": coords[1] if coords else None,
                            "source": "google_maps_browser",
                        })
                except Exception:
                    pass

                if coords and not results:
                    results.append({
                        "name": query,
                        "lat": coords[0],
                        "lon": coords[1],
                        "source": "google_maps_browser",
                    })

                return results

        except Exception as e:
            logger.error("Google Maps browser search failed: {}", e)
            return []

    async def verify_street_view(self, lat: float, lon: float) -> dict[str, Any] | None:
        """Check if Street View is available at coordinates and extract info."""
        try:
            async with self.pool.get_page() as page:
                url = f"https://www.google.com/maps/@{lat},{lon},3a,75y,0h,90t/data=!3m6!1e1!3m4!1s"
                await page.goto(url, wait_until="domcontentloaded", timeout=15000)
                await self.pool.throttle()

                # Check if Street View loaded (vs being redirected)
                current_url = page.url
                has_street_view = "!3m" in current_url or "streetview" in current_url.lower()

                return {
                    "available": has_street_view,
                    "lat": lat,
                    "lon": lon,
                    "url": current_url,
                }

        except Exception as e:
            logger.debug("Street View check failed: {}", e)
            return None

    async def scrape_url(self, url: str) -> str:
        """Scrape a URL and return text content."""
        try:
            async with self.pool.get_page() as page:
                await page.goto(url, wait_until="domcontentloaded", timeout=15000)
                await self.pool.throttle()

                # Extract main text content
                content = await page.evaluate("""
                    () => {
                        const main = document.querySelector('main') || document.querySelector('article') || document.body;
                        return main ? main.innerText : document.body.innerText;
                    }
                """)
                return content[:5000] if content else ""

        except Exception as e:
            logger.error("URL scrape failed for {}: {}", url, e)
            return ""

    def to_evidence(self, results: list[dict], query: str = "") -> list[Evidence]:
        """Convert browser search results to Evidence."""
        evidences = []
        for r in results:
            if r.get("lat") and r.get("lon"):
                evidences.append(
                    Evidence(
                        source=EvidenceSource.BROWSER,
                        content=f"Browser search: {r.get('name', query)}",
                        confidence=0.6,
                        latitude=r["lat"],
                        longitude=r["lon"],
                        url=r.get("url"),
                        metadata={"source": r.get("source", "browser"), "query": query},
                    )
                )
        return evidences


def _extract_coords_from_url(url: str) -> tuple[float, float] | None:
    """Extract lat/lon from a Google Maps URL."""
    patterns = [
        r"@(-?\d+\.?\d*),(-?\d+\.?\d*)",  # @lat,lon
        r"!3d(-?\d+\.?\d*)!4d(-?\d+\.?\d*)",  # !3dlat!4dlon
        r"ll=(-?\d+\.?\d*),(-?\d+\.?\d*)",  # ll=lat,lon
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            try:
                lat, lon = float(match.group(1)), float(match.group(2))
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    return (lat, lon)
            except ValueError:
                continue
    return None
