"""Mapillary v4 API client for free street-level imagery.

Only active if MAPILLARY_ACCESS_TOKEN env var is set.
Provides nearby street-level image URLs for visual verification.
"""

from __future__ import annotations

import httpx
from loguru import logger


class MapillaryClient:
    """Lightweight client for Mapillary v4 image search API."""

    BASE_URL = "https://graph.mapillary.com"

    def __init__(self, access_token: str):
        self.access_token = access_token
        self._client = httpx.AsyncClient(
            timeout=10.0,
            headers={"Authorization": f"OAuth {access_token}"},
            transport=httpx.AsyncHTTPTransport(retries=2),
        )

    async def search_nearby(
        self,
        lat: float,
        lon: float,
        radius_m: int = 500,
        limit: int = 5,
    ) -> list[dict]:
        """Search for street-level images near a coordinate.

        Returns list of dicts: {image_url, thumb_url, lat, lon, id}.
        """
        bbox = _bbox_from_point(lat, lon, radius_m)
        try:
            resp = await self._client.get(
                f"{self.BASE_URL}/images",
                params={
                    "fields": "id,geometry,thumb_1024_url,thumb_256_url",
                    "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                    "limit": limit,
                },
            )
            resp.raise_for_status()
            data = resp.json()

            results = []
            for feature in data.get("data", []):
                geom = feature.get("geometry", {}).get("coordinates", [])
                if len(geom) >= 2:
                    results.append({
                        "id": feature.get("id"),
                        "image_url": feature.get("thumb_1024_url", ""),
                        "thumb_url": feature.get("thumb_256_url", ""),
                        "lat": geom[1],
                        "lon": geom[0],
                    })
            return results

        except Exception as e:
            logger.debug("Mapillary search failed near ({}, {}): {}", lat, lon, e)
            return []

    async def close(self):
        await self._client.aclose()


def _bbox_from_point(lat: float, lon: float, radius_m: int) -> tuple[float, float, float, float]:
    """Simple bounding box from center point and radius in meters.

    Returns (min_lon, min_lat, max_lon, max_lat).
    """
    # ~111km per degree latitude, ~111*cos(lat) per degree longitude
    import math

    lat_delta = radius_m / 111_000
    lon_delta = radius_m / (111_000 * max(math.cos(math.radians(lat)), 0.01))

    return (
        lon - lon_delta,
        lat - lat_delta,
        lon + lon_delta,
        lat + lat_delta,
    )
