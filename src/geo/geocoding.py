"""Reverse geocoding and coordinate validation utilities.

Provides unified reverse geocoding from multiple sources.
"""

from __future__ import annotations

from typing import Any, Optional

from loguru import logger

from src.utils.geo_math import validate_coordinates


async def reverse_geocode(
    lat: float,
    lon: float,
) -> dict[str, Any] | None:
    """Reverse geocode coordinates to a location name via OSM Nominatim."""
    if not validate_coordinates(lat, lon):
        return None

    return await _reverse_geocode_osm(lat, lon)


async def _reverse_geocode_osm(lat: float, lon: float) -> dict[str, Any] | None:
    """Reverse geocode via OSM Nominatim."""
    try:
        import httpx

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://nominatim.openstreetmap.org/reverse",
                params={
                    "lat": lat,
                    "lon": lon,
                    "format": "json",
                    "zoom": 16,
                    "addressdetails": 1,
                },
                headers={"User-Agent": "OpenGeoSpy/2.0"},
            )
            resp.raise_for_status()
            data = resp.json()

        address = data.get("address", {})
        return {
            "display_name": data.get("display_name", ""),
            "country": address.get("country", ""),
            "country_code": address.get("country_code", ""),
            "state": address.get("state", ""),
            "city": (
                address.get("city")
                or address.get("town")
                or address.get("village")
                or address.get("municipality", "")
            ),
            "suburb": address.get("suburb", ""),
            "road": address.get("road", ""),
            "postcode": address.get("postcode", ""),
        }

    except Exception as e:
        logger.error("OSM reverse geocode failed: {}", e)
        return None


def validate_location_result(result: dict[str, Any]) -> bool:
    """Validate a location result has minimum required fields."""
    lat = result.get("lat") or result.get("latitude")
    lon = result.get("lon") or result.get("longitude")
    if lat is None or lon is None:
        return False
    try:
        return validate_coordinates(float(lat), float(lon))
    except (TypeError, ValueError):
        return False
