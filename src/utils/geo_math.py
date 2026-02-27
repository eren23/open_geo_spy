"""Geographic math utilities: Haversine distance, bounding boxes, coordinate validation."""

from __future__ import annotations

import math
from typing import Optional


EARTH_RADIUS_KM = 6371.0


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two points in kilometers."""
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return EARTH_RADIUS_KM * c


def validate_coordinates(lat: float, lon: float) -> bool:
    """Check if coordinates are within valid ranges."""
    return -90 <= lat <= 90 and -180 <= lon <= 180


def bounding_box(lat: float, lon: float, radius_km: float) -> tuple[float, float, float, float]:
    """Calculate bounding box (min_lat, min_lon, max_lat, max_lon) for a radius around a point."""
    lat_delta = radius_km / 111.0  # ~111 km per degree latitude
    lon_delta = radius_km / (111.0 * math.cos(math.radians(lat)))

    return (
        max(-90, lat - lat_delta),
        max(-180, lon - lon_delta),
        min(90, lat + lat_delta),
        min(180, lon + lon_delta),
    )


def weighted_centroid(
    points: list[tuple[float, float, float]],
) -> Optional[tuple[float, float]]:
    """Calculate weighted centroid from list of (lat, lon, weight) tuples.

    Returns None if no points or all weights are zero.
    """
    if not points:
        return None

    total_weight = sum(w for _, _, w in points)
    if total_weight == 0:
        return None

    avg_lat = sum(lat * w for lat, _, w in points) / total_weight
    avg_lon = sum(lon * w for _, lon, w in points) / total_weight

    return (avg_lat, avg_lon)


def geographic_spread(coords: list[tuple[float, float]]) -> float:
    """Calculate the maximum distance between any two points in a set (km).

    Useful for measuring model agreement - low spread = high agreement.
    """
    if len(coords) < 2:
        return 0.0

    max_dist = 0.0
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            d = haversine_distance(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
            if d > max_dist:
                max_dist = d
    return max_dist


def country_level_agreement(countries: list[str]) -> float:
    """Calculate agreement score from a list of country predictions.

    Returns 0-1 where 1 = all agree, 0 = all different.
    """
    if not countries:
        return 0.0
    from collections import Counter

    counts = Counter(c.lower().strip() for c in countries if c)
    if not counts:
        return 0.0
    most_common_count = counts.most_common(1)[0][1]
    return most_common_count / len(countries)
