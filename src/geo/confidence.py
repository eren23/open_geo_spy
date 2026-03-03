"""Confidence scoring utilities for search results.

Provides dynamic confidence calculation based on result quality signals
rather than hardcoded values.
"""

from __future__ import annotations

from typing import Any


def safe_coords(lat: Any, lon: Any) -> tuple[float | None, float | None]:
    """Validate and convert coordinates to floats with bounds checking.
    
    Args:
        lat: Latitude value (any type)
        lon: Longitude value (any type)
        
    Returns:
        Tuple of (latitude, longitude) as floats, or (None, None) if invalid
    """
    try:
        lat_f = float(lat) if lat is not None else None
        lon_f = float(lon) if lon is not None else None
        
        if lat_f is None or lon_f is None:
            return None, None
            
        # Validate bounds
        if not (-90 <= lat_f <= 90):
            return None, None
        if not (-180 <= lon_f <= 180):
            return None, None
            
        return lat_f, lon_f
    except (TypeError, ValueError):
        return None, None


def calculate_search_confidence(
    result: dict[str, Any],
    query: str,
    base_confidence: float = 0.3,
    position: int = 0,
) -> float:
    """Calculate dynamic confidence score for a search result.
    
    Scoring factors:
    - Base confidence (default 0.3)
    - Query match in title (+0.15)
    - Query match in snippet (+0.08)
    - Coordinates present (+0.2)
    - Address present (+0.1)
    - Position decay (top results get boost)
    
    Args:
        result: Search result dict with title, snippet, latitude, longitude, address
        query: Original search query
        base_confidence: Starting confidence value
        position: Result position (0-indexed) for decay calculation
        
    Returns:
        Confidence score between 0.0 and 1.0
    """
    confidence = base_confidence
    query_lower = query.lower().strip()
    
    # Title match boost
    title = result.get("title", "") or ""
    if query_lower in title.lower():
        confidence += 0.15
    
    # Snippet match boost (smaller than title)
    snippet = result.get("snippet", "") or result.get("description", "") or ""
    if query_lower in snippet.lower():
        confidence += 0.08
    
    # Coordinates present - strong signal for geolocation
    lat, lon = safe_coords(result.get("latitude"), result.get("longitude"))
    if lat is not None and lon is not None:
        confidence += 0.2
    
    # Address present - good for local verification
    address = result.get("address", "")
    if address:
        confidence += 0.1
    
    # Rating present - indicates real place
    rating = result.get("rating")
    if rating is not None:
        try:
            rating_f = float(rating)
            if rating_f >= 4.0:
                confidence += 0.05
        except (TypeError, ValueError):
            pass
    
    # Position-based decay: top results are more relevant
    # Decay formula: 0.1 * (1 - position/10), minimum 0
    if position > 0:
        decay = max(0, 0.1 * (1 - position / 10))
        confidence += decay
    
    # Ensure bounds
    return max(0.0, min(1.0, round(confidence, 3)))


def calculate_osm_confidence(
    result: dict[str, Any],
    distance_km: float | None = None,
) -> float:
    """Calculate confidence for OSM results based on quality signals.
    
    Args:
        result: OSM result with tags, distance_km, importance
        distance_km: Distance from search center (if applicable)
        
    Returns:
        Confidence score between 0.0 and 1.0
    """
    confidence = 0.5  # OSM base confidence
    
    # Distance penalty (closer is better)
    if distance_km is not None:
        if distance_km < 1:
            confidence += 0.15
        elif distance_km < 5:
            confidence += 0.08
        elif distance_km > 50:
            confidence -= 0.1
    
    # Importance score (Nominatim provides this)
    importance = result.get("importance")
    if importance is not None:
        try:
            conf_boost = min(0.2, float(importance) * 0.2)
            confidence += conf_boost
        except (TypeError, ValueError):
            pass
    
    # POI type boost - some types are more distinctive
    osm_type = result.get("type", "") or ""
    tags = result.get("tags", {}) or {}
    
    # Tourism/landmark POIs are highly distinctive
    if tags.get("tourism") or tags.get("historic"):
        confidence += 0.1
    
    # Transport infrastructure is distinctive
    if tags.get("railway") or tags.get("aeroway"):
        confidence += 0.08
    
    # Named places are better than unnamed
    if tags.get("name"):
        confidence += 0.05
    
    return max(0.0, min(1.0, round(confidence, 3)))


def calculate_visual_match_confidence(
    similarity_score: float,
    source_reliability: float = 0.8,
) -> float:
    """Calculate confidence for visual similarity matches.
    
    Args:
        similarity_score: Embedding similarity (0-1)
        source_reliability: Reliability of the image source
        
    Returns:
        Confidence score between 0.0 and 1.0
    """
    # High similarity is a strong signal
    if similarity_score >= 0.9:
        return min(1.0, 0.85 * source_reliability)
    elif similarity_score >= 0.8:
        return min(1.0, 0.7 * source_reliability)
    elif similarity_score >= 0.7:
        return min(1.0, 0.55 * source_reliability)
    elif similarity_score >= 0.6:
        return min(1.0, 0.4 * source_reliability)
    else:
        return min(1.0, similarity_score * 0.5 * source_reliability)


def normalize_result(result: dict[str, Any], provider: str) -> dict[str, Any]:
    """Normalize result dict from different providers to common format.
    
    Args:
        result: Raw result from provider
        provider: Provider name (serper, brave, searxng)
        
    Returns:
        Normalized dict with: title, snippet, link, latitude, longitude, address
    """
    normalized = {
        "title": "",
        "snippet": "",
        "link": "",
        "latitude": None,
        "longitude": None,
        "address": None,
        "rating": None,
        "position": result.get("position", 0),
    }
    
    if provider == "serper":
        normalized["title"] = result.get("title", "")
        normalized["snippet"] = result.get("snippet", "")
        normalized["link"] = result.get("link", "")
        normalized["latitude"] = result.get("latitude")
        normalized["longitude"] = result.get("longitude")
        normalized["address"] = result.get("address")
        normalized["rating"] = result.get("rating")
        
    elif provider == "brave":
        normalized["title"] = result.get("title", "")
        normalized["snippet"] = result.get("description", "")
        normalized["link"] = result.get("url", "") or result.get("link", "")
        
    elif provider == "searxng":
        normalized["title"] = result.get("title", "")
        normalized["snippet"] = result.get("content", "") or result.get("snippet", "")
        normalized["link"] = result.get("url", "") or result.get("link", "")
    
    return normalized
