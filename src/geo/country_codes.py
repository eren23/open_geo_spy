"""Dynamic country code resolution for geographic search constraints.

Uses geocoding APIs and LLM to resolve ANY location to ISO codes.
No hardcoded country lists - works for any city, region, or country.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

# Simple cache for resolved locations (in-memory, expires on restart)
_RESOLVED_CACHE: dict[str, str | None] = {}


def get_iso_code(country_name: str) -> str | None:
    """Get ISO country code from a country name or validate an existing ISO code.
    
    This is a synchronous helper that:
    1. Returns 2-letter codes as-is (validated)
    2. Checks the async resolution cache for previously resolved names
    3. Returns None for uncached names (use resolve_location_to_country for full resolution)
    
    Args:
        country_name: Country name or ISO code
    
    Returns:
        ISO 3166-1 alpha-2 code or None
    """
    if not country_name:
        return None
    
    name = country_name.strip()
    
    # Already an ISO code - validate and return
    if len(name) == 2 and name.isalpha():
        return name.upper()
    
    # Check cache for previously resolved names
    name_lower = name.lower()
    if name_lower in _RESOLVED_CACHE:
        return _RESOLVED_CACHE[name_lower]
    
    # Not in cache - caller should use resolve_location_to_country for full resolution
    return None


def get_google_cr(iso_code: str) -> str | None:
    """Get Google cr parameter value from ISO code.
    
    Args:
        iso_code: ISO 3166-1 alpha-2 code like "TR", "US"
    
    Returns:
        Google cr value like "countryTR" or None if not a valid ISO code
    """
    if not iso_code or len(iso_code) != 2:
        return None
    return f"country{iso_code.upper()}"


def get_google_gl(iso_code: str) -> str:
    """Get Google gl parameter value from ISO code.
    
    Args:
        iso_code: ISO 3166-1 alpha-2 code
    
    Returns:
        Google gl value (lowercase ISO code, defaults to "us")
    """
    return iso_code.lower() if iso_code and len(iso_code) == 2 else "us"


async def resolve_location_to_country(
    location_hint: str,
    llm_client: Any = None,
) -> str | None:
    """Resolve any location hint to an ISO country code.
    
    Uses multiple strategies in order:
    1. Check cache for previously resolved hints
    2. Use geocoding API (Nominatim/OSM) to get country
    3. Fallback to LLM extraction for ambiguous cases
    
    Args:
        location_hint: Any location string (city, country, region, etc.)
        llm_client: Optional OpenAI client for LLM fallback
    
    Returns:
        ISO 3166-1 alpha-2 code like "TR", "JP", "US" or None
    """
    if not location_hint:
        return None
    
    hint_normalized = location_hint.strip().lower()
    
    # Check cache first
    if hint_normalized in _RESOLVED_CACHE:
        return _RESOLVED_CACHE[hint_normalized]
    
    # Try geocoding first (free, no LLM cost)
    iso_code = await _geocode_to_country(hint_normalized)
    if iso_code:
        _RESOLVED_CACHE[hint_normalized] = iso_code
        return iso_code
    
    # Fallback to LLM if available
    if llm_client:
        iso_code = await _llm_extract_country(location_hint, llm_client)
        if iso_code:
            _RESOLVED_CACHE[hint_normalized] = iso_code
            return iso_code
    
    # Cache negative result to avoid repeated lookups
    _RESOLVED_CACHE[hint_normalized] = None
    return None


async def _geocode_to_country(location: str) -> str | None:
    """Use Nominatim/OSM geocoding to resolve location to country ISO code.
    
    Args:
        location: Location string to geocode
    
    Returns:
        ISO country code or None
    """
    try:
        import httpx
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Use Nominatim (free, no API key needed)
            resp = await client.get(
                "https://nominatim.openstreetmap.org/search",
                params={
                    "q": location,
                    "format": "json",
                    "limit": 1,
                    "addressdetails": 1,
                },
                headers={"User-Agent": "OpenGeoSpy/1.0"},
            )
            resp.raise_for_status()
            results = resp.json()
            
            if results:
                address = results[0].get("address", {})
                country_code = address.get("country_code", "").upper()
                if country_code and len(country_code) == 2:
                    logger.debug(
                        "Geocoded '{}' to country '{}' via Nominatim",
                        location, country_code
                    )
                    return country_code
                
    except Exception as e:
        logger.debug("Geocoding failed for '{}': {}", location, e)
    
    return None


async def _llm_extract_country(location: str, client: Any) -> str | None:
    """Use LLM to extract country ISO code from location hint.
    
    Handles:
    - City names: "Istanbul" → TR
    - Regional names: "Catalonia" → ES
    - Ambiguous names: "Georgia" → US state or GE country
    - Misspellings: "Constaninople" → TR
    
    Args:
        location: Location hint string
        client: OpenAI async client
    
    Returns:
        ISO country code or None
    """
    try:
        prompt = f"""Given this location hint: "{location}"

Return ONLY the ISO 3166-1 alpha-2 country code (2 letters) for where this location is.

Rules:
- For cities: return the country code (e.g., "Istanbul" → "TR")
- For regions: return the country code (e.g., "Bavaria" → "DE")
- For US states: return "US" (e.g., "Georgia (US state)" → "US")
- For countries: return their code (e.g., "Turkey" → "TR")
- If ambiguous, pick the most likely
- If you can't determine, return "UNKNOWN"

Return ONLY the 2-letter code, nothing else."""

        resp = await client.chat.completions.create(
            model="google/gemini-2.5-flash",  # Fast, cheap
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        
        result = resp.choices[0].message.content.strip().upper()
        
        # Validate it's a proper 2-letter code
        if len(result) == 2 and result.isalpha() and result != "UNKNOWN":
            logger.debug("LLM extracted country '{}' from '{}'", result, location)
            return result
            
    except Exception as e:
        logger.debug("LLM country extraction failed: {}", e)
    
    return None


def extract_country_from_hint(hint: str) -> str | None:
    """Synchronous version that only does basic extraction.
    
    For full resolution including geocoding, use resolve_location_to_country().
    
    Args:
        hint: Location hint string
    
    Returns:
        ISO code if directly provided, otherwise None
    """
    if not hint:
        return None
    
    hint_clean = hint.strip()
    
    # Direct ISO code provided
    if len(hint_clean) == 2 and hint_clean.isalpha():
        return hint_clean.upper()
    
    # Check cache for previously resolved
    hint_lower = hint_clean.lower()
    if hint_lower in _RESOLVED_CACHE:
        return _RESOLVED_CACHE[hint_lower]
    
    return None
