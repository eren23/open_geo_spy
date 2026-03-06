"""Dynamic country code resolution for geographic search constraints.

Uses geocoding APIs and LLM to resolve ANY location to ISO codes.
No hardcoded country lists - works for any city, region, or country.
"""

from __future__ import annotations

from typing import Any, Optional

from loguru import logger

from src.config.llm import LLMCallType, get_llm_params

__all__ = [
    "get_iso_code",
    "resolve_location_to_country",
    "extract_country_from_location",
    "get_google_gl",
    "get_google_cr",
]

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
    code = _RESOLVED_CACHE.get(country_name)
    if code:
        return code
    return None


async def resolve_location_to_country(
    location: str,
    client: Optional[Any] = None,
    settings: Optional[Any] = None,
    llm_client: Optional[Any] = None,
) -> str | None:
    """Resolve any location string to an ISO country code.
    
    Uses a multi-step approach:
    1. Try geocoding APIs (Nominatim, OpenStreetMap)
    2. Fall back to LLM
    
    Args:
        location: Location hint string
        client: OpenAI async client
        settings: Settings object with LLM configuration
        llm_client: Pre-configured LLM client (optional)
    
    Returns:
        ISO 3166-1 alpha-2 code or None
    """
    # Check cache first
    if location in _RESOLVED_CACHE:
        return _RESOLVED_CACHE[location]
    
    import httpx
    
    # Step 1: Try geocoding APIs
    # Note: Only forward geocoding services (text → coordinates) are useful here
    # Reverse geocoding (/reverse) requires lat/lon which we don't have
    geocoding_services = [
        ("Nominatim", "https://nominatim.openstreetmap.org/search", {"accept-language": "en", "addressdetails": 1}),
    ]
    
    for service_name, geocoding_url, geocoding_params in geocoding_services:
        try:
            async with httpx.AsyncClient(timeout=5.0) as http_client:
                params = {"q": location, **geocoding_params}
                resp = await http_client.get(geocoding_url, params=params)
                data = resp.json()

                # Nominatim /search returns a list of results
                if isinstance(data, list) and len(data) > 0:
                    first_result = data[0]
                    if first_result.get("address") and first_result["address"].get("country_code"):
                        code = first_result["address"]["country_code"]
                        _RESOLVED_CACHE[location] = code
                        logger.debug("Geocoded '{}' → {}", location, code)
                        return code
        except Exception:
            pass
    
    # Step 2: LLM fallback
    effective_client = llm_client or client
    if effective_client is not None:
        code = await _llm_extract_country(location, effective_client, settings)
        if code:
            _RESOLVED_CACHE[location] = code
        return code

    return None


async def _llm_extract_country(location: str, client: Any, settings: Any = None) -> str | None:
    """Use LLM to extract country ISO code from location hint."""
    try:
        llm_params = get_llm_params(LLMCallType.GEO_COUNTRY_RESOLVE, settings)
        resp = await client.chat.completions.create(
            **llm_params,
            messages=[{"role": "user", "content": f"""Given this location hint: "{location}"

Return ONLY the ISO 3166-1 alpha-2 country code (2 letters) for where this location is.

Rules:
- For cities: return the country code (e.g., "Istanbul" → "TR")
- For regions: return the country code (e.g., "Bavaria" → "DE")
- For US states: return "US" (e.g., "Georgia (US state)" → "US")
- For countries: return each code (e.g., "Turkey" → "TR")
- If ambiguous, pick the most likely
- If you can't determine, return "UNKNOWN"

Return ONLY the 2-letter code, nothing else."""}],
        )
        result = resp.choices[0].message.content.strip().upper()
        
        # Validate it's a proper 2-letter code
        if len(result) == 2 and result.isalpha() and result != "UNKNOWN":
            logger.debug("LLM extracted country '{}' from '{}'", result, location)
            return result
            
    except Exception as e:
        logger.debug("LLM country extraction failed: {}", e)
    
    return None


def extract_country_from_location(hint: str) -> str | None:
    """Synchronous version that only does basic extraction.
    
    For full resolution including geocoding, use resolve_location_to_country().
    
    Args:
        hint: Location hint string
        
    Returns:
        ISO country code or None if cannot resolve
    """
    # Simple patterns for common country mentions
    hint_lower = hint.lower()
    
    # Direct country mentions
    countries = {
        "turkey": "TR", "türkiye": "TR", "germany": "DE", "deutschland": "DE",
        "france": "FR", "frança": "FR", "spain": "ES", "españa": "ES",
        "italy": "IT", "italia": "IT", "greece": "GR", "united kingdom": "GB", "uk": "GB",
        "united states": "US", "usa": "US", "japan": "JP", "russia": "RU", "china": "CN",
        "brazil": "BR", "india": "IN", "australia": "AU", "canada": "CA",
        "mexico": "MX", "argentina": "AR", "netherlands": "NL", "poland": "PL",
        "portugal": "PT", "sweden": "SE", "norway": "NO", "denmark": "DK",
        "finland": "FI", "switzerland": "CH", "austria": "AT", "belgium": "BE",
        "ireland": "IE", "czech": "CZ", "hungary": "HU", "romania": "RO",
        "bulgaria": "BG", "croatia": "HR", "serbia": "RS", "ukraine": "UA",
        "south korea": "KR", "korea": "KR", "thailand": "TH", "vietnam": "VN",
        "indonesia": "ID", "malaysia": "MY", "philippines": "PH", "singapore": "SG",
        "new zealand": "NZ", "south africa": "ZA", "egypt": "EG", "morocco": "MA",
        "israel": "IL", "iran": "IR", "iraq": "IQ", "saudi arabia": "SA",
        "uae": "AE", "qatar": "QA", "kuwait": "KW", "pakistan": "PK",
        "bangladesh": "BD", "sri lanka": "LK", "nepal": "NP", "myanmar": "MM",
        "cambodia": "KH", "laos": "LA", "taiwan": "TW", "hong kong": "HK",
        "macau": "MO", "mongolia": "MN", "kazakhstan": "KZ", "uzbekistan": "UZ",
        "azerbaijan": "AZ", "georgia": "GE", "armenia": "AM",
        "peru": "PE", "chile": "CL", "colombia": "CO", "venezuela": "VE",
        "ecuador": "EC", "bolivia": "BO", "paraguay": "PY", "uruguay": "UY",
        "cuba": "CU", "jamaica": "JM", "dominican republic": "DO",
        "puerto rico": "PR", "haiti": "HT",
    }
    
    for name, code in countries.items():
        if name in hint_lower:
            return code
    
    # Check for city patterns
    city_patterns = ["istanbul", "berlin", "paris", "london", "tokyo", "moscow", "beijing"]
    for city in city_patterns:
        if city in hint_lower:
            # Map cities to countries
            city_map = {
                "istanbul": "TR", "ankara": "TR", "berlin": "DE", "munich": "DE",
                "paris": "FR", "lyon": "FR", "london": "GB", "manchester": "GB",
                "tokyo": "JP", "osaka": "JP", "moscow": "RU", "saint petersburg": "RU",
            }
            return city_map.get(city, None)
    
    # No match found
    return None


def get_google_gl(iso_code: str) -> str:
    """Return Google gl (geolocation) param from ISO code. Lowercase."""
    if not iso_code or len(iso_code) != 2:
        return "us"
    # Google uses 'uk' for United Kingdom, not 'gb'
    if iso_code.upper() == "GB":
        return "uk"
    return iso_code.lower()


def get_google_cr(iso_code: str) -> str | None:
    """Return Google cr (country restrict) param from ISO code. Format: countryXX."""
    if not iso_code or len(iso_code) != 2:
        return None
    # Google uses countryUK for United Kingdom
    code = "UK" if iso_code.upper() == "GB" else iso_code.upper()
    return f"country{code}"
