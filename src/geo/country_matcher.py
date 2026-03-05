"""Robust country name matching with aliases, ISO codes, and fuzzy matching.

Handles:
- Native names: "Deutschland" = "Germany"
- Abbreviations: "US" = "United States", "UK" = "United Kingdom"  
- Common aliases: "Holland" = "Netherlands"
- Misspellings: Similarity-based fuzzy matching
- ISO codes: "DE", "US", "GB"
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Optional

# Comprehensive country name aliases mapping
# Format: normalized_name -> ISO 3166-1 alpha-2 code
_COUNTRY_ALIASES: dict[str, str] = {
    # A
    "afghanistan": "AF",
    "albania": "AL",
    "algeria": "DZ",
    "andorra": "AD",
    "angola": "AO",
    "argentina": "AR",
    "armenia": "AM",
    "australia": "AU",
    "austria": "AT",
    "osterreich": "AT",  # Native (normalized, without umlaut)
    "azerbaijan": "AZ",
    
    # B
    "bahamas": "BS",
    "bahrain": "BH",
    "bangladesh": "BD",
    "barbados": "BB",
    "belarus": "BY",
    "belgium": "BE",
    "belize": "BZ",
    "benin": "BJ",
    "bhutan": "BT",
    "bolivia": "BO",
    "bosnia": "BA",
    "bosnia and herzegovina": "BA",
    "botswana": "BW",
    "brazil": "BR",
    "brasil": "BR",  # Portuguese
    "brunei": "BN",
    "bulgaria": "BG",
    "burkina faso": "BF",
    "burundi": "BI",
    
    # C
    "cambodia": "KH",
    "cameroon": "CM",
    "canada": "CA",
    "cape verde": "CV",
    "central african republic": "CF",
    "chad": "TD",
    "chile": "CL",
    "china": "CN",
    "colombia": "CO",
    "comoros": "KM",
    "congo": "CG",
    "democratic republic of congo": "CD",
    "costa rica": "CR",
    "croatia": "HR",
    "hrvatska": "HR",  # Native
    "cuba": "CU",
    "cyprus": "CY",
    "czech republic": "CZ",
    "czechia": "CZ",
    "cesko": "CZ",  # Native
    
    # D
    "denmark": "DK",
    "danmark": "DK",  # Native
    "djibouti": "DJ",
    "dominica": "DM",
    "dominican republic": "DO",
    
    # E
    "east timor": "TL",
    "ecuador": "EC",
    "egypt": "EG",
    "el salvador": "SV",
    "equatorial guinea": "GQ",
    "eritrea": "ER",
    "estonia": "EE",
    "eesti": "EE",  # Native
    "eswatini": "SZ",
    "ethiopia": "ET",
    
    # F
    "fiji": "FJ",
    "finland": "FI",
    "suomi": "FI",  # Native
    "france": "FR",
    
    # G
    "gabon": "GA",
    "gambia": "GM",
    "georgia": "GE",
    "germany": "DE",
    "deutschland": "DE",  # Native
    "ghana": "GH",
    "greece": "GR",
    "grenada": "GD",
    "guatemala": "GT",
    "guinea": "GN",
    "guinea-bissau": "GW",
    "guyana": "GY",
    
    # H
    "haiti": "HT",
    "honduras": "HN",
    "hungary": "HU",
    "magyarorszag": "HU",  # Native
    
    # I
    "iceland": "IS",
    "india": "IN",
    "indonesia": "ID",
    "iran": "IR",
    "iraq": "IQ",
    "ireland": "IE",
    "israel": "IL",
    "italy": "IT",
    "italia": "IT",  # Native
    
    # J
    "jamaica": "JM",
    "japan": "JP",
    "nihon": "JP",  # Native
    "jordan": "JO",
    
    # K
    "kazakhstan": "KZ",
    "kenya": "KE",
    "kiribati": "KI",
    "kosovo": "XK",
    "kuwait": "KW",
    "kyrgyzstan": "KG",
    
    # L
    "laos": "LA",
    "latvia": "LV",
    "latvija": "LV",  # Native
    "lebanon": "LB",
    "lesotho": "LS",
    "liberia": "LR",
    "libya": "LY",
    "liechtenstein": "LI",
    "lithuania": "LT",
    "lietuva": "LT",  # Native
    "luxembourg": "LU",
    
    # M
    "macedonia": "MK",
    "north macedonia": "MK",
    "madagascar": "MG",
    "malawi": "MW",
    "malaysia": "MY",
    "maldives": "MV",
    "mali": "ML",
    "malta": "MT",
    "marshall islands": "MH",
    "mauritania": "MR",
    "mauritius": "MU",
    "mexico": "MX",
    "micronesia": "FM",
    "moldova": "MD",
    "monaco": "MC",
    "mongolia": "MN",
    "montenegro": "ME",
    "morocco": "MA",
    "mozambique": "MZ",
    "myanmar": "MM",
    "burma": "MM",  # Historical
    
    # N
    "namibia": "NA",
    "nauru": "NR",
    "nepal": "NP",
    "netherlands": "NL",
    "holland": "NL",  # Common alias
    "nederland": "NL",  # Native
    "new zealand": "NZ",
    "nicaragua": "NI",
    "niger": "NE",
    "nigeria": "NG",
    "north korea": "KP",
    "norway": "NO",
    "norge": "NO",  # Native
    
    # O
    "oman": "OM",
    
    # P
    "pakistan": "PK",
    "palau": "PW",
    "palestine": "PS",
    "panama": "PA",
    "papua new guinea": "PG",
    "paraguay": "PY",
    "peru": "PE",
    "philippines": "PH",
    "poland": "PL",
    "polska": "PL",  # Native
    "portugal": "PT",
    "puerto rico": "PR",
    
    # Q
    "qatar": "QA",
    
    # R
    "romania": "RO",
    "russia": "RU",
    "russian federation": "RU",
    "rwanda": "RW",
    
    # S
    "saint kitts and nevis": "KN",
    "saint lucia": "LC",
    "saint vincent": "VC",
    "samoa": "WS",
    "san marino": "SM",
    "sao tome and principe": "ST",
    "saudi arabia": "SA",
    "senegal": "SN",
    "serbia": "RS",
    "seychelles": "SC",
    "sierra leone": "SL",
    "singapore": "SG",
    "slovakia": "SK",
    "slovenia": "SI",
    "solomon islands": "SB",
    "somalia": "SO",
    "south africa": "ZA",
    "south korea": "KR",
    "korea": "KR",  # Common usage
    "south sudan": "SS",
    "spain": "ES",
    "espana": "ES",  # Native (without accent)
    "españa": "ES",  # Native (with accent)
    "sri lanka": "LK",
    "sudan": "SD",
    "suriname": "SR",
    "swaziland": "SZ",  # Old name
    "sweden": "SE",
    "sverige": "SE",  # Native
    "switzerland": "CH",
    "schweiz": "CH",  # German native
    "suisse": "CH",  # French native
    "svizzera": "CH",  # Italian native
    "syria": "SY",
    
    # T
    "taiwan": "TW",
    "tajikistan": "TJ",
    "tanzania": "TZ",
    "thailand": "TH",
    "togo": "TG",
    "tonga": "TO",
    "trinidad and tobago": "TT",
    "tunisia": "TN",
    "turkey": "TR",
    "turkiye": "TR",  # Official Turkish
    "turkmenistan": "TM",
    "tuvalu": "TV",
    
    # U
    "uganda": "UG",
    "ukraine": "UA",
    "united arab emirates": "AE",
    "uae": "AE",
    "united kingdom": "GB",
    "uk": "GB",
    "great britain": "GB",
    "britain": "GB",
    "england": "GB",  # Often used interchangeably
    "scotland": "GB",
    "wales": "GB",
    "united states": "US",
    "united states of america": "US",
    "usa": "US",
    "us": "US",
    "america": "US",  # Common usage
    "uruguay": "UY",
    "uzbekistan": "UZ",
    
    # V
    "vanuatu": "VU",
    "vatican": "VA",
    "venezuela": "VE",
    "vietnam": "VN",
    "viet nam": "VN",
    
    # W
    "yemen": "YE",
    
    # Z
    "zambia": "ZM",
    "zimbabwe": "ZW",
}

# Reverse mapping: ISO code -> list of names
_ISO_TO_NAMES: dict[str, list[str]] = {}
for name, iso in _COUNTRY_ALIASES.items():
    if iso not in _ISO_TO_NAMES:
        _ISO_TO_NAMES[iso] = []
    _ISO_TO_NAMES[iso].append(name)


def normalize_country_name(name: str | None) -> str:
    """Normalize country name for comparison.
    
    - Lowercase
    - Remove punctuation
    - Strip whitespace
    - Remove "the" prefix
    - Preserve accented characters but also add unaccented version lookup
    """
    if not name:
        return ""
    
    # Lowercase and strip
    normalized = name.lower().strip()
    
    # Remove "the" prefix
    if normalized.startswith("the "):
        normalized = normalized[4:]
    
    # Remove punctuation except hyphens (but keep letters with diacritics)
    normalized = re.sub(r"[^\w\s\u00C0-\u017F-]", "", normalized)
    
    # Collapse multiple spaces
    normalized = re.sub(r"\s+", " ", normalized).strip()
    
    return normalized


def _remove_accents(text: str) -> str:
    """Remove accents from text for fallback matching."""
    import unicodedata
    # Normalize to decomposed form, then remove combining marks
    normalized = unicodedata.normalize('NFD', text)
    return ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')


def get_iso_code(name: str | None) -> Optional[str]:
    """Get ISO 3166-1 alpha-2 code from country name or alias.
    
    Args:
        name: Country name, alias, or ISO code (can be None)
        
    Returns:
        ISO code (e.g., "DE", "US") or None if not found
    """
    if not name:
        return None
    
    # Check if it's a 2-letter code (but check aliases first since UK->GB)
    name_stripped = name.strip()
    if len(name_stripped) == 2 and name_stripped.isalpha():
        upper = name_stripped.upper()
        # Check if this is a known alias (e.g., UK -> GB)
        normalized = name_stripped.lower()
        if normalized in _COUNTRY_ALIASES:
            return _COUNTRY_ALIASES[normalized]
        # Otherwise return as-is (valid ISO code)
        return upper
    
    normalized = normalize_country_name(name)
    if not normalized:
        return None
    
    # Direct lookup
    result = _COUNTRY_ALIASES.get(normalized)
    if result:
        return result
    
    # Try without accents for names like "España" -> "espana"
    unaccented = _remove_accents(normalized)
    if unaccented != normalized:
        result = _COUNTRY_ALIASES.get(unaccented)
        if result:
            return result
    
    return None


def get_all_names(iso_code: str) -> list[str]:
    """Get all known names for an ISO code.
    
    Args:
        iso_code: ISO 3166-1 alpha-2 code
        
    Returns:
        List of names including native names and aliases
    """
    return _ISO_TO_NAMES.get(iso_code.upper(), [])


def countries_match(hint: str | None, candidate: str | None, fuzzy_threshold: float = 0.85) -> bool:
    """Check if two country references match.
    
    Handles:
    - ISO codes: "DE" == "DE"
    - Aliases: "Deutschland" == "Germany"
    - Abbreviations: "US" == "United States"
    - Fuzzy matching for typos: "Gernmany" ~= "Germany"
    
    Args:
        hint: User-provided country hint (can be None)
        candidate: Country from prediction/evidence (can be None)
        fuzzy_threshold: Similarity threshold for fuzzy matching (0-1)
        
    Returns:
        True if countries match, False if either is None/empty
    """
    if not hint or not candidate:
        return False
    
    # Normalize both
    hint_norm = normalize_country_name(hint)
    cand_norm = normalize_country_name(candidate)
    
    if not hint_norm or not cand_norm:
        return False
    
    # Direct string match
    if hint_norm == cand_norm:
        return True
    
    # Get ISO codes for both
    hint_iso = get_iso_code(hint_norm)
    cand_iso = get_iso_code(cand_norm)
    
    # Both resolved to same ISO code
    if hint_iso and cand_iso and hint_iso == cand_iso:
        return True
    
    # One is ISO code, check if other resolves to it
    if hint_iso and len(cand_norm) == 2 and cand_norm.isalpha():
        if hint_iso == cand_norm.upper():
            return True
    if cand_iso and len(hint_norm) == 2 and hint_norm.isalpha():
        if cand_iso == hint_norm.upper():
            return True
    
    # Check if hint is substring of candidate or vice versa
    if hint_norm in cand_norm or cand_norm in hint_norm:
        return True
    
    # Fuzzy matching for typos
    similarity = SequenceMatcher(None, hint_norm, cand_norm).ratio()
    if similarity >= fuzzy_threshold:
        return True
    
    # Check all names for the resolved ISO codes
    if hint_iso:
        hint_names = get_all_names(hint_iso)
        for name in hint_names:
            if name == cand_norm:
                return True
            if SequenceMatcher(None, name, cand_norm).ratio() >= fuzzy_threshold:
                return True
    
    if cand_iso:
        cand_names = get_all_names(cand_iso)
        for name in cand_names:
            if name == hint_norm:
                return True
            if SequenceMatcher(None, name, hint_norm).ratio() >= fuzzy_threshold:
                return True
    
    return False


def hint_matches_country(hint: str, country: str) -> bool:
    """Check if a location hint matches a country.
    
    Convenience function that handles:
    - Direct country names
    - City names that imply countries
    - Region names
    
    Args:
        hint: User's location hint
        country: Country from prediction
        
    Returns:
        True if hint implies the country
    """
    return countries_match(hint, country)


# Common city-to-country mappings for major cities
_CITY_TO_COUNTRY: dict[str, str] = {
    # Germany
    "berlin": "DE",
    "munich": "DE",
    "munchen": "DE",
    "hamburg": "DE",
    "frankfurt": "DE",
    "cologne": "DE",
    "koln": "DE",
    "stuttgart": "DE",
    "dusseldorf": "DE",
    "dresden": "DE",
    "leipzig": "DE",
    "hannover": "DE",
    "nuremberg": "DE",
    "nurnberg": "DE",
    "bremen": "DE",
    "dortmund": "DE",
    "essen": "DE",

    # France
    "paris": "FR",
    "lyon": "FR",
    "marseille": "FR",
    "toulouse": "FR",
    "nice": "FR",
    "bordeaux": "FR",
    "strasbourg": "FR",
    "lille": "FR",
    "nantes": "FR",
    
    # UK
    "london": "GB",
    "manchester": "GB",
    "birmingham": "GB",
    "liverpool": "GB",
    "edinburgh": "GB",
    "glasgow": "GB",
    "bristol": "GB",
    "oxford": "GB",
    "cambridge": "GB",
    
    # Italy
    "rome": "IT",
    "roma": "IT",
    "milan": "IT",
    "milano": "IT",
    "florence": "IT",
    "firenze": "IT",
    "venice": "IT",
    "venezia": "IT",
    "naples": "IT",
    "napoli": "IT",
    "turin": "IT",
    "torino": "IT",
    "bologna": "IT",
    
    # Spain
    "madrid": "ES",
    "barcelona": "ES",
    "valencia": "ES",
    "seville": "ES",
    "sevilla": "ES",
    "bilbao": "ES",
    "malaga": "ES",
    
    # Turkey
    "istanbul": "TR",
    "ankara": "TR",
    "izmir": "TR",
    "antalya": "TR",
    "bursa": "TR",
    
    # Japan
    "tokyo": "JP",
    "osaka": "JP",
    "kyoto": "JP",
    "yokohama": "JP",
    "nagoya": "JP",
    "sapporo": "JP",
    "fukuoka": "JP",
    
    # USA
    "new york": "US",
    "los angeles": "US",
    "chicago": "US",
    "houston": "US",
    "phoenix": "US",
    "philadelphia": "US",
    "san antonio": "US",
    "san diego": "US",
    "dallas": "US",
    "san jose": "US",
    "austin": "US",
    "seattle": "US",
    "denver": "US",
    "boston": "US",
    "miami": "US",
    "atlanta": "US",
    "san francisco": "US",
    "washington": "US",
    "washington dc": "US",
    "las vegas": "US",
    "portland": "US",
    "detroit": "US",
    
    # Netherlands
    "amsterdam": "NL",
    "rotterdam": "NL",
    "the hague": "NL",
    "den haag": "NL",
    "utrecht": "NL",
    
    # Belgium
    "brussels": "BE",
    "bruxelles": "BE",
    "antwerp": "BE",
    "gent": "BE",
    "liege": "BE",
    
    # Austria
    "vienna": "AT",
    "wien": "AT",
    "graz": "AT",
    "salzburg": "AT",
    
    # Switzerland
    "zurich": "CH",
    "geneva": "CH",
    "basel": "CH",
    "bern": "CH",
    "lausanne": "CH",
    
    # Poland
    "warsaw": "PL",
    "warszawa": "PL",
    "krakow": "PL",
    "lodz": "PL",
    "wroclaw": "PL",
    "poznan": "PL",
    "gdansk": "PL",
    
    # Czech Republic
    "prague": "CZ",
    "praha": "CZ",
    "brno": "CZ",
    
    # Sweden
    "stockholm": "SE",
    "gothenburg": "SE",
    "malmo": "SE",
    
    # Norway
    "oslo": "NO",
    "bergen": "NO",
    "trondheim": "NO",
    
    # Denmark
    "copenhagen": "DK",
    "kobenhavn": "DK",
    "aarhus": "DK",
    
    # Finland
    "helsinki": "FI",
    "tampere": "FI",
    "turku": "FI",
    
    # Russia
    "moscow": "RU",
    "saint petersburg": "RU",
    "st petersburg": "RU",
    "novosibirsk": "RU",
    
    # China
    "beijing": "CN",
    "shanghai": "CN",
    "guangzhou": "CN",
    "shenzhen": "CN",
    "chengdu": "CN",
    "hong kong": "HK",  # Special administrative region
    
    # South Korea
    "seoul": "KR",
    "busan": "KR",
    "incheon": "KR",
    
    # India
    "mumbai": "IN",
    "delhi": "IN",
    "bangalore": "IN",
    "chennai": "IN",
    "kolkata": "IN",
    "hyderabad": "IN",
    
    # Brazil
    "sao paulo": "BR",
    "são paulo": "BR",
    "rio de janeiro": "BR",
    "rio": "BR",
    "brasilia": "BR",
    "salvador": "BR",
    
    # Argentina
    "buenos aires": "AR",
    "cordoba": "AR",
    "rosario": "AR",
    "mendoza": "AR",
    
    # Chile
    "santiago": "CL",
    "valparaiso": "CL",
    
    # Colombia
    "bogota": "CO",
    "medellin": "CO",
    "cartagena": "CO",
    "cali": "CO",
    
    # Peru
    "lima": "PE",
    "cusco": "PE",
    
    # Mexico
    "mexico city": "MX",
    "guadalajara": "MX",
    "monterrey": "MX",
    
    # Canada
    "toronto": "CA",
    "montreal": "CA",
    "vancouver": "CA",
    "calgary": "CA",
    "ottawa": "CA",
    
    # Australia
    "sydney": "AU",
    "melbourne": "AU",
    "brisbane": "AU",
    "perth": "AU",
    
    # UAE
    "dubai": "AE",
    "abu dhabi": "AE",
    
    # Singapore
    "singapore": "SG",
    
    # Thailand
    "bangkok": "TH",
    
    # Vietnam
    "ho chi minh": "VN",
    "hanoi": "VN",
    
    # Indonesia
    "jakarta": "ID",
    
    # Malaysia
    "kuala lumpur": "MY",
    
    # Philippines
    "manila": "PH",
    
    # Egypt
    "cairo": "EG",
    "alexandria": "EG",
    
    # South Africa
    "johannesburg": "ZA",
    "cape town": "ZA",
    "durban": "ZA",
    
    # Israel
    "tel aviv": "IL",
    "jerusalem": "IL",
    
    # Greece
    "athens": "GR",
    "thessaloniki": "GR",
    
    # Portugal
    "lisbon": "PT",
    "lisboa": "PT",
    "porto": "PT",
    
    # Hungary
    "budapest": "HU",
    
    # Romania
    "bucharest": "RO",
    
    # Ireland
    "dublin": "IE",
    "cork": "IE",
    
    # Morocco
    "casablanca": "MA",
    "marrakech": "MA",
    "rabat": "MA",
}


def extract_country_from_location(location: str) -> Optional[str]:
    """Extract ISO country code from any location string.
    
    Tries multiple strategies:
    1. Direct country name lookup
    2. City name lookup
    3. Substring matching
    
    Args:
        location: Location string (city, country, region, etc.)
        
    Returns:
        ISO country code or None
    """
    if not location:
        return None
    
    normalized = normalize_country_name(location)
    if not normalized:
        return None
    
    # Direct country lookup
    iso = get_iso_code(normalized)
    if iso:
        return iso
    
    # City lookup
    iso = _CITY_TO_COUNTRY.get(normalized)
    if iso:
        return iso
    
    # Check for city in string
    for city, country_iso in _CITY_TO_COUNTRY.items():
        if city in normalized:
            return country_iso
    
    # Check for country name as substring
    for name, iso in _COUNTRY_ALIASES.items():
        if name in normalized:
            return iso
    
    return None
