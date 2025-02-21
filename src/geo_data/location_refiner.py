from typing import Dict, List, Optional
import requests
from dataclasses import dataclass
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from .search_utils import LocationSearcher


@dataclass
class RefinedLocation:
    name: str
    address: str
    lat: float
    lon: float
    confidence: float
    distance_from_landmark: Optional[float] = None
    source: str = "geocoding"


class LocationRefiner:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="geolocation_refiner")
        self.searcher = LocationSearcher()

    def refine_location(self, city_info: Dict, analysis: Dict) -> List[RefinedLocation]:
        """Refine a city-level location into specific address suggestions"""
        suggestions = []

        # Extract key information from analysis
        landmarks = self._extract_landmarks(analysis)
        business_names = self._extract_businesses(analysis)

        # Search for businesses and landmarks using both geocoding and web search
        for name in business_names + landmarks:
            # Try geocoding first
            geocoded = self._search_business(name, city_info["name"])
            suggestions.extend(geocoded)

            # Then try web search
            web_results = self.searcher.search_location_info(name, city_info["name"])
            for result in web_results:
                for addr in result["addresses"]:
                    try:
                        # Try to geocode found addresses
                        location = self.geolocator.geocode(f"{addr}, {city_info['name']}")
                        if location:
                            suggestions.append(
                                RefinedLocation(
                                    name=f"{name} ({result['title']})",
                                    address=addr,
                                    lat=location.latitude,
                                    lon=location.longitude,
                                    confidence=0.85,
                                    source="web_search",
                                )
                            )
                    except Exception as e:
                        print(f"Error geocoding web result: {e}")

        # Filter and rank suggestions
        ranked_suggestions = self._rank_suggestions(suggestions, city_info, analysis)

        return ranked_suggestions[:3]

    def _extract_landmarks(self, analysis: Dict) -> List[str]:
        """Extract landmark information from analysis"""
        landmarks = []

        # Get explicit landmarks
        if "landmarks" in analysis.get("features", {}):
            landmarks.extend([l.strip("*: ") for l in analysis["features"]["landmarks"]])

        # Look for landmarks in text
        text_info = analysis.get("features", {}).get("extracted_text", {})
        if "informational" in text_info:
            landmarks.extend([t.strip("*: ") for t in text_info["informational"] if any(k in t.lower() for k in ["building", "church", "monument", "station"])])

        return landmarks

    def _extract_businesses(self, analysis: Dict) -> List[str]:
        """Extract business names from analysis"""
        businesses = []

        text_info = analysis.get("features", {}).get("extracted_text", {})
        if "business_names" in text_info:
            businesses.extend([b.strip("*: ") for b in text_info["business_names"]])

        # Also check informational text for business names
        if "informational" in text_info:
            businesses.extend([t.strip("*: ") for t in text_info["informational"] if any(k in t.lower() for k in ["shop", "store", "market", "restaurant"])])

        return businesses

    def _search_business(self, business_name: str, city: str) -> List[RefinedLocation]:
        """Search for specific business locations in a city"""
        query = f"{business_name}, {city}"

        try:
            locations = self.geolocator.geocode(query, exactly_one=False, limit=3, addressdetails=True)

            if not locations:
                return []

            return [
                RefinedLocation(
                    name=f"{business_name} ({loc.raw['address'].get('road', 'Unknown Street')})",
                    address=loc.address,
                    lat=loc.latitude,
                    lon=loc.longitude,
                    confidence=0.9 if business_name.lower() in loc.address.lower() else 0.7,
                )
                for loc in locations
            ]

        except Exception as e:
            print(f"Error searching for business {business_name}: {e}")
            return []

    def _rank_suggestions(self, suggestions: List[RefinedLocation], city_info: Dict, analysis: Dict) -> List[RefinedLocation]:
        """Rank and filter location suggestions"""

        # Remove duplicates
        unique_suggestions = []
        seen_addresses = set()

        for suggestion in suggestions:
            if suggestion.address not in seen_addresses:
                seen_addresses.add(suggestion.address)
                unique_suggestions.append(suggestion)

        # Score each suggestion
        scored_suggestions = []
        for suggestion in unique_suggestions:
            score = suggestion.confidence

            # Adjust score based on distance from city center
            city_center = (city_info["lat"], city_info["lon"])
            suggestion_coords = (suggestion.lat, suggestion.lon)
            distance = geodesic(city_center, suggestion_coords).kilometers

            if distance < 2:
                score += 0.1
            elif distance > 10:
                score -= 0.1

            # Adjust score based on matching features
            if analysis.get("features", {}).get("architecture_style"):
                # Implementation: Check if architecture style matches area
                pass

            suggestion.confidence = min(0.99, score)
            scored_suggestions.append(suggestion)

        return sorted(scored_suggestions, key=lambda x: x.confidence, reverse=True)
