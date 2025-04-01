import overpy
import requests
from typing import Dict, List, Optional
import json
from .enhanced_search import EnhancedLocationSearch
import re
import googlemaps
from src.config import CONFIG
from math import radians, cos
from .browser_search import BrowserSearchAdapter


class GeoDataInterface:
    def __init__(self, geonames_username: str = None):
        self.osm_api = overpy.Overpass()
        self.geonames_username = geonames_username
        self.enhanced_search = EnhancedLocationSearch(geonames_username)
        # Initialize Google Maps client
        self.gmaps = googlemaps.Client(key=CONFIG.GOOGLE_MAPS_API_KEY) if CONFIG.GOOGLE_MAPS_API_KEY else None
        self.browser_search = BrowserSearchAdapter() if CONFIG.USE_BROWSER else None

    def _preprocess_features(self, features: Dict) -> Dict:
        """Preprocess and clean up features with improved entity recognition"""
        processed = {
            "landmarks": [],
            "business_names": [],
            "street_signs": [],
            "building_info": [],
            "architecture_style": features.get("architecture_style", ""),
            "geographic_features": features.get("geographic_features", []),
            "entity_locations": {},  # Map entities to potential locations
        }

        # Process landmarks and extract business names with better classification
        for landmark in features.get("landmarks", []):
            # Clean up the text
            clean_text = landmark.split("**")[0].strip().strip("\"'")

            # Use entity type classification if available
            entity_type = features.get("entity_types", {}).get(clean_text, self._classify_entity(clean_text))

            if entity_type == "business":
                processed["business_names"].append(clean_text)
            elif entity_type == "street":
                processed["street_signs"].append(clean_text)
            elif entity_type == "building":
                processed["building_info"].append(clean_text)
            else:
                processed["landmarks"].append(clean_text)

        # Add existing categorized entities
        for category in ["business_names", "street_signs", "building_info"]:
            processed[category].extend(features.get("extracted_text", {}).get(category, []))

        # Extract potential location context from entities
        for entity in processed["business_names"] + processed["landmarks"]:
            location_context = self._extract_location_context(entity)
            if location_context:
                processed["entity_locations"][entity] = location_context

        # Remove duplicates while preserving order
        for key in ["landmarks", "business_names", "street_signs", "building_info"]:
            processed[key] = list(dict.fromkeys(processed[key]))

        return processed

    def _classify_entity(self, text: str) -> str:
        """Classify an entity based on text patterns"""
        text_lower = text.lower()

        # Business indicators
        business_indicators = [
            "gmbh",
            "ltd",
            "inc",
            "llc",
            "co.",
            "kg",
            "ag",
            "restaurant",
            "café",
            "cafe",
            "hotel",
            "shop",
            "store",
            "markt",
            "market",
            "supermarket",
            "biomarkt",
            "bakery",
            "bäckerei",
            "apotheke",
            "pharmacy",
            "bank",
        ]

        # Street indicators
        street_indicators = [
            "straße",
            "strasse",
            "str.",
            "street",
            "avenue",
            "ave",
            "road",
            "rd",
            "boulevard",
            "blvd",
            "lane",
            "ln",
            "way",
            "allee",
            "platz",
            "square",
        ]

        # Building indicators
        building_indicators = ["building", "gebäude", "haus", "house", "apartment", "apt", "suite", "floor", "etage", "no.", "nr.", "number", "hausnummer"]

        if any(indicator in text_lower for indicator in business_indicators):
            return "business"
        elif any(indicator in text_lower for indicator in street_indicators):
            return "street"
        elif any(indicator in text_lower for indicator in building_indicators):
            return "building"
        else:
            return "landmark"

    def _extract_location_context(self, text: str) -> Optional[str]:
        """Extract potential location context from entity text"""
        # Look for "in [Location]" pattern
        location_match = re.search(r"(?:in|at|near|of)\s+([A-Z][a-zA-Z\s]+)(?:,|\.|$)", text)
        if location_match:
            return location_match.group(1).strip()

        # Look for city names after commas
        comma_match = re.search(r",\s*([A-Z][a-zA-Z\s]+)(?:,|\.|$)", text)
        if comma_match:
            return comma_match.group(1).strip()

        return None

    async def search_location_candidates(self, features: Dict, location_hint: str = None, metadata: Dict = None) -> List[Dict]:
        """Enhanced location search using multiple sources"""
        # Preprocess features first
        processed_features = self._preprocess_features(features)
        print("\n=== Processed Features ===")
        print(f"Business Names: {processed_features['business_names']}")
        print(f"Landmarks: {processed_features['landmarks']}")
        print(f"Street Signs: {processed_features['street_signs']}")
        print("========================\n")

        # Convert location hint to coordinates and region info
        initial_coords = None
        region_info = None
        if location_hint:
            # Try to extract coordinates from the hint
            coords_match = re.search(r"(\d+\.?\d*),\s*(\d+\.?\d*)", location_hint)
            if coords_match:
                lat, lon = map(float, coords_match.groups())
                # Add a default bounding box of roughly 5km around the point
                initial_coords = {
                    "lat": lat,
                    "lon": lon,
                    "confidence": 0.9 if "confidence" in location_hint.lower() else 0.7,
                    "bbox": {
                        "north": lat + 0.045,  # Approximately 5km
                        "south": lat - 0.045,
                        "east": lon + 0.045 / cos(radians(lat)),
                        "west": lon - 0.045 / cos(radians(lat)),
                    },
                }
            else:
                # Try to geocode the location name
                try:
                    # Use Google Maps geocoding
                    if self.gmaps:
                        geocode_result = self.gmaps.geocode(location_hint)
                        if geocode_result:
                            location = geocode_result[0]
                            lat = location["geometry"]["location"]["lat"]
                            lon = location["geometry"]["location"]["lng"]
                            bounds = location["geometry"].get("bounds") or location["geometry"].get("viewport")

                            initial_coords = {
                                "lat": lat,
                                "lon": lon,
                                "confidence": 0.9,  # Higher confidence for geocoded results
                                "bbox": (
                                    {
                                        "north": bounds["northeast"]["lat"],
                                        "south": bounds["southwest"]["lat"],
                                        "east": bounds["northeast"]["lng"],
                                        "west": bounds["southwest"]["lng"],
                                    }
                                    if bounds
                                    else {
                                        "north": lat + 0.045,
                                        "south": lat - 0.045,
                                        "east": lon + 0.045 / cos(radians(lat)),
                                        "west": lon - 0.045 / cos(radians(lat)),
                                    }
                                ),
                            }

                            # Store region info for filtering
                            region_info = {
                                "city": next((comp["long_name"] for comp in location["address_components"] if "locality" in comp["types"]), None),
                                "state": next(
                                    (comp["long_name"] for comp in location["address_components"] if "administrative_area_level_1" in comp["types"]), None
                                ),
                                "country": next((comp["long_name"] for comp in location["address_components"] if "country" in comp["types"]), None),
                            }
                except Exception as e:
                    print(f"Error geocoding location hint: {e}")

        # Add location hint to features for better context
        if region_info:
            processed_features["location_hint"] = region_info

        # Get candidates from enhanced search
        enhanced_candidates = await self.enhanced_search.find_location_candidates(processed_features, metadata=metadata, location_hint=location_hint)

        # Get traditional OSM candidates (limit to 5 for performance)
        osm_candidates = []
        if processed_features["landmarks"] or processed_features["business_names"]:
            osm_candidates = self._search_osm(processed_features, initial_coords)[:5]

        # Get business-specific candidates
        business_candidates = []
        if processed_features["business_names"]:
            business_candidates = self._search_businesses(processed_features, initial_coords)[:3]

        # Get transport infrastructure candidates if relevant features exist
        transport_candidates = []
        if any("tram" in f.lower() or "station" in f.lower() for f in processed_features.get("geographic_features", [])):
            transport_candidates = self._search_transport_infrastructure(processed_features, initial_coords)[:3]

        # Get cultural landmark candidates
        cultural_candidates = []
        if processed_features["landmarks"] or processed_features["architecture_style"]:
            cultural_candidates = self._search_cultural_landmarks(processed_features, initial_coords)[:3]

        # Combine all candidates
        all_candidates = (
            enhanced_candidates  # Enhanced search results (Google + OSM)
            + osm_candidates  # Basic OSM search
            + business_candidates  # Business-specific search
            + transport_candidates  # Transport infrastructure
            + cultural_candidates  # Cultural landmarks
        )

        # Filter and boost confidence for candidates within the hinted region
        if region_info:
            for candidate in all_candidates:
                # Check if candidate is within the hinted region
                is_in_region = self._is_in_region(candidate, region_info, initial_coords)
                if is_in_region:
                    # Smaller boost for matches within the region
                    candidate["confidence"] = min(1.0, candidate["confidence"] + 0.1)
                else:
                    # Stronger penalty for matches outside the region
                    candidate["confidence"] *= 0.3

        # Deduplicate and rank candidates
        ranked_candidates = self._rank_candidates(all_candidates, processed_features, initial_coords)

        # If no results within region, try Google Maps
        if not ranked_candidates and self.gmaps and region_info:
            google_candidates = self._search_google_maps(processed_features, initial_coords)
            for candidate in google_candidates:
                if self._is_in_region(candidate, region_info, initial_coords):
                    ranked_candidates.append(candidate)

        search_queries = []
        location_contexts = []

        # Get location contexts
        if location_hint:
            location_contexts.append(location_hint)
        if processed_features.get("entity_locations"):
            location_contexts.extend(processed_features["entity_locations"].values())

        # Add business names with location context
        if processed_features.get("business_names"):
            for business in processed_features["business_names"]:
                for context in location_contexts:
                    search_queries.append(f"{business} {context}")

        # Add street signs with location context
        if processed_features.get("street_signs"):
            for street in processed_features["street_signs"]:
                for context in location_contexts:
                    search_queries.append(f"{street} {context}")

        # Add landmarks with location context
        if processed_features.get("landmarks"):
            for landmark in processed_features["landmarks"]:
                for context in location_contexts:
                    search_queries.append(f"{landmark} {context}")

        if not search_queries and location_hint:
            search_queries.append(location_hint)

        # Initialize candidates list
        candidates = []

        # If browser search is enabled, add browser results
        if CONFIG.USE_BROWSER and self.browser_search:
            print("\n=== Searching with browser-use ===")
            for query in search_queries[:3]:  # Limit to first 3 queries
                print(f"Searching for: {query}")
                browser_results = await self.browser_search.search_locations(query)
                candidates.extend(browser_results)
                print(f"Found {len(browser_results)} browser results for query: {query}")

        # Print summary for debugging
        print("\n=== Location Candidates ===")
        print(f"Enhanced Search: {len(enhanced_candidates)} results")
        print(f"OSM Basic: {len(osm_candidates)} results")
        print(f"Business: {len(business_candidates)} results")
        print(f"Transport: {len(transport_candidates)} results")
        print(f"Cultural: {len(cultural_candidates)} results")
        print(f"Total after ranking: {len(ranked_candidates)} results")
        print("=======================\n")

        return ranked_candidates[:10]  # Return top 10 candidates

    def _is_in_region(self, candidate: Dict, region_info: Dict, initial_coords: Optional[Dict]) -> bool:
        """Check if a candidate is within the hinted region"""
        # If we have coordinates and bounding box, check that first
        if initial_coords and initial_coords.get("bbox"):
            bbox = initial_coords["bbox"]
            if bbox["south"] <= candidate["lat"] <= bbox["north"] and bbox["west"] <= candidate["lon"] <= bbox["east"]:
                return True

        # If we have region info, check against that
        if region_info:
            # Try to geocode the candidate's location
            if self.gmaps:
                try:
                    reverse_result = self.gmaps.reverse_geocode((candidate["lat"], candidate["lon"]))
                    if reverse_result:
                        location = reverse_result[0]
                        for component in location["address_components"]:
                            # Check city match
                            if region_info["city"] and "locality" in component["types"]:
                                if component["long_name"] == region_info["city"]:
                                    return True
                            # Check state match
                            if region_info["state"] and "administrative_area_level_1" in component["types"]:
                                if component["long_name"] == region_info["state"]:
                                    return True
                            # Check country match
                            if region_info["country"] and "country" in component["types"]:
                                if component["long_name"] == region_info["country"]:
                                    return True
                except Exception as e:
                    print(f"Error in reverse geocoding: {e}")
                    return True  # Default to True to avoid false negatives

        return False

    def _get_location_coordinates(self, location: str) -> Optional[Dict]:
        """Get coordinates for a location string using GeoNames"""
        try:
            url = "http://api.geonames.org/searchJSON"
            params = {"q": location, "maxRows": 1, "username": self.geonames_username, "style": "FULL"}

            response = requests.get(url, params=params)
            data = response.json()

            if data.get("geonames"):
                result = data["geonames"][0]
                return {
                    "lat": float(result["lat"]),
                    "lon": float(result["lng"]),
                    "bbox": {
                        "north": float(result.get("bbox", {}).get("north", result["lat"]) or result["lat"]),
                        "south": float(result.get("bbox", {}).get("south", result["lat"]) or result["lat"]),
                        "east": float(result.get("bbox", {}).get("east", result["lng"]) or result["lng"]),
                        "west": float(result.get("bbox", {}).get("west", result["lng"]) or result["lng"]),
                    },
                }
            return None
        except Exception as e:
            print(f"Error getting location coordinates: {e}")
            return None

    def _search_osm(self, features: Dict, initial_coords: Optional[Dict] = None) -> List[Dict]:
        """Search OpenStreetMap for matching locations"""
        query = self._build_osm_query(features, initial_coords)
        try:
            results = self.osm_api.query(query)
            candidates = []

            for result in results.ways:
                # Calculate base confidence
                confidence = self._calculate_confidence(features, result)

                # Boost confidence if within initial location area
                if initial_coords:
                    if (
                        initial_coords["bbox"]["south"] <= float(result.nodes[0].lat) <= initial_coords["bbox"]["north"]
                        and initial_coords["bbox"]["west"] <= float(result.nodes[0].lon) <= initial_coords["bbox"]["east"]
                    ):
                        confidence = min(1.0, confidence + 0.2)

                candidate = {
                    "source": "osm",
                    "name": result.tags.get("name", "Unknown"),
                    "lat": result.nodes[0].lat,
                    "lon": result.nodes[0].lon,
                    "type": result.tags.get("amenity", "unknown"),
                    "confidence": confidence,
                    "metadata": {"osm_id": result.id, "tags": result.tags},
                }
                candidates.append(candidate)

            return candidates
        except Exception as e:
            print(f"OSM search error: {e}")
            return []

    def _search_geonames(self, features: Dict, initial_coords: Optional[Dict] = None) -> List[Dict]:
        """Search GeoNames database"""
        if not self.geonames_username:
            return []

        try:
            landmarks = features.get("landmarks", [])
            results = []

            for landmark in landmarks:
                params = {"q": landmark, "maxRows": 5, "username": self.geonames_username, "style": "FULL"}

                # Add location context if available
                if initial_coords:
                    params.update(
                        {
                            "north": initial_coords["bbox"]["north"],
                            "south": initial_coords["bbox"]["south"],
                            "east": initial_coords["bbox"]["east"],
                            "west": initial_coords["bbox"]["west"],
                        }
                    )

                response = requests.get("http://api.geonames.org/searchJSON", params=params)
                data = response.json()

                for result in data.get("geonames", []):
                    confidence = self._calculate_geonames_confidence(features, result)

                    # Boost confidence if initial location was provided
                    if initial_coords:
                        confidence = min(1.0, confidence + 0.2)

                    candidate = {
                        "source": "geonames",
                        "name": result["name"],
                        "lat": float(result["lat"]),
                        "lon": float(result["lng"]),
                        "type": result.get("fcode", "unknown"),
                        "confidence": confidence,
                        "metadata": result,
                    }
                    results.append(candidate)

            return results
        except Exception as e:
            print(f"GeoNames search error: {e}")
            return []

    def _search_wikidata(self, features: Dict, initial_coords: Optional[Dict] = None) -> List[Dict]:
        """Search Wikidata for landmarks and locations"""
        try:
            landmarks = features.get("landmarks", [])
            results = []

            for landmark in landmarks:
                # Build SPARQL query with location constraints if available
                location_filter = ""
                if initial_coords:
                    location_filter = f"""
                    FILTER(
                        ?lat >= {initial_coords["bbox"]["south"]} &&
                        ?lat <= {initial_coords["bbox"]["north"]} &&
                        ?lon >= {initial_coords["bbox"]["west"]} &&
                        ?lon <= {initial_coords["bbox"]["east"]}
                    )
                    """

                query = f"""
                SELECT ?item ?itemLabel ?lat ?lon ?type ?typeLabel WHERE {{
                  SERVICE wikibase:mbox {{
                    bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en".
                  }}
                  ?item rdfs:label ?label.
                  ?item wdt:P625 ?coords.
                  ?item wdt:P31 ?type.
                  BIND(geof:latitude(?coords) AS ?lat)
                  BIND(geof:longitude(?coords) AS ?lon)
                  FILTER(CONTAINS(LCASE(?label), LCASE("{landmark}")))
                  {location_filter}
                }}
                LIMIT 5
                """

                url = "https://query.wikidata.org/sparql"
                headers = {"Accept": "application/json"}
                params = {"query": query, "format": "json"}

                response = requests.get(url, headers=headers, params=params)
                data = response.json()

                for result in data.get("results", {}).get("bindings", []):
                    confidence = 0.7  # Base confidence for Wikidata matches

                    # Boost confidence if initial location was provided
                    if initial_coords:
                        confidence = min(1.0, confidence + 0.2)

                    candidate = {
                        "source": "wikidata",
                        "name": result["itemLabel"]["value"],
                        "lat": float(result["lat"]["value"]),
                        "lon": float(result["lon"]["value"]),
                        "type": result["typeLabel"]["value"],
                        "confidence": confidence,
                        "metadata": {"wikidata_id": result["item"]["value"].split("/")[-1], "type_id": result["type"]["value"]},
                    }
                    results.append(candidate)

            return results
        except Exception as e:
            print(f"Wikidata search error: {e}")
            return []

    def _search_businesses(self, features: Dict, initial_coords: Optional[Dict] = None) -> List[Dict]:
        """Search for businesses using Google Maps Places API"""
        if not self.gmaps:
            return []

        results = []
        for business in features.get("business_names", []):
            try:
                # Use nearby search if we have coordinates
                if initial_coords:
                    places = self.gmaps.places_nearby(location=(initial_coords["lat"], initial_coords["lon"]), radius=5000, keyword=business)  # 5km radius
                else:
                    # Use text search otherwise
                    places = self.gmaps.places(query=business, type="establishment")

                if not places.get("results"):
                    continue

                for place in places["results"][:3]:  # Limit to top 3 matches
                    results.append(
                        {
                            "name": place["name"],
                            "lat": place["geometry"]["location"]["lat"],
                            "lon": place["geometry"]["location"]["lng"],
                            "confidence": 0.8 if place.get("rating") else 0.6,
                            "source": "google_maps",
                            "type": "business",
                            "metadata": {
                                "place_id": place["place_id"],
                                "address": place.get("formatted_address"),
                                "types": place.get("types", []),
                                "rating": place.get("rating"),
                            },
                        }
                    )

            except Exception as e:
                print(f"Error searching Google Maps: {e}")
                continue

        return results

    def _search_transport_infrastructure(self, features: Dict, initial_coords: Optional[Dict] = None) -> List[Dict]:
        """Search for transportation infrastructure like tram lines, stations, etc."""
        results = []

        try:
            # Build area bounds
            area_bounds = (
                f"({initial_coords['bbox']['south']},{initial_coords['bbox']['west']}," f"{initial_coords['bbox']['north']},{initial_coords['bbox']['east']})"
                if initial_coords
                else "(-90,-180,90,180)"
            )

            # Only search for transport features mentioned in the image
            transport_types = []
            for feature in features.get("geographic_features", []):
                feature_lower = str(feature).lower()
                if "tram" in feature_lower:
                    transport_types.extend(["tram", "tram_stop"])
                if "bus" in feature_lower:
                    transport_types.extend(["bus_stop", "bus_station"])
                if "train" in feature_lower or "railway" in feature_lower:
                    transport_types.extend(["station", "railway"])
                if "subway" in feature_lower or "metro" in feature_lower:
                    transport_types.extend(["subway", "subway_entrance"])

            if not transport_types:
                return []

            # Build optimized query with shorter timeout
            query = f"""
            [out:json][timeout:10];
            (
              // Transport nodes
              node["railway"~"{"|".join(transport_types)}",i]{area_bounds};
              node["public_transport"~"{"|".join(transport_types)}",i]{area_bounds};
              
              // Transport ways (for routes)
              way["railway"~"{"|".join(transport_types)}",i]{area_bounds};
            );
            out body;
            >;
            out skel qt;
            """

            response = self.osm_api.query(query)

            for element in response.nodes:
                transport_type = element.tags.get("railway") or element.tags.get("public_transport") or "transport"

                confidence = 0.7  # Base confidence

                # Boost confidence for exact matches
                if any(t_type in element.tags.get("railway", "").lower() for t_type in transport_types):
                    confidence = min(0.9, confidence + 0.2)

                results.append(
                    {
                        "source": "osm_transport",
                        "name": element.tags.get("name", f"{transport_type.title()} Stop"),
                        "lat": float(element.lat),
                        "lon": float(element.lon),
                        "type": "transport",
                        "confidence": confidence,
                        "metadata": {"osm_id": element.id, "tags": dict(element.tags), "transport_type": transport_type},
                    }
                )

            return results
        except Exception as e:
            print(f"Error searching transport infrastructure: {e}")
            return []

    def _search_cultural_landmarks(self, features: Dict, initial_coords: Optional[Dict] = None) -> List[Dict]:
        """Search for cultural landmarks and points of interest"""
        results = []

        try:
            # Build area bounds
            area_bounds = (
                f"({initial_coords['bbox']['south']},{initial_coords['bbox']['west']}," f"{initial_coords['bbox']['north']},{initial_coords['bbox']['east']})"
                if initial_coords
                else "(-90,-180,90,180)"
            )

            # Build style filter safely
            style_filter = ""
            if features.get("architecture_style"):
                safe_style = features["architecture_style"].replace('"', '\\"')
                style_filter = f'["architecture"~"{safe_style}",i]'

            # Build query with separate filters for historic and tourism
            query = f"""
            [out:json][timeout:15];
            (
              // Historic sites
              way["historic"]{style_filter}{area_bounds};
              node["historic"]{style_filter}{area_bounds};
              
              // Tourism landmarks
              way["tourism"]["historic"]{style_filter}{area_bounds};
              node["tourism"]["historic"]{style_filter}{area_bounds};
            );
            out body;
            >;
            out skel qt;
            """

            response = self.osm_api.query(query)

            for element in response.nodes:
                confidence = 0.6  # Base confidence

                # Boost confidence based on matches
                if features.get("architecture_style"):
                    if features["architecture_style"].lower() in element.tags.get("architecture", "").lower():
                        confidence = min(0.9, confidence + 0.3)

                if any(landmark.lower() in element.tags.get("name", "").lower() for landmark in features.get("landmarks", [])):
                    confidence = min(0.9, confidence + 0.2)

                results.append(
                    {
                        "source": "osm_cultural",
                        "name": element.tags.get("name", "Cultural Site"),
                        "lat": float(element.lat),
                        "lon": float(element.lon),
                        "type": "cultural",
                        "confidence": confidence,
                        "metadata": {
                            "osm_id": element.id,
                            "tags": dict(element.tags),
                            "historic_type": element.tags.get("historic"),
                            "tourism_type": element.tags.get("tourism"),
                        },
                    }
                )

            return results
        except Exception as e:
            print(f"Error searching cultural landmarks: {e}")
            return []

    def _rank_candidates(self, candidates: List[Dict], features: Dict, initial_coords: Optional[Dict] = None) -> List[Dict]:
        """Rank candidates based on multiple factors"""
        for candidate in candidates:
            score = candidate.get("confidence", 0.5)  # Start with base confidence

            # Boost score based on feature matches
            if features.get("architecture_style") and candidate.get("metadata", {}).get("tags", {}).get("architecture") == features["architecture_style"]:
                score += 0.2

            # Boost score for business name matches
            if candidate["type"] == "business" and any(
                business.lower() in candidate["name"].lower() for business in features.get("extracted_text", {}).get("business_names", [])
            ):
                score += 0.3

            # Boost score for transport infrastructure matches
            if candidate["type"] == "transport" and any("tram" in feature.lower() for feature in features.get("geographic_features", [])):
                score += 0.2

            # Boost score for cultural landmark matches
            if candidate["type"] == "cultural" and any(landmark.lower() in candidate["name"].lower() for landmark in features.get("landmarks", [])):
                score += 0.2

            # Boost score for proximity to initial location
            if initial_coords:
                distance = self._calculate_distance(
                    (float(candidate["lat"]), float(candidate["lon"])), (float(initial_coords["lat"]), float(initial_coords["lon"]))
                )
                if distance < 1000:  # Within 1km
                    score += 0.3
                elif distance < 5000:  # Within 5km
                    score += 0.1

            candidate["confidence"] = min(1.0, score)

        # Sort by confidence
        return sorted(candidates, key=lambda x: x["confidence"], reverse=True)

    def _calculate_distance(self, point1: tuple, point2: tuple) -> float:
        """Calculate distance between two points in meters"""
        from math import sin, cos, sqrt, atan2, radians

        R = 6371000  # Earth radius in meters

        lat1, lon1 = map(radians, point1)
        lat2, lon2 = map(radians, point2)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = R * c

        return distance

    def _deduplicate_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """Remove duplicate locations based on proximity and name similarity"""
        unique_candidates = []
        seen_locations = set()

        for candidate in candidates:
            location_key = f"{round(candidate['lat'], 4)},{round(candidate['lon'], 4)}"
            if location_key not in seen_locations:
                seen_locations.add(location_key)
                unique_candidates.append(candidate)

        return unique_candidates

    def _calculate_confidence(self, features: Dict, location: Dict) -> float:
        """Calculate confidence score for OSM matches"""
        score = 0.5  # Base score

        # Add scoring logic based on feature matches
        # Example: Check if architectural style matches
        if features.get("architecture_style") and location.tags.get("architecture") == features["architecture_style"]:
            score += 0.2

        return min(score, 1.0)

    def _calculate_geonames_confidence(self, features: Dict, location: Dict) -> float:
        """Calculate confidence score for GeoNames matches"""
        score = 0.5  # Base score

        # Add scoring logic based on feature matches
        # Example: Check if feature class matches expected type
        if location.get("fcl") in ["P", "S", "L"]:  # Population center, Spot, or Landmark
            score += 0.2

        return min(score, 1.0)

    def _build_osm_query(self, features: Dict, initial_coords: Optional[Dict] = None) -> str:
        """Build optimized OSM query with location constraints"""
        # Define search radius based on confidence and location type
        if initial_coords:
            confidence = initial_coords.get("confidence", 0.5)
            # If we have a street address, use a very small radius (500m)
            if any(
                "straße" in str(f).lower() or "strasse" in str(f).lower() or "street" in str(f).lower()
                for f in features.get("extracted_text", {}).get("street_signs", [])
            ):
                radius_km = 0.5
            # If we have high confidence (>0.8), use smaller radius (2km)
            elif confidence > 0.8:
                radius_km = 2
            # For medium confidence (0.5-0.8), use medium radius (5km)
            elif confidence > 0.5:
                radius_km = 5
            # For low confidence (<0.5), use larger radius (10km)
            else:
                radius_km = 10

            # Calculate bounding box
            from math import radians, cos, sin, asin, sqrt

            lat = float(initial_coords.get("lat", 0))
            lon = float(initial_coords.get("lon", 0))

            # Rough approximation of km to degrees
            # 1 degree latitude = ~111km
            # 1 degree longitude = ~111km * cos(latitude)
            lat_offset = radius_km / 111.0
            lon_offset = radius_km / (111.0 * cos(radians(lat)))

            area_bounds = f"({lat - lat_offset},{lon - lon_offset},{lat + lat_offset},{lon + lon_offset})"
        else:
            area_bounds = "(47.0,5.0,55.0,15.0)"  # Default to rough Germany bounds if no coords

        # Build query parts for each feature type
        query_parts = []

        # Add street name queries first (highest priority)
        for street in features.get("extracted_text", {}).get("street_signs", []):
            safe_name = street.replace('"', '\\"')
            # Search for the exact street
            query_parts.append(f'way["highway"]["name"~"^{safe_name}$",i]{area_bounds};')
            # Also search for streets containing this name
            query_parts.append(f'way["highway"]["name"~"{safe_name}",i]{area_bounds};')

        # Add business name queries
        for business in features.get("extracted_text", {}).get("business_names", []):
            safe_name = business.replace('"', '\\"')
            # Search for exact business name match
            query_parts.append(f'node["name"~"^{safe_name}$",i]{area_bounds};')
            query_parts.append(f'way["name"~"^{safe_name}$",i]{area_bounds};')
            # Also search for businesses containing this name
            query_parts.append(f'node["name"~"{safe_name}",i]["shop"]{area_bounds};')
            query_parts.append(f'way["name"~"{safe_name}",i]["shop"]{area_bounds};')

        # Add landmark queries
        for landmark in features.get("landmarks", []):
            safe_name = landmark.replace('"', '\\"')
            query_parts.append(f'node["historic"~"."]["name"~"{safe_name}",i]{area_bounds};')
            query_parts.append(f'way["historic"~"."]["name"~"{safe_name}",i]{area_bounds};')
            # Also search for general amenities and buildings with this name
            query_parts.append(f'node["amenity"]["name"~"{safe_name}",i]{area_bounds};')
            query_parts.append(f'way["amenity"]["name"~"{safe_name}",i]{area_bounds};')

        # Add architectural style queries
        if features.get("architecture_style"):
            style = features["architecture_style"].replace('"', '\\"')
            query_parts.append(f'way["building"]["architecture"~"{style}",i]{area_bounds};')
            # Also search for historic buildings with this style
            query_parts.append(f'way["historic"]["architecture"~"{style}",i]{area_bounds};')

        # Combine all query parts
        if not query_parts:
            # Fallback to general amenity search if no specific features
            query_parts = [f'node["amenity"]{area_bounds};', f'way["amenity"]{area_bounds};']

        # Add shorter timeout for more focused searches
        timeout = 10 if initial_coords and initial_coords.get("confidence", 0) > 0.7 else 25

        query = f"""
        [out:json][timeout:{timeout}];
        (
            {chr(10).join(query_parts)}
        );
        out body;
        >;
        out skel qt;
        """

        return query

    def _search_google_maps(self, features: Dict, initial_coords: Optional[Dict] = None) -> List[Dict]:
        """Search locations using Google Maps API"""
        if not self.gmaps:
            return []

        results = []
        try:
            # Search for business names
            for business in features["business_names"]:  # Use processed business names
                print(f"\nSearching Google Maps for business: {business}")

                places = self.gmaps.places(
                    business, location=f"{initial_coords['lat']},{initial_coords['lon']}" if initial_coords else None, radius=5000 if initial_coords else None
                )

                for place in places.get("results", []):
                    print(f"Found place: {place['name']} at {place['geometry']['location']}")
                    results.append(
                        {
                            "source": "google_maps",
                            "name": place["name"],
                            "lat": place["geometry"]["location"]["lat"],
                            "lon": place["geometry"]["location"]["lng"],
                            "confidence": 0.8 if place.get("rating") else 0.6,
                            "type": "business",
                            "metadata": {
                                "place_id": place["place_id"],
                                "address": place.get("formatted_address"),
                                "types": place.get("types", []),
                                "rating": place.get("rating"),
                            },
                        }
                    )

            # Search for street addresses
            for street in features["street_signs"]:
                print(f"\nSearching Google Maps for street: {street}")

                geocode_results = self.gmaps.geocode(
                    street, region="de", location=f"{initial_coords['lat']},{initial_coords['lon']}" if initial_coords else None
                )

                for result in geocode_results:
                    print(f"Found address: {result['formatted_address']}")
                    results.append(
                        {
                            "source": "google_maps",
                            "name": result["formatted_address"],
                            "lat": result["geometry"]["location"]["lat"],
                            "lon": result["geometry"]["location"]["lng"],
                            "confidence": 0.85 if result.get("partial_match") else 0.95,
                            "type": "address",
                            "metadata": {
                                "place_id": result["place_id"],
                                "types": result.get("types", []),
                                "address_components": result.get("address_components", []),
                            },
                        }
                    )

            return results

        except Exception as e:
            print(f"Error searching Google Maps: {e}")
            return []
