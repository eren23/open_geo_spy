import overpy
import requests
from typing import Dict, List, Optional
import json


class GeoDataInterface:
    def __init__(self, geonames_username: str = None):
        self.osm_api = overpy.Overpass()
        self.geonames_username = geonames_username

    def search_location_candidates(self, features: Dict, location: str = None) -> List[Dict]:
        candidates = []

        # If location is provided, get its coordinates first
        initial_coords = None
        if location and self.geonames_username:
            initial_coords = self._get_location_coordinates(location)

        # Search OpenStreetMap
        osm_results = self._search_osm(features, initial_coords)
        candidates.extend(osm_results)

        # Search GeoNames if credentials provided
        if self.geonames_username:
            geonames_results = self._search_geonames(features, initial_coords)
            candidates.extend(geonames_results)

        # Search Wikidata for landmarks
        wikidata_results = self._search_wikidata(features, initial_coords)
        candidates.extend(wikidata_results)

        return self._deduplicate_candidates(candidates)

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

    def _build_osm_query(self, features, initial_coords: Optional[Dict] = None):
        # Convert features to OSM query
        pass
