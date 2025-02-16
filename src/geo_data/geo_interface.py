import overpy
import requests
from typing import Dict, List
import json


class GeoDataInterface:
    def __init__(self, geonames_username: str = None):
        self.osm_api = overpy.Overpass()
        self.geonames_username = geonames_username

    def search_location_candidates(self, features: Dict) -> List[Dict]:
        candidates = []

        # Search OpenStreetMap
        osm_results = self._search_osm(features)
        candidates.extend(osm_results)

        # Search GeoNames if credentials provided
        if self.geonames_username:
            geonames_results = self._search_geonames(features)
            candidates.extend(geonames_results)

        # Search Wikidata for landmarks
        wikidata_results = self._search_wikidata(features)
        candidates.extend(wikidata_results)

        return self._deduplicate_candidates(candidates)

    def _search_osm(self, features: Dict) -> List[Dict]:
        """Search OpenStreetMap for matching locations"""
        query = self._build_osm_query(features)
        try:
            results = self.osm_api.query(query)
            candidates = []

            for result in results.ways:
                candidate = {
                    "source": "osm",
                    "name": result.tags.get("name", "Unknown"),
                    "lat": result.nodes[0].lat,
                    "lon": result.nodes[0].lon,
                    "type": result.tags.get("amenity", "unknown"),
                    "confidence": self._calculate_confidence(features, result),
                    "metadata": {"osm_id": result.id, "tags": result.tags},
                }
                candidates.append(candidate)

            return candidates
        except Exception as e:
            print(f"OSM search error: {e}")
            return []

    def _search_geonames(self, features: Dict) -> List[Dict]:
        """Search GeoNames database"""
        if not self.geonames_username:
            return []

        try:
            # Search for landmarks and features
            landmarks = features.get("landmarks", [])
            results = []

            for landmark in landmarks:
                url = f"http://api.geonames.org/searchJSON"
                params = {"q": landmark, "maxRows": 5, "username": self.geonames_username, "style": "FULL"}

                response = requests.get(url, params=params)
                data = response.json()

                for result in data.get("geonames", []):
                    candidate = {
                        "source": "geonames",
                        "name": result["name"],
                        "lat": float(result["lat"]),
                        "lon": float(result["lng"]),
                        "type": result.get("fcode", "unknown"),
                        "confidence": self._calculate_geonames_confidence(features, result),
                        "metadata": result,
                    }
                    results.append(candidate)

            return results
        except Exception as e:
            print(f"GeoNames search error: {e}")
            return []

    def _search_wikidata(self, features: Dict) -> List[Dict]:
        """Search Wikidata for landmarks and locations"""
        try:
            landmarks = features.get("landmarks", [])
            results = []

            for landmark in landmarks:
                # SPARQL query to find locations
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
                }}
                LIMIT 5
                """

                url = "https://query.wikidata.org/sparql"
                headers = {"Accept": "application/json"}
                params = {"query": query, "format": "json"}

                response = requests.get(url, headers=headers, params=params)
                data = response.json()

                for result in data.get("results", {}).get("bindings", []):
                    candidate = {
                        "source": "wikidata",
                        "name": result["itemLabel"]["value"],
                        "lat": float(result["lat"]["value"]),
                        "lon": float(result["lon"]["value"]),
                        "type": result["typeLabel"]["value"],
                        "confidence": 0.7,  # Base confidence for Wikidata matches
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

    def _build_osm_query(self, features):
        # Convert features to OSM query
        pass
