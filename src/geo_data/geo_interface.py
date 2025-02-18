import overpy
import requests
from typing import Dict, List, Optional
import json


class GeoDataInterface:
    def __init__(self, geonames_username: str = None):
        self.osm_api = overpy.Overpass()
        self.geonames_username = geonames_username

    def search_location_candidates(self, features: Dict, location: str = None) -> List[Dict]:
        """Search for location candidates using multiple data sources"""
        candidates = []

        # If location is provided, get its coordinates first
        initial_coords = None
        if location and self.geonames_username:
            initial_coords = self._get_location_coordinates(location)

        # 1. Search OpenStreetMap
        osm_results = self._search_osm(features, initial_coords)
        candidates.extend(osm_results)

        # 2. Search GeoNames if credentials provided
        if self.geonames_username:
            geonames_results = self._search_geonames(features, initial_coords)
            candidates.extend(geonames_results)

        # 3. Search Wikidata for landmarks
        wikidata_results = self._search_wikidata(features, initial_coords)
        candidates.extend(wikidata_results)

        # 4. Search for specific businesses
        business_results = self._search_businesses(features, initial_coords)
        candidates.extend(business_results)

        # 5. Search for transportation infrastructure
        transport_results = self._search_transport_infrastructure(features, initial_coords)
        candidates.extend(transport_results)

        # 6. Search for cultural landmarks
        cultural_results = self._search_cultural_landmarks(features, initial_coords)
        candidates.extend(cultural_results)

        # Deduplicate and rank results
        candidates = self._deduplicate_candidates(candidates)
        return self._rank_candidates(candidates, features, initial_coords)

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
        """Search for specific businesses mentioned in features"""
        results = []
        business_names = features.get("extracted_text", {}).get("business_names", [])

        if not business_names:
            return results

        try:
            for business in business_names:
                # Search OSM for businesses
                query = f"""
                [out:json][timeout:25];
                (
                  node["name"~"{business}",i]
                    ({initial_coords['bbox']['south'] if initial_coords else -90},
                     {initial_coords['bbox']['west'] if initial_coords else -180},
                     {initial_coords['bbox']['north'] if initial_coords else 90},
                     {initial_coords['bbox']['east'] if initial_coords else 180});
                  way["name"~"{business}",i]
                    ({initial_coords['bbox']['south'] if initial_coords else -90},
                     {initial_coords['bbox']['west'] if initial_coords else -180},
                     {initial_coords['bbox']['north'] if initial_coords else 90},
                     {initial_coords['bbox']['east'] if initial_coords else 180});
                );
                out body;
                >;
                out skel qt;
                """

                response = self.osm_api.query(query)

                for element in response.nodes:
                    results.append(
                        {
                            "source": "osm_business",
                            "name": element.tags.get("name", business),
                            "lat": float(element.lat),
                            "lon": float(element.lon),
                            "type": "business",
                            "confidence": 0.8 if business.lower() in element.tags.get("name", "").lower() else 0.6,
                            "metadata": {
                                "osm_id": element.id,
                                "tags": dict(element.tags),
                                "business_type": element.tags.get("shop") or element.tags.get("amenity"),
                            },
                        }
                    )

            return results
        except Exception as e:
            print(f"Error searching businesses: {e}")
            return []

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
        """Build OSM query based on features and location constraints"""
        # Build area constraints
        area_bounds = (
            f"({initial_coords['bbox']['south']},{initial_coords['bbox']['west']}," f"{initial_coords['bbox']['north']},{initial_coords['bbox']['east']})"
            if initial_coords
            else "(-90,-180,90,180)"
        )

        # Build feature filters
        filters = []

        # Add architectural style filter if present
        if features.get("architecture_style"):
            style = features["architecture_style"].replace('"', '\\"')  # Escape quotes
            filters.append(f'["architecture"~"{style}",i]')

        # Add landmark filters
        for landmark in features.get("landmarks", []):
            safe_landmark = landmark.replace('"', '\\"')  # Escape quotes
            filters.append(f'["name"~"{safe_landmark}",i]')

        # Add business name filters
        for business in features.get("extracted_text", {}).get("business_names", []):
            safe_business = business.replace('"', '\\"')  # Escape quotes
            filters.append(f'["name"~"{safe_business}",i]')

        # Combine filters
        filter_str = "".join(filters) if filters else ""

        # Build complete query with reasonable timeout
        query = f"""
        [out:json][timeout:15];
        (
          node{filter_str}{area_bounds};
          way{filter_str}{area_bounds};
          relation{filter_str}{area_bounds};
        );
        out body;
        >;
        out skel qt;
        """

        return query
