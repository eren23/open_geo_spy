from typing import List, Dict, Optional
from googlesearch import search
import overpy
import asyncio
from concurrent.futures import ThreadPoolExecutor
import re
from collections import defaultdict


class EnhancedLocationSearch:
    def __init__(self, geonames_username: str = None, max_google_results: int = 10, max_osm_queries: int = 5):
        self.osm_api = overpy.Overpass()
        self.geonames_username = geonames_username
        self.max_google_results = max_google_results
        self.max_osm_queries = max_osm_queries

    async def find_location_candidates(self, features: Dict, metadata: Dict = None, location_hint: str = None) -> List[Dict]:
        """Find location candidates using multiple sources"""
        print("\n=== Starting Location Search ===")
        print("Features received:", features)

        # Extract location names from description and text
        description = features.get("description", "")
        extracted_text = features.get("extracted_text", {})

        # Add any informational text to landmarks
        for info in extracted_text.get("informational", []):
            if "straße" in info.lower() or "strasse" in info.lower():
                street_name = re.search(r"([A-Za-zäöüß]+straße|[A-Za-zäöüß]+strasse)", info, re.IGNORECASE)
                if street_name:
                    features.setdefault("landmarks", []).append(street_name.group(1))
                    print(f"Added street from text: {street_name.group(1)}")

        # Extract locations from description
        if description:
            # Extract potential location names from description
            location_matches = []
            patterns = [
                r"in ([A-Z][a-zA-Z\s]+)(?:,|\.|$)",
                r"shows ([A-Z][a-zA-Z\s]+)(?:,|\.|$)",
                r"near ([A-Z][a-zA-Z\s]+)(?:,|\.|$)",
                r"at ([A-Z][a-zA-Z\s]+)(?:,|\.|$)",
                r"(?:city|town|village|district) of ([A-Z][a-zA-Z\s]+)(?:,|\.|$)",
            ]

            for pattern in patterns:
                matches = re.findall(pattern, description)
                location_matches.extend(matches)

            # Add description-based queries
            for location in location_matches:
                features.setdefault("landmarks", []).append(location.strip())
                print(f"Added location from description: {location.strip()}")

        # Build search queries
        search_queries = []

        # Add street-based queries
        for text in extracted_text.get("informational", []):
            if "straße" in text.lower() or "strasse" in text.lower():
                search_queries.append(f"{text} location")
                print(f"Added street search query: {text} location")

        # Add regular queries
        search_queries.extend(self._build_search_queries(features, metadata, location_hint))

        print(f"\nExecuting {len(search_queries)} search queries...")

        # Run Google searches in parallel
        search_results = []
        for query in search_queries[: self.max_google_results]:
            try:
                print(f"\nSearching for: {query}")
                results = await self._run_google_search(query)
                search_results.append(results)
                print(f"Found {len(results)} results")
            except Exception as e:
                print(f"Search error: {e}")
                continue

        # Extract location mentions from search results
        location_mentions = self._extract_locations(search_results)
        print("\nLocation mentions found:", dict(location_mentions))

        # Get coordinates for top mentioned locations
        candidates = await self._get_location_coordinates(location_mentions)

        print(f"\nFound {len(candidates)} candidates:")
        for candidate in candidates:
            print(f"- {candidate['name']} ({candidate['lat']}, {candidate['lon']}) confidence: {candidate['confidence']}")
        print("===========================\n")

        return candidates

    def _build_search_queries(self, features: Dict, metadata: Dict = None, location_hint: str = None) -> List[str]:
        """Build optimized search queries from features"""
        print("\n=== Building Search Queries ===")
        queries = []

        # Start with any business names
        businesses = features.get("extracted_text", {}).get("business_names", [])
        if businesses:
            print(f"Found businesses: {businesses}")
        for business in businesses[:2]:  # Limit to 2 most prominent businesses
            query = f"{business} location address"
            queries.append(query)
            print(f"Added business query: {query}")

        # Add landmark queries
        landmarks = features.get("landmarks", [])
        if landmarks:
            print(f"Found landmarks: {landmarks}")
        for landmark in landmarks[:2]:
            query = f"{landmark} location coordinates"
            queries.append(query)
            print(f"Added landmark query: {query}")

        # Add street names
        streets = features.get("extracted_text", {}).get("street_signs", [])
        if streets:
            print(f"Found streets: {streets}")
            street_query = f"{streets[0]} street"
            if location_hint:
                street_query += f" {location_hint}"
            queries.append(street_query)
            print(f"Added street query: {street_query}")

        # Add architectural style if distinctive
        if features.get("architecture_style"):
            print(f"Found architecture: {features['architecture_style']}")
            arch_query = f"{features['architecture_style']} architecture location"
            if location_hint:
                arch_query += f" {location_hint}"
            queries.append(arch_query)
            print(f"Added architecture query: {arch_query}")

        # Add license plate region if available
        if features.get("license_plate_info"):
            print(f"Found license plates: {features['license_plate_info']}")
            for plate_info in features["license_plate_info"]:
                if plate_info.get("region_name"):
                    query = f"{plate_info['region_name']} {plate_info.get('country', '')} location"
                    queries.append(query)
                    print(f"Added license plate query: {query}")

        print(f"Total queries built: {len(queries)}")
        print("===========================\n")
        return queries

    async def _run_google_search(self, query: str) -> List[str]:
        """Run Google search and return results"""
        try:
            results = list(search(query, num_results=5, lang="en"))  # Limit to 5 results per query
            print(f"Search results for '{query}':")
            for result in results:
                print(f"- {result}")
            return results
        except Exception as e:
            print(f"Google search error for '{query}': {e}")
            return []

    def _extract_locations(self, search_results: List[List[str]]) -> Dict[str, int]:
        """Extract and count location mentions from search results"""
        location_counts = defaultdict(int)

        # Common location patterns
        location_patterns = [
            r"in ([A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*)",  # City names
            r"([A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*) district",  # Districts
            r"([A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*) neighborhood",  # Neighborhoods
            r"([A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*) street",  # Streets
            r"([A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*) avenue",  # Avenues
            r"([A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*), [A-Z]{2}",  # City, State
            r"([A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*) [0-9]{5}",  # City Zipcode
        ]

        for results in search_results:
            for result in results:
                # Clean up the result text
                result = result.replace("\n", " ").replace("\r", " ")

                # Look for patterns
                for pattern in location_patterns:
                    matches = re.findall(pattern, result)
                    for match in matches:
                        if isinstance(match, tuple):
                            match = match[0]  # Take first group if multiple groups
                        location_counts[match.strip()] += 1

                # Special handling for German streets
                street_matches = re.findall(r"([A-Za-zäöüß]+(?:straße|strasse))", result, re.IGNORECASE)
                for match in street_matches:
                    location_counts[match.strip()] += 2  # Give extra weight to street matches

        return location_counts

    async def _get_location_coordinates(self, location_mentions: Dict[str, int]) -> List[Dict]:
        """Get coordinates for locations using OSM"""
        candidates = []
        query_count = 0

        for location, count in sorted(location_mentions.items(), key=lambda x: x[1], reverse=True):
            if query_count >= self.max_osm_queries:
                break

            try:
                # Build OSM query
                query = f"""
                [out:json][timeout:5];
                (
                  node["name"~"{location}",i];
                  way["name"~"{location}",i];
                  relation["name"~"{location}",i];
                );
                out center;
                """

                response = self.osm_api.query(query)
                query_count += 1

                # Process results
                for element in response.nodes:
                    candidates.append(
                        {
                            "source": "osm",
                            "name": location,
                            "lat": float(element.lat),
                            "lon": float(element.lon),
                            "confidence": min(0.7 + (count / 10), 0.9),  # Boost confidence based on mention count
                            "type": "search_result",
                            "metadata": {"mentions": count, "tags": dict(element.tags)},
                        }
                    )

            except Exception as e:
                print(f"OSM query error for {location}: {e}")
                continue

        return candidates

    def _rank_candidates(self, candidates: List[Dict], features: Dict) -> List[Dict]:
        """Rank and deduplicate candidates"""
        # Remove duplicates based on proximity
        unique_candidates = []
        seen_locations = set()

        for candidate in candidates:
            location_key = f"{round(candidate['lat'], 3)},{round(candidate['lon'], 3)}"
            if location_key not in seen_locations:
                seen_locations.add(location_key)

                # Adjust confidence based on feature matches
                confidence = candidate["confidence"]

                # Boost confidence if features match
                if features.get("architecture_style") and candidate.get("metadata", {}).get("tags", {}).get("architecture") == features["architecture_style"]:
                    confidence = min(confidence + 0.1, 1.0)

                if any(business.lower() in candidate["name"].lower() for business in features.get("extracted_text", {}).get("business_names", [])):
                    confidence = min(confidence + 0.15, 1.0)

                candidate["confidence"] = confidence
                unique_candidates.append(candidate)

        # Sort by confidence
        return sorted(unique_candidates, key=lambda x: x["confidence"], reverse=True)
