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
        """Build optimized search queries with improved entity-location pairing"""
        print("\n=== Building Search Queries ===")
        queries = []

        # Extract location context from various sources
        location_contexts = self._extract_location_contexts(features, location_hint)
        print(f"Identified location contexts: {location_contexts}")

        # Process business names with location context
        businesses = features.get("business_names", [])
        if businesses:
            print(f"Found businesses: {businesses}")
            for business in businesses:
                # Clean business name
                business = business.strip("\"'")

                # Check if this business has a specific location context
                business_location = features.get("entity_locations", {}).get(business)

                # Build queries with different location contexts
                if business_location:
                    # Use the specific location associated with this business
                    queries.append(f"{business} {business_location}")
                    queries.append(f"{business} address {business_location}")
                    print(f"Added business-specific query: {business} in {business_location}")
                elif location_contexts:
                    # Try each location context
                    for context in location_contexts:
                        queries.append(f"{business} {context}")
                        queries.append(f"{business} address {context}")
                    print(f"Added business query with contexts: {business}")
                else:
                    # Generic search
                    queries.append(f"{business} location")
                    print(f"Added generic business query: {business}")

        # Process street names with location context
        streets = features.get("street_signs", [])
        if streets:
            print(f"Found streets: {streets}")
            for street in streets:
                if location_contexts:
                    for context in location_contexts:
                        queries.append(f"{street} {context}")
                    print(f"Added street query with contexts: {street}")
                else:
                    queries.append(f"{street} location")
                    print(f"Added generic street query: {street}")

        # Process building info with streets for more precise searches
        buildings = features.get("building_info", [])
        if buildings and streets:
            for building in buildings:
                for street in streets:
                    queries.append(f"{building} {street}")
                    print(f"Added building-street query: {building} {street}")

        # Add combined queries for higher precision
        if businesses and streets:
            for business in businesses:
                for street in streets:
                    queries.append(f"{business} {street}")
                    print(f"Added business-street query: {business} {street}")

        print(f"Total queries built: {len(queries)}")
        print("===========================\n")
        return queries

    def _extract_location_contexts(self, features: Dict, location_hint: str = None) -> List[str]:
        """Extract all potential location contexts from features and hints"""
        contexts = set()

        # Add explicit location hint
        if location_hint:
            contexts.add(location_hint)

        # Add entity-specific locations
        for location in features.get("entity_locations", {}).values():
            contexts.add(location)

        # Extract locations from license plate info
        for plate_info in features.get("license_plate_info", []):
            if "region_name" in plate_info:
                contexts.add(plate_info["region_name"])

        # Look for location names in informational text
        for info in features.get("extracted_text", {}).get("informational", []):
            location_match = re.search(r"(?:in|at|near)\s+([A-Z][a-zA-Z\s]+)(?:,|\.|$)", info)
            if location_match:
                contexts.add(location_match.group(1).strip())

        return list(contexts)

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

        # Simplified patterns focusing on addresses and streets
        location_patterns = [
            # Street addresses (both German and general format)
            r"(\d+\s+[A-Za-zäöüß\s\-]+(?:straße|strasse|str\.|street))",
            r"([A-Za-zäöüß\s\-]+(?:straße|strasse|str\.|street)\s+\d+)",
            # Postal codes with city
            r"(\d{5}\s+[A-Za-zäöüß\s\-]+)",
            # Simple street names
            r"([A-Za-zäöüß]+(?:straße|strasse))",
        ]

        for results in search_results:
            for result in results:
                # Clean up the result text
                result = result.replace("\n", " ").replace("\r", " ")

                # Look for patterns
                for pattern in location_patterns:
                    matches = re.findall(pattern, result, re.IGNORECASE)
                    for match in matches:
                        if isinstance(match, tuple):
                            match = match[0]
                        location = match.strip()
                        # Give higher weight to full addresses
                        if re.search(r"\d+", location):  # Contains numbers (likely an address)
                            location_counts[location] += 3
                        else:
                            location_counts[location] += 1

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
