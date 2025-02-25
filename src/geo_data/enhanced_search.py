from typing import List, Dict, Optional
from googlesearch import search
import overpy
import asyncio
from concurrent.futures import ThreadPoolExecutor
import re
from collections import defaultdict
import requests
from bs4 import BeautifulSoup
import logging
from urllib.parse import urlparse, unquote
import json

logger = logging.getLogger("geolocator_client")


class EnhancedLocationSearch:
    def __init__(self, geonames_username: str = None, max_google_results: int = 10, max_osm_queries: int = 5):
        self.osm_api = overpy.Overpass()
        self.geonames_username = geonames_username
        self.max_google_results = max_google_results
        self.max_osm_queries = max_osm_queries

    def _extract_structured_data(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract structured address data from schema.org markup"""
        addresses = []

        # Look for schema.org structured data
        for item in soup.find_all(["script", "div", "span"], attrs={"type": "application/ld+json"}):
            try:
                data = json.loads(item.string)
                if isinstance(data, dict):
                    self._extract_address_from_schema(data, addresses)
                elif isinstance(data, list):
                    for entry in data:
                        if isinstance(entry, dict):
                            self._extract_address_from_schema(entry, addresses)
            except:
                continue

        # Look for address elements
        for addr in soup.find_all(["address", "div"], class_=lambda x: x and any(c in x.lower() for c in ["address", "location", "contact"])):
            address = self._parse_address_element(addr)
            if address:
                addresses.append(address)

        return addresses

    def _parse_address_element(self, element) -> Optional[Dict[str, str]]:
        """Parse an HTML element containing address information"""
        text = element.get_text(separator=" ", strip=True)

        # Skip if text is too short or doesn't look like an address
        if len(text) < 10 or not any(x in text.lower() for x in ["straße", "strasse", "str.", "platz", "allee"]):
            return None

        # Try to identify address components
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        address = {}
        for line in lines:
            line = line.strip(",")
            # Look for postal code and city
            if re.match(r"^\d{5}\s+\w+", line):
                postal_city = line.split(None, 1)
                address["postal_code"] = postal_city[0]
                address["city"] = postal_city[1]
            # Look for street and number
            elif any(x in line.lower() for x in ["straße", "strasse", "str.", "platz", "allee"]):
                parts = line.split()
                if parts[-1].replace("-", "").isdigit():  # Number at end
                    address["street"] = " ".join(parts[:-1])
                    address["number"] = parts[-1]
                elif parts[0].replace("-", "").isdigit():  # Number at start
                    address["number"] = parts[0]
                    address["street"] = " ".join(parts[1:])

        return address if "street" in address else None

    async def _fetch_webpage_content(self, url: str) -> Optional[Dict]:
        """Fetch and extract content from webpage"""
        try:
            if "/search?" in url:
                return None

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            }

            async with asyncio.timeout(10):
                response = await asyncio.to_thread(requests.get, url, headers=headers, timeout=10)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")

                # Remove non-content elements
                for tag in soup(["script", "style", "meta", "link", "noscript"]):
                    tag.decompose()

                # Extract addresses from structured data
                structured_addresses = self._extract_structured_data(soup)

                # Extract addresses from contact/location sections
                contact_sections = soup.find_all(
                    ["div", "section"], class_=lambda x: x and any(c in x.lower() for c in ["contact", "location", "address", "footer"])
                )

                section_addresses = []
                for section in contact_sections:
                    addr = self._parse_address_element(section)
                    if addr:
                        section_addresses.append(addr)

                return {
                    "url": url,
                    "structured_addresses": structured_addresses,
                    "section_addresses": section_addresses,
                    "title": soup.title.string if soup.title else None,
                }

        except Exception as e:
            logger.debug(f"Error fetching {url}: {str(e)}")
        return None

    async def _analyze_search_results(self, search_results: List[str]) -> List[Dict[str, str]]:
        """Analyze search results to find addresses"""
        addresses = []
        seen_addresses = set()

        # Fetch and analyze webpage content
        tasks = []
        for url in search_results:
            if not url.startswith("/search"):
                tasks.append(self._fetch_webpage_content(url))

        contents = await asyncio.gather(*tasks)

        for content in contents:
            if not content:
                continue

            # Process structured addresses
            for addr in content["structured_addresses"]:
                addr_key = f"{addr.get('street', '')}{addr.get('number', '')}{addr.get('city', '')}"
                if addr_key and addr_key not in seen_addresses:
                    seen_addresses.add(addr_key)
                    addresses.append(addr)

            # Process section addresses
            for addr in content["section_addresses"]:
                addr_key = f"{addr.get('street', '')}{addr.get('number', '')}{addr.get('city', '')}"
                if addr_key and addr_key not in seen_addresses:
                    seen_addresses.add(addr_key)
                    addresses.append(addr)

            # Extract any address from the URL itself
            parsed_url = urlparse(content["url"])
            path_parts = unquote(parsed_url.path).split("/")

            for part in path_parts:
                if any(x in part.lower() for x in ["strasse", "str", "allee"]):
                    # Try to extract street and number from URL part
                    components = part.split("-")
                    for i, comp in enumerate(components):
                        if comp.replace("-", "").isdigit() and i > 0:
                            potential_street = " ".join(components[:i])
                            if len(potential_street) > 5:  # Avoid too short strings
                                addr = {"street": potential_street.replace("-", " ").title(), "number": comp, "city": parsed_url.netloc.split(".")[0].title()}
                                addr_key = f"{addr['street']}{addr['number']}{addr['city']}"
                                if addr_key not in seen_addresses:
                                    seen_addresses.add(addr_key)
                                    addresses.append(addr)

        return addresses

    async def _run_google_search(self, query: str) -> List[str]:
        """Run Google search and return results"""
        try:
            results = list(search(query, num_results=5, lang="de"))  # Use German results
            logger.info(f"Search results for '{query}':")
            for result in results:
                logger.info(f"- {result}")

            # Analyze search results for addresses
            addresses = await self._analyze_search_results(results)
            if addresses:
                logger.info("Found addresses in search results:")
                for addr in addresses:
                    logger.info(f"- {addr}")
                    # Add the found address as a new search query
                    addr_query = f"{addr.get('street', '')} {addr.get('number', '')}"
                    if addr.get("city"):
                        addr_query += f", {addr['city']}"
                    if addr_query not in results:
                        results.append(addr_query)

            return results
        except Exception as e:
            logger.error(f"Google search error for '{query}': {e}")
            return []

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

        # Extract city from location hint if available
        city = None
        if location_hint:
            city_match = re.search(r"(?:^|\s)([A-Z][a-zA-Z]+)(?:\s|$)", location_hint)
            if city_match:
                city = city_match.group(1)

        # Start with business names - make direct queries
        businesses = features.get("extracted_text", {}).get("business_names", [])
        if businesses:
            print(f"Found businesses: {businesses}")
            for business in businesses:
                # Clean business name
                business = business.strip("\"'")  # Remove quotes
                if city:
                    queries.append(f"{business} {city}")  # Direct city search
                    queries.append(f"{business} address {city}")  # Try to find exact address
                else:
                    queries.append(f"{business} location")
                print(f"Added business query: {business}")

        # Add any specific business mentions from text
        for info in features.get("extracted_text", {}).get("informational", []):
            if "biomarkt" in info.lower() or "bio markt" in info.lower():
                if city:
                    queries.append(f"Denn's Biomarkt {city}")
                    queries.append(f"Denn's Bio Markt {city}")
                else:
                    queries.append("Denn's Biomarkt location")
                print("Added Denn's Biomarkt query")

        # Add street names if available
        streets = features.get("extracted_text", {}).get("street_signs", [])
        if streets:
            print(f"Found streets: {streets}")
            for street in streets:
                if city:
                    queries.append(f"{street} {city}")
                else:
                    queries.append(f"{street} location")
                print(f"Added street query: {street}")

        # Add license plate info if available
        for text in features.get("extracted_text", {}).get("other", []):
            if "KA" in text:  # Karlsruhe license plate
                queries.append("Denn's Biomarkt Karlsruhe")
                queries.append("Biomarkt Karlsruhe")
                print("Added Karlsruhe-specific queries")

        print(f"Total queries built: {len(queries)}")
        print("===========================\n")
        return queries

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
