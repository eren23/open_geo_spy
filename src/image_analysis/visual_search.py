from typing import List, Dict, Optional
import requests
from PIL import Image
import io
import base64
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from googlesearch import search
import overpy


class VisualSearchEngine:
    def __init__(self, google_api_key: str = None, bing_api_key: str = None):
        self.google_api_key = google_api_key
        self.bing_api_key = bing_api_key
        self.osm_api = overpy.Overpass()
        self.search_apis = {"google": self._search_google_images, "duckduckgo": self._search_duckduckgo, "osm": self._search_osm_images}

    async def find_similar_locations(self, image_path: str, initial_location: Dict = None) -> List[Dict]:
        """Find visually similar locations using multiple search engines"""
        results = []
        search_area = self._get_search_area(initial_location) if initial_location else None

        # Convert image to format suitable for API requests
        image_data = self._prepare_image(image_path)

        # Perform parallel searches
        with ThreadPoolExecutor() as executor:
            futures = []
            for engine, search_func in self.search_apis.items():
                future = executor.submit(search_func, image_data, search_area)
                futures.append((engine, future))

            # Collect results
            for engine, future in futures:
                try:
                    engine_results = future.result()
                    if engine_results:
                        results.extend(engine_results)
                except Exception as e:
                    print(f"Error with {engine} search: {e}")

        # Deduplicate and rank results
        unique_results = self._deduplicate_results(results)
        ranked_results = self._rank_results(unique_results, initial_location)

        return ranked_results

    def _prepare_image(self, image_path: str) -> Dict:
        """Prepare image data for API requests"""
        if image_path.startswith(("http://", "https://")):
            response = requests.get(image_path)
            image = Image.open(io.BytesIO(response.content))
        else:
            image = Image.open(image_path)

        # Convert to bytes for API requests
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()
        base64_image = base64.b64encode(img_bytes).decode()

        return {"base64": base64_image, "bytes": img_bytes, "size": image.size}

    def _get_search_area(self, location: Dict) -> Dict:
        """Calculate search area based on initial location"""
        lat, lon = location["lat"], location["lon"]
        return {
            "center": (lat, lon),
            "radius": 5000,  # 5km radius
            "bounds": {"north": lat + 0.045, "south": lat - 0.045, "east": lon + 0.045, "west": lon - 0.045},  # Roughly 5km in degrees
        }

    def _search_google_images(self, image_data: Dict, search_area: Dict = None) -> List[Dict]:
        """Search Google for similar images"""
        if not self.google_api_key:
            return []

        endpoint = "https://customsearch.googleapis.com/customsearch/v1"
        params = {
            "key": self.google_api_key,
            "cx": CONFIG["GOOGLE_SEARCH_CX"],
            "searchType": "image",
            "imgSize": "large",
            "num": 10,
            "rights": "cc_publicdomain,cc_attribute,cc_sharealike",  # Creative Commons images
            "safe": "active",
        }

        # Add location context if available
        if search_area:
            location_str = f"{search_area['center'][0]},{search_area['center'][1]}"
            params.update({"gl": "us", "googlehost": "google.com", "q": f"location:{location_str}"})  # Geolocation

        try:
            # First, try reverse image search
            if "base64" in image_data:
                params["searchType"] = "image"
                params["imgdata"] = image_data["base64"]

            response = requests.get(endpoint, params=params)
            data = response.json()

            results = []
            for item in data.get("items", []):
                # Extract location information from image metadata
                location_candidates = self._extract_location_from_text(f"{item.get('title', '')} {item.get('snippet', '')} {item.get('displayLink', '')}")

                for location in location_candidates:
                    similarity_score = 0.6  # Base score
                    if "image" in item and "thumbnailLink" in item["image"]:
                        try:
                            similarity_score = self._calculate_image_similarity(image_data["bytes"], requests.get(item["image"]["thumbnailLink"]).content)
                        except:
                            pass

                    results.append(
                        {
                            "source": "google",
                            "url": item["link"],
                            "title": item.get("title", "Unknown"),
                            "lat": location["lat"],
                            "lon": location["lon"],
                            "similarity_score": similarity_score * location["confidence"],
                            "type": "image_match",
                            "metadata": {
                                "snippet": item.get("snippet"),
                                "page_url": item.get("image", {}).get("contextLink"),
                                "source_domain": item.get("displayLink"),
                                "location_source": location["source"],
                                "location_confidence": location["confidence"],
                            },
                        }
                    )

            return results

        except Exception as e:
            print(f"Google search error: {e}")
            return []

    def _calculate_image_similarity(self, img1_bytes: bytes, img2_bytes: bytes) -> float:
        """Calculate visual similarity between two images"""
        try:
            # Convert images to numpy arrays
            img1 = cv2.imdecode(np.frombuffer(img1_bytes, np.uint8), cv2.IMREAD_COLOR)
            img2 = cv2.imdecode(np.frombuffer(img2_bytes, np.uint8), cv2.IMREAD_COLOR)

            # Resize images to same size
            size = (224, 224)  # Standard size
            img1 = cv2.resize(img1, size)
            img2 = cv2.resize(img2, size)

            # Convert to grayscale
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # Calculate histograms
            hist1 = cv2.calcHist([img1_gray], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([img2_gray], [0], None, [256], [0, 256])

            # Normalize histograms
            cv2.normalize(hist1, hist1)
            cv2.normalize(hist2, hist2)

            # Calculate similarity
            similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

            # Return normalized similarity score
            return max(0.0, min(1.0, (similarity + 1) / 2))

        except Exception as e:
            print(f"Error calculating image similarity: {e}")
            return 0.5  # Default similarity score

    def _search_duckduckgo(self, image_data: Dict, search_area: Dict = None) -> List[Dict]:
        """Search DuckDuckGo for similar images"""
        try:
            # DuckDuckGo image search URL
            url = "https://duckduckgo.com/"
            params = {"q": "!gi", "iax": "images", "ia": "images"}  # Image search

            # First request to get token
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
            res = requests.post(url, data=params, headers=headers)

            # Extract search token
            token = res.text.split("vqd='", 1)[1].split("'", 1)[0]

            # Image search request
            url = "https://duckduckgo.com/i.js"
            params.update({"l": "us-en", "o": "json", "q": "!gi", "vqd": token, "f": ",,,", "p": "1"})

            if search_area:
                # Add location context to search
                location = f"{search_area['center'][0]},{search_area['center'][1]}"
                params["q"] += f" location:{location}"

            res = requests.get(url, headers=headers, params=params)
            data = res.json()

            results = []
            for image in data.get("results", []):
                # Extract location information from image title and source
                location_info = self._extract_location_from_text(image.get("title", "") + " " + image.get("source", ""))
                if location_info:
                    results.append(
                        {
                            "source": "duckduckgo",
                            "url": image["image"],
                            "title": image["title"],
                            "lat": location_info["lat"],
                            "lon": location_info["lon"],
                            "similarity_score": 0.6,  # Base similarity score
                            "type": "image_match",
                        }
                    )

            return results
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
            return []

    def _search_osm_images(self, image_data: Dict, search_area: Dict = None) -> List[Dict]:
        """Search OpenStreetMap for images and locations"""
        try:
            if not search_area:
                return []

            # Build OSM query for images in the area
            # Convert coordinates to proper format and ensure they're valid
            try:
                south = float(search_area["bounds"]["south"])
                west = float(search_area["bounds"]["west"])
                north = float(search_area["bounds"]["north"])
                east = float(search_area["bounds"]["east"])
            except (KeyError, ValueError, TypeError):
                print("Invalid search area coordinates")
                return []

            # Validate coordinate ranges
            if not (-90 <= south <= 90 and -90 <= north <= 90 and -180 <= west <= 180 and -180 <= east <= 180):
                print("Coordinates out of valid range")
                return []

            # Build the query using proper Overpass QL syntax with explicit string formatting
            query = """
            [out:json][timeout:25];
            (
              way[image]({south},{west},{north},{east});
              node[image]({south},{west},{north},{east});
              relation[image]({south},{west},{north},{east});
            );
            out body;
            >;
            out skel qt;
            """.format(
                south=south, west=west, north=north, east=east
            )

            results = []
            try:
                response = self.osm_api.query(query)

                # Process ways with images
                for way in response.ways:
                    if way.tags.get("image") and way.nodes:
                        try:
                            # Calculate center point of way
                            valid_nodes = [n for n in way.nodes if hasattr(n, "lat") and hasattr(n, "lon")]
                            if valid_nodes:
                                lat = sum(n.lat for n in valid_nodes) / len(valid_nodes)
                                lon = sum(n.lon for n in valid_nodes) / len(valid_nodes)

                                results.append(
                                    {
                                        "source": "osm",
                                        "url": way.tags.get("image"),
                                        "title": way.tags.get("name", "Unknown location"),
                                        "lat": lat,
                                        "lon": lon,
                                        "similarity_score": 0.7,
                                        "type": "osm_location",
                                        "osm_type": "way",
                                        "osm_id": way.id,
                                        "metadata": {"tags": dict(way.tags), "nodes_count": len(valid_nodes)},
                                    }
                                )
                        except Exception as e:
                            print(f"Error processing way {way.id}: {e}")
                            continue

                # Process nodes with images
                for node in response.nodes:
                    if node.tags.get("image"):
                        try:
                            results.append(
                                {
                                    "source": "osm",
                                    "url": node.tags.get("image"),
                                    "title": node.tags.get("name", "Unknown location"),
                                    "lat": float(node.lat),
                                    "lon": float(node.lon),
                                    "similarity_score": 0.7,
                                    "type": "osm_location",
                                    "osm_type": "node",
                                    "osm_id": node.id,
                                    "metadata": {"tags": dict(node.tags)},
                                }
                            )
                        except Exception as e:
                            print(f"Error processing node {node.id}: {e}")
                            continue

                # Build POI query with explicit string formatting
                poi_query = """
                [out:json][timeout:25];
                (
                  node["tourism"]({south},{west},{north},{east});
                  node["historic"]({south},{west},{north},{east});
                  node["landmark"]({south},{west},{north},{east});
                );
                out body;
                >;
                out skel qt;
                """.format(
                    south=south, west=west, north=north, east=east
                )

                try:
                    poi_response = self.osm_api.query(poi_query)

                    # Process POIs
                    for node in poi_response.nodes:
                        if hasattr(node, "lat") and hasattr(node, "lon"):
                            try:
                                poi_type = next((k for k in ["tourism", "historic", "landmark"] if k in node.tags), "unknown")
                                results.append(
                                    {
                                        "source": "osm",
                                        "url": node.tags.get("image", ""),
                                        "title": node.tags.get("name", "Unknown POI"),
                                        "lat": float(node.lat),
                                        "lon": float(node.lon),
                                        "similarity_score": 0.5,
                                        "type": "osm_poi",
                                        "osm_type": "node",
                                        "osm_id": node.id,
                                        "metadata": {"tags": dict(node.tags), "type": poi_type},
                                    }
                                )
                            except Exception as e:
                                print(f"Error processing POI {node.id}: {e}")
                                continue

                except overpy.exception.OverpassGatewayTimeout:
                    print("POI query timeout, skipping POIs")

            except overpy.exception.OverpassTooManyRequests:
                print("OSM rate limit exceeded, waiting...")
                import time

                time.sleep(5)
                return self._search_osm_images(image_data, search_area)

            except overpy.exception.OverpassGatewayTimeout:
                print("OSM gateway timeout, trying with smaller area...")
                smaller_area = {
                    "center": search_area["center"],
                    "radius": search_area["radius"] / 2,
                    "bounds": {
                        "north": (north + float(search_area["center"][0])) / 2,
                        "south": (south + float(search_area["center"][0])) / 2,
                        "east": (east + float(search_area["center"][1])) / 2,
                        "west": (west + float(search_area["center"][1])) / 2,
                    },
                }
                return self._search_osm_images(image_data, smaller_area)

            return results

        except Exception as e:
            print(f"OSM search error: {str(e)}")
            return []

    def _extract_location_from_text(self, text: str) -> List[Dict]:
        """Extract multiple potential location information from text using various methods"""
        candidates = []

        try:
            # 1. Direct coordinate extraction
            coords = self._extract_coordinates(text)
            if coords:
                candidates.append(
                    {"lat": coords["lat"], "lon": coords["lon"], "confidence": 0.9, "source": "direct_match"}  # High confidence for direct coordinates
                )

            # 2. Use googlesearch to find location references
            search_results = search(f"location coordinates {text}", num_results=5)

            for result in search_results:
                # Look for coordinate patterns in search results
                coords = self._extract_coordinates(result)
                if coords:
                    candidates.append(
                        {"lat": coords["lat"], "lon": coords["lon"], "confidence": 0.7, "source": "search_match"}  # Lower confidence for search results
                    )

            # 3. Look for place names and landmarks
            place_patterns = [
                r"near\s+([A-Z][a-zA-Z\s]+(?:,\s*[A-Z][a-zA-Z\s]+)*)",
                r"in\s+([A-Z][a-zA-Z\s]+(?:,\s*[A-Z][a-zA-Z\s]+)*)",
                r"at\s+([A-Z][a-zA-Z\s]+(?:,\s*[A-Z][a-zA-Z\s]+)*)",
                r"([A-Z][a-zA-Z\s]+)\s+(?:district|area|region|city|town|village)",
            ]

            import re

            for pattern in place_patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    place_name = match.group(1)
                    # Use geocoding to get coordinates
                    try:
                        url = f"http://api.geonames.org/searchJSON"
                        params = {"q": place_name, "maxRows": 3, "username": CONFIG["geonames_username"], "style": "FULL"}

                        response = requests.get(url, params=params)
                        data = response.json()

                        for result in data.get("geonames", [])[:3]:  # Get top 3 matches
                            candidates.append(
                                {
                                    "lat": float(result["lat"]),
                                    "lon": float(result["lng"]),
                                    "confidence": 0.5 * float(result.get("score", 0.5)),  # Scale confidence by search score
                                    "source": "place_name",
                                    "name": result["name"],
                                    "country": result.get("countryName"),
                                }
                            )
                    except Exception as e:
                        print(f"Error geocoding place name: {e}")

            # 4. Look for relative locations
            relative_patterns = {
                r"(\d+)\s*km\s*(north|south|east|west)\s*of\s*([A-Z][a-zA-Z\s]+)": 1,
                r"between\s+([A-Z][a-zA-Z\s]+)\s+and\s+([A-Z][a-zA-Z\s]+)": 0.4,
                r"outskirts\s+of\s+([A-Z][a-zA-Z\s]+)": 0.6,
            }

            for pattern, conf_score in relative_patterns.items():
                matches = re.finditer(pattern, text)
                for match in matches:
                    # Process relative locations
                    if "km" in pattern:
                        distance = float(match.group(1))
                        direction = match.group(2)
                        place = match.group(3)
                        # Get base coordinates and adjust
                        base_coords = self._get_place_coordinates(place)
                        if base_coords:
                            adjusted_coords = self._adjust_coordinates(base_coords["lat"], base_coords["lon"], distance, direction)
                            candidates.append(
                                {
                                    "lat": adjusted_coords["lat"],
                                    "lon": adjusted_coords["lon"],
                                    "confidence": conf_score,
                                    "source": "relative_location",
                                    "reference": f"{distance}km {direction} of {place}",
                                }
                            )

            # Remove duplicates while keeping the highest confidence for each location
            unique_candidates = {}
            for candidate in candidates:
                key = f"{candidate['lat']:.4f},{candidate['lon']:.4f}"
                if key not in unique_candidates or unique_candidates[key]["confidence"] < candidate["confidence"]:
                    unique_candidates[key] = candidate

            return list(unique_candidates.values())

        except Exception as e:
            print(f"Error extracting locations from text: {e}")
            return []

    def _get_place_coordinates(self, place_name: str) -> Optional[Dict]:
        """Get coordinates for a place name using GeoNames"""
        try:
            url = f"http://api.geonames.org/searchJSON"
            params = {"q": place_name, "maxRows": 1, "username": CONFIG["geonames_username"]}

            response = requests.get(url, params=params)
            data = response.json()

            if data.get("geonames"):
                result = data["geonames"][0]
                return {"lat": float(result["lat"]), "lon": float(result["lng"])}
            return None
        except:
            return None

    def _adjust_coordinates(self, lat: float, lon: float, distance_km: float, direction: str) -> Dict:
        """Adjust coordinates based on distance and direction"""
        # Approximate degrees per km
        lat_km = 1 / 110.574  # 1 degree latitude = 110.574 km
        lon_km = 1 / (111.320 * cos(radians(lat)))  # 1 degree longitude = 111.320*cos(lat) km

        if direction.lower() == "north":
            return {"lat": lat + (distance_km * lat_km), "lon": lon}
        elif direction.lower() == "south":
            return {"lat": lat - (distance_km * lat_km), "lon": lon}
        elif direction.lower() == "east":
            return {"lat": lat, "lon": lon + (distance_km * lon_km)}
        elif direction.lower() == "west":
            return {"lat": lat, "lon": lon - (distance_km * lon_km)}

        return {"lat": lat, "lon": lon}

    def _extract_coordinates(self, text: str) -> Dict:
        """Extract coordinates from text using regex patterns"""
        import re

        # Pattern for decimal coordinates
        decimal_pattern = r"(-?\d+\.\d+)[,\s]+(-?\d+\.\d+)"

        # Pattern for DMS coordinates
        dms_pattern = r'(\d+)°\s*(\d+)\'\s*(\d+)"\s*([NS])[,\s]+(\d+)°\s*(\d+)\'\s*(\d+)"\s*([EW])'

        # Try decimal pattern
        match = re.search(decimal_pattern, text)
        if match:
            return {"lat": float(match.group(1)), "lon": float(match.group(2))}

        # Try DMS pattern
        match = re.search(dms_pattern, text)
        if match:
            lat = float(match.group(1)) + float(match.group(2)) / 60 + float(match.group(3)) / 3600
            if match.group(4) == "S":
                lat = -lat

            lon = float(match.group(5)) + float(match.group(6)) / 60 + float(match.group(7)) / 3600
            if match.group(8) == "W":
                lon = -lon

            return {"lat": lat, "lon": lon}

        return None

    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate locations based on image and location similarity"""
        unique_results = []
        seen_locations = set()

        for result in results:
            location_key = f"{result['lat']:.4f},{result['lon']:.4f}"
            if location_key not in seen_locations:
                seen_locations.add(location_key)
                unique_results.append(result)

        return unique_results

    def _rank_results(self, results: List[Dict], initial_location: Dict = None) -> List[Dict]:
        """Rank results based on multiple factors"""
        for result in results:
            score = 0.5  # Base score

            # Location proximity if initial location is known
            if initial_location:
                distance = self._calculate_distance((result["lat"], result["lon"]), (initial_location["lat"], initial_location["lon"]))
                score += max(0, 0.3 * (1 - distance / 5000))  # Up to 0.3 for proximity

            # Visual similarity score
            if "similarity_score" in result:
                score += 0.2 * result["similarity_score"]

            result["confidence"] = min(score, 1.0)

        return sorted(results, key=lambda x: x["confidence"], reverse=True)

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
