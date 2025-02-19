from typing import List, Dict, Optional
import requests
from PIL import Image
import io
import base64
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
from overpy import Overpass
from config import CONFIG


class VisualSearchEngine:
    def __init__(self, google_api_key: Optional[str] = None, bing_api_key: Optional[str] = None):
        self.google_api_key = google_api_key
        self.bing_api_key = bing_api_key
        self.osm_api = Overpass()
        self.search_apis = {
            "google": self._search_google_images,
            "duckduckgo": self._search_duckduckgo,
            "osm": self._search_osm_images,
            "business": self._search_business_locations,
        }

    async def find_similar_locations(self, image_data: Dict, search_area: Optional[Dict] = None) -> List[Dict]:
        """Find visually similar locations using multiple search engines"""
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(search_func, image_data, search_area) for search_func in self.search_apis.values()]
            return [future.result() for future in futures]

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

    def _get_search_area(self, location: Dict) -> Optional[Dict]:
        """Get search area from location"""
        if not location or not isinstance(location, dict):
            return None

        try:
            lat = float(location.get("lat", 0))
            lon = float(location.get("lon", 0))

            # Validate coordinates
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                print(f"Invalid coordinates: lat={lat}, lon={lon}")
                return None

            # Create search area with validated coordinates
            return {
                "center": (lat, lon),
                "radius": 5000,  # 5km radius
                "bounds": {
                    "north": str(min(90, lat + 0.045)),
                    "south": str(max(-90, lat - 0.045)),
                    "east": str(min(180, lon + 0.045)),
                    "west": str(max(-180, lon - 0.045)),
                },
            }
        except (ValueError, TypeError) as e:
            print(f"Error creating search area: {str(e)}")
            return None

    def _search_google_images(self, image_data: Dict, search_area: Optional[Dict] = None) -> List[Dict]:
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

    def _search_osm_images(self, image_data: Dict, search_area: Optional[Dict] = None) -> List[Dict]:
        """Search OpenStreetMap for images and locations"""
        if not search_area or not isinstance(search_area, dict):
            print("Invalid or missing search area")
            return []

        try:
            # Validate and extract coordinates
            bounds = search_area.get("bounds", {})
            if not isinstance(bounds, dict) or not all(k in bounds for k in ["south", "west", "north", "east"]):
                print("Missing or invalid bounds in search area")
                return []

            # Convert coordinates to strings and validate
            try:
                coords = {}
                for key in ["south", "west", "north", "east"]:
                    value = bounds.get(key)
                    if value is None:
                        print(f"Missing {key} coordinate")
                        return []
                    try:
                        float_val = float(value)
                        if key in ["south", "north"] and not -90 <= float_val <= 90:
                            print(f"Invalid {key} coordinate: {float_val}")
                            return []
                        if key in ["east", "west"] and not -180 <= float_val <= 180:
                            print(f"Invalid {key} coordinate: {float_val}")
                            return []
                        coords[key] = str(float_val)
                    except (ValueError, TypeError):
                        print(f"Invalid {key} coordinate value: {value}")
                        return []
            except Exception as e:
                print(f"Error validating coordinates: {str(e)}")
                return []

            # Build the query using string formatting
            try:
                query = """
                [out:json][timeout:25];
                (
                  way["image"]({south},{west},{north},{east});
                  node["image"]({south},{west},{north},{east});
                  relation["image"]({south},{west},{north},{east});
                );
                out body;
                >;
                out skel qt;
                """.format(
                    **coords
                )
            except Exception as e:
                print(f"Error formatting query: {str(e)}")
                return []

            results = []
            try:
                response = self.osm_api.query(query)

                # Process ways with images
                if hasattr(response, "ways"):
                    for way in response.ways:
                        if hasattr(way, "tags") and way.tags.get("image") and way.nodes:
                            try:
                                # Calculate center point of way
                                valid_nodes = [n for n in way.nodes if hasattr(n, "lat") and hasattr(n, "lon")]
                                if valid_nodes:
                                    lat = sum(float(n.lat) for n in valid_nodes) / len(valid_nodes)
                                    lon = sum(float(n.lon) for n in valid_nodes) / len(valid_nodes)

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
                                print(f"Error processing way {way.id}: {str(e)}")
                                continue

                # Process nodes with images
                if hasattr(response, "nodes"):
                    for node in response.nodes:
                        if hasattr(node, "tags") and node.tags.get("image"):
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
                                print(f"Error processing node {node.id}: {str(e)}")
                                continue

                # Search for nearby POIs
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
                    **coords
                )

                try:
                    poi_response = self.osm_api.query(poi_query)

                    if hasattr(poi_response, "nodes"):
                        for node in poi_response.nodes:
                            if hasattr(node, "lat") and hasattr(node, "lon") and hasattr(node, "tags"):
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
                                    print(f"Error processing POI {node.id}: {str(e)}")
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
                # Try with a smaller search area
                center = search_area.get("center", [0, 0])
                smaller_area = {
                    "center": center,
                    "radius": search_area.get("radius", 5000) / 2,
                    "bounds": {
                        "north": (float(coords["north"]) + float(center[0])) / 2,
                        "south": (float(coords["south"]) + float(center[0])) / 2,
                        "east": (float(coords["east"]) + float(center[1])) / 2,
                        "west": (float(coords["west"]) + float(center[1])) / 2,
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

    def _search_business_locations(self, image_data: Dict, search_area: Dict = None) -> List[Dict]:
        """Search for specific businesses and landmarks"""
        if not search_area:
            return []

        try:
            # Extract business names from image text
            business_names = self._extract_business_names(image_data)
            if not business_names:
                return []

            results = []
            for business in business_names:
                # Search Google Places API
                if self.google_api_key:
                    places_results = self._search_google_places(business, search_area)
                    results.extend(places_results)

                # Search OpenStreetMap for businesses
                osm_results = self._search_osm_businesses(business, search_area)
                results.extend(osm_results)

                # Search local business directories
                local_results = self._search_local_directories(business, search_area)
                results.extend(local_results)

            return results
        except Exception as e:
            print(f"Error searching business locations: {e}")
            return []

    def _search_google_places(self, business_name: str, search_area: Dict) -> List[Dict]:
        """Search Google Places API for business locations"""
        try:
            url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
            location = f"{search_area['center'][0]},{search_area['center'][1]}"
            radius = min(5000, search_area.get("radius", 5000))  # Max 5km radius

            params = {"query": business_name, "location": location, "radius": radius, "key": self.google_api_key}

            response = requests.get(url, params=params)
            data = response.json()

            results = []
            for place in data.get("results", []):
                results.append(
                    {
                        "source": "google_places",
                        "name": place["name"],
                        "lat": place["geometry"]["location"]["lat"],
                        "lon": place["geometry"]["location"]["lng"],
                        "type": "business",
                        "confidence": 0.9 if business_name.lower() in place["name"].lower() else 0.7,
                        "metadata": {"address": place.get("formatted_address"), "place_id": place["place_id"], "types": place.get("types", [])},
                    }
                )

            return results
        except Exception as e:
            print(f"Error searching Google Places: {e}")
            return []

    def _search_osm_businesses(self, business_name: str, search_area: Dict) -> List[Dict]:
        """Search OpenStreetMap for businesses"""
        try:
            query = f"""
            [out:json][timeout:25];
            (
              node["name"~"{business_name}",i]
                ({search_area['bbox']['south']},{search_area['bbox']['west']},
                 {search_area['bbox']['north']},{search_area['bbox']['east']});
              way["name"~"{business_name}",i]
                ({search_area['bbox']['south']},{search_area['bbox']['west']},
                 {search_area['bbox']['north']},{search_area['bbox']['east']});
              relation["name"~"{business_name}",i]
                ({search_area['bbox']['south']},{search_area['bbox']['west']},
                 {search_area['bbox']['north']},{search_area['bbox']['east']});
            );
            out body;
            >;
            out skel qt;
            """

            results = []
            response = self.osm_api.query(query)

            for node in response.nodes:
                results.append(
                    {
                        "source": "osm",
                        "name": node.tags.get("name", business_name),
                        "lat": float(node.lat),
                        "lon": float(node.lon),
                        "type": "business",
                        "confidence": 0.8 if business_name.lower() in node.tags.get("name", "").lower() else 0.6,
                        "metadata": {"osm_id": node.id, "tags": dict(node.tags)},
                    }
                )

            return results
        except Exception as e:
            print(f"Error searching OSM businesses: {e}")
            return []

    def _search_local_directories(self, business_name: str, search_area: Dict) -> List[Dict]:
        """Search local business directories"""
        try:
            # Implement searches for:
            # - Yelp API
            # - Yellow Pages
            # - Local chamber of commerce
            # - Regional business directories
            return []  # Placeholder for future implementation
        except Exception as e:
            print(f"Error searching local directories: {e}")
            return []

    def _verify_business_locations(self, results: List[Dict], initial_location: Dict) -> List[Dict]:
        """Verify and refine business locations"""
        verified_results = []

        for result in results:
            try:
                # Skip if not a business result
                if result.get("type") != "business":
                    verified_results.append(result)
                    continue

                # Get additional business details
                if result["source"] == "google_places" and self.google_api_key:
                    details = self._get_google_place_details(result["metadata"]["place_id"])
                    if details:
                        result["metadata"].update(details)
                        result["confidence"] = min(1.0, result["confidence"] + 0.1)

                # Verify with street view if available
                if self.google_api_key:
                    has_streetview = self._check_street_view(result["lat"], result["lon"])
                    if has_streetview:
                        result["confidence"] = min(1.0, result["confidence"] + 0.1)

                # Cross-reference with other sources
                if self._cross_reference_location(result, initial_location):
                    result["confidence"] = min(1.0, result["confidence"] + 0.1)

                verified_results.append(result)
            except Exception as e:
                print(f"Error verifying result: {e}")
                verified_results.append(result)

        return verified_results

    def _get_google_place_details(self, place_id: str) -> Optional[Dict]:
        """Get detailed information about a place from Google Places API"""
        try:
            url = "https://maps.googleapis.com/maps/api/place/details/json"
            params = {"place_id": place_id, "fields": "name,formatted_address,formatted_phone_number,website,opening_hours,photos", "key": self.google_api_key}

            response = requests.get(url, params=params)
            data = response.json()

            if data.get("result"):
                return {
                    "full_address": data["result"].get("formatted_address"),
                    "phone": data["result"].get("formatted_phone_number"),
                    "website": data["result"].get("website"),
                    "hours": data["result"].get("opening_hours", {}).get("weekday_text", []),
                    "has_photos": bool(data["result"].get("photos", [])),
                }
            return None
        except Exception as e:
            print(f"Error getting place details: {e}")
            return None

    def _check_street_view(self, lat: float, lon: float) -> bool:
        """Check if Street View imagery is available for location"""
        try:
            url = "https://maps.googleapis.com/maps/api/streetview/metadata"
            params = {"location": f"{lat},{lon}", "key": self.google_api_key}

            response = requests.get(url, params=params)
            data = response.json()

            return data.get("status") == "OK"
        except Exception:
            return False

    def _cross_reference_location(self, result: Dict, initial_location: Dict) -> bool:
        """Cross-reference location with other data sources"""
        try:
            # Check if location is within expected area
            if initial_location:
                distance = self._calculate_distance((result["lat"], result["lon"]), (initial_location["lat"], initial_location["lon"]))
                if distance > 5000:  # More than 5km away
                    return False

            # Implement additional cross-referencing:
            # - Check against government databases
            # - Verify with business registration records
            # - Compare with map data
            return True
        except Exception:
            return False
