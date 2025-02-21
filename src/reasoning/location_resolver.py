from typing import Dict, List, Optional
import requests
from openai import OpenAI
from geo_data.location_refiner import LocationRefiner


class LocationResolver:
    def __init__(self, llm_api_key: str):
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=llm_api_key)
        self.location_refiner = LocationRefiner()

    def resolve_location(self, features: Dict, candidates: List[Dict], image_description: str, location_hint: str = None) -> Dict:
        """Use LLM to reason about the most likely location through multiple refinement steps"""
        # Get initial location
        initial_location = self._initial_resolution(features, candidates, image_description, location_hint)

        if not initial_location:
            return {"success": False, "error": "Could not determine initial location"}

        # If we have a city-level match with high confidence, try to refine it
        if initial_location["confidence"] > 0.7 and "," in initial_location["name"]:
            refined_locations = self.location_refiner.refine_location(initial_location, {"features": features})

            if refined_locations:
                best_match = refined_locations[0]
                return {
                    "name": best_match.name,
                    "address": best_match.address,
                    "lat": best_match.lat,
                    "lon": best_match.lon,
                    "confidence": best_match.confidence,
                    "alternative_locations": refined_locations[1:],
                    "reasoning": f"Refined from {initial_location['name']} based on visible landmarks and features",
                }

        return initial_location

    def _initial_resolution(self, features: Dict, candidates: List[Dict], description: str, location_hint: str = None) -> Dict:
        """Initial location resolution using all available information"""
        prompt = self._build_reasoning_prompt(features, candidates, description, location_hint)
        completion = self.client.chat.completions.create(model="google/gemini-2.0-flash-001", messages=[{"role": "user", "content": prompt}])
        return self._parse_llm_response(completion.choices[0].message.content, candidates)

    def _refine_with_business_verification(self, location: Dict, features: Dict) -> Optional[Dict]:
        """Refine location using business and landmark verification"""
        businesses = features.get("extracted_text", {}).get("business_names", [])
        landmarks = features.get("landmarks", [])
        license_plates = features.get("extracted_text", {}).get("license_plates", [])

        if not (businesses or landmarks or license_plates):
            return None

        verification_prompt = f"""
        Given a potential location at {location['name']} ({location['lat']}, {location['lon']}),
        verify and refine this location using the following information:
        
        Businesses: {', '.join(businesses)}
        Landmarks: {', '.join(landmarks)}
        License Plates: {', '.join(license_plates)}
        
        Environmental Context:
        - Terrain: {', '.join(features.get('terrain_type', []))}
        - Water Bodies: {', '.join(features.get('water_bodies', []))}
        - Building Density: {features.get('building_density', 'unknown')}
        - Road Types: {', '.join(features.get('road_types', []))}
        - Vegetation: {features.get('vegetation_density', 'unknown')}
        
        Tasks:
        1. Verify if the environmental features match the proposed location
        2. Check if license plates match the region's format
        3. Search for the businesses/landmarks near the coordinates
        4. Verify if they exist in the expected location
        5. If found, provide their exact coordinates
        6. Calculate confidence based on all matches
        
        Return your response in this format:
        Location: [refined name with street/area]
        Coordinates: [lat], [lon]
        Confidence: [0-1]
        Reasoning: [explanation including all verification steps]
        """

        completion = self.client.chat.completions.create(model="google/gemini-2.0-flash-001", messages=[{"role": "user", "content": verification_prompt}])

        refined = self._parse_llm_response(completion.choices[0].message.content, [location])
        return refined if refined["confidence"] > location["confidence"] else None

    def _refine_with_street_details(self, location: Dict, features: Dict) -> Optional[Dict]:
        """Refine location using street-level details"""
        street_signs = features.get("extracted_text", {}).get("street_signs", [])
        building_info = features.get("extracted_text", {}).get("building_info", [])

        if not street_signs and not building_info:
            return None

        street_prompt = f"""
        Given a location at {location['name']} ({location['lat']}, {location['lon']}),
        refine it using these street-level details:
        
        Street Signs: {', '.join(street_signs)}
        Building Info: {', '.join(building_info)}
        
        Tasks:
        1. Search for these street names and building numbers in the area
        2. Cross-reference with the existing location
        3. Provide the most precise coordinates possible
        4. Update confidence based on match accuracy
        
        Return your response in this format:
        Location: [precise address or intersection]
        Coordinates: [lat], [lon]
        Confidence: [0-1]
        Reasoning: [explanation including street matches]
        """

        completion = self.client.chat.completions.create(model="google/gemini-2.0-flash-001", messages=[{"role": "user", "content": street_prompt}])

        refined = self._parse_llm_response(completion.choices[0].message.content, [location])
        return refined if refined["confidence"] > location["confidence"] else None

    def _build_reasoning_prompt(self, features: Dict, candidates: List[Dict], description: str, location_hint: str = None) -> str:
        """Build prompt for the LLM"""
        candidates_text = self._format_candidates(candidates)
        location_context = f"\nProvided Location Context: {location_hint}" if location_hint else ""

        # Extract environmental features
        env_features = {
            "terrain": features.get("terrain_type", []),
            "water": features.get("water_bodies", []),
            "sky": features.get("sky_features", []),
            "buildings": features.get("building_density", "unknown"),
            "roads": features.get("road_types", []),
            "vegetation": features.get("vegetation_density", "unknown"),
        }

        # Extract license plate information with region details
        license_plates = features.get("extracted_text", {}).get("license_plates", [])
        license_plate_info = features.get("extracted_text", {}).get("license_plate_info", [])

        # Format license plate information
        plate_details = []
        for plate_info in license_plate_info:
            plate_details.append(
                f"Plate: {plate_info['plate_number']} "
                f"(Country: {plate_info['country']}, "
                f"Region: {plate_info['region_name']} [{plate_info['region_code']}])"
            )

        return f"""
        Given an image with the following description and features, determine the MOST SPECIFIC location possible.
        IMPORTANT: Prefer city/district level locations over country-level. If a license plate indicates a specific city (e.g. 16 for Bursa), that should be your primary location indicator.{location_context}

        Image Description: {description}
        
        Environmental Features:
        - Terrain Type: {', '.join(env_features['terrain']) if env_features['terrain'] else 'Unknown'}
        - Water Bodies: {', '.join(env_features['water']) if env_features['water'] else 'None detected'}
        - Sky Conditions: {', '.join(env_features['sky']) if env_features['sky'] else 'Unknown'}
        - Building Density: {env_features['buildings']}
        - Road Types: {', '.join(env_features['roads']) if env_features['roads'] else 'Unknown'}
        - Vegetation Density: {env_features['vegetation']}
        
        License Plate Analysis:
        {chr(10).join(plate_details) if plate_details else 'No license plates detected'}
        
        Other Features:
        - Landmarks: {', '.join(features['landmarks']) if features['landmarks'] else 'None detected'}
        - Architecture: {features['architecture_style'] or 'Unknown'}
        - Time of Day: {features['time_of_day'] or 'Unknown'}
        - Weather: {features['weather'] or 'Unknown'}
        - Business Names: {', '.join(features.get('extracted_text', {}).get('business_names', []))}
        - Street Signs: {', '.join(features.get('extracted_text', {}).get('street_signs', []))}
        - Building Info: {', '.join(features.get('extracted_text', {}).get('building_info', []))}
        
        Potential Locations:
        {candidates_text}
        
        Please analyze all evidence to determine the most specific location possible, considering:
        1. License plate region codes - these provide exact city information (e.g. 16 = Bursa)
        2. Specific districts or neighborhoods within the identified city
        3. Street names and building numbers
        4. Business locations and landmarks
        5. Match between environmental features and local geography
        6. Architectural styles and building patterns
        7. Language and script used in signs
        8. Road types and infrastructure style
        9. Vegetation patterns and climate indicators
        
        If a location context was provided, prioritize matches within that area.
        License plate region codes should be your primary indicator for city-level location.
        Only fall back to broader regions if no specific location can be determined.
        
        Return your response in this format:
        Location: [most specific name - include city AND district/neighborhood if possible]
        Coordinates: [lat], [lon]
        Confidence: [0-1]
        Reasoning: [detailed explanation prioritizing specific location indicators]
        """

    def _format_candidates(self, candidates: List[Dict]) -> str:
        """Format location candidates for the prompt"""
        if not candidates:
            return "No location candidates found."

        formatted = []
        for i, candidate in enumerate(candidates, 1):
            formatted.append(
                f"{i}. {candidate['name']} ({candidate['lat']}, {candidate['lon']})\n" f"   Type: {candidate['type']}\n" f"   Source: {candidate['source']}"
            )

        return "\n".join(formatted)

    def _parse_llm_response(self, response: str, candidates: List[Dict]) -> Dict:
        """Parse LLM response into structured location data"""
        # Default to the highest confidence candidate if parsing fails
        fallback = max(candidates, key=lambda x: x.get("confidence", 0)) if candidates else {"name": "Unknown", "lat": 0.0, "lon": 0.0, "confidence": 0.0}

        try:
            # Extract location details from response
            lines = response.strip().split("\n")
            result = {}

            for line in lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().lower()
                    value = value.strip()

                    if key == "location":
                        result["name"] = value
                    elif key == "coordinates":
                        try:
                            lat, lon = map(float, value.replace(" ", "").split(","))
                            result["lat"] = lat
                            result["lon"] = lon
                        except:
                            result["lat"] = fallback["lat"]
                            result["lon"] = fallback["lon"]
                    elif key == "confidence":
                        try:
                            result["confidence"] = float(value)
                        except:
                            result["confidence"] = fallback["confidence"]
                    elif key == "reasoning":
                        result["reasoning"] = value

            # Ensure all required fields are present
            required_fields = {"name", "lat", "lon", "confidence"}
            if not all(field in result for field in required_fields):
                return fallback

            return result

        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return fallback
