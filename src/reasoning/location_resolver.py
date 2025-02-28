from typing import Dict, List, Optional

# TODO: requests is currently unused
# import requests
from openai import OpenAI


class LocationResolver:
    def __init__(self, llm_api_key: str):
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=llm_api_key)

    def resolve_location(self, features: Dict, candidates: List[Dict], description: str, metadata: Dict = None, osv5m_prediction: Dict = None) -> Dict:
        """Enhanced location resolution considering multiple models"""
        # Get initial resolution
        initial_location = self._initial_resolution(features, candidates, description)

        # If OSV5M prediction is highly confident, bias towards it
        if osv5m_prediction and osv5m_prediction.get("confidence", 0) > 0.8:
            # Merge OSV5M prediction with initial resolution
            initial_location = self._merge_predictions(initial_location, osv5m_prediction)

        # Continue with existing refinement steps
        if initial_location and initial_location["confidence"] > 0.7:
            refined_location = self._refine_with_business_verification(initial_location, features)
            if refined_location:
                initial_location = refined_location

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
        """Build prompt for the LLM with improved entity information"""
        candidates_text = self._format_candidates(candidates)
        location_context = f"\nProvided Location Context: {location_hint}" if location_hint else ""

        # Extract entity-location associations
        entity_locations = features.get("entity_locations", {})
        entity_location_text = ""
        if entity_locations:
            entity_location_text = "\nEntity-Location Associations:\n"
            for entity, location in entity_locations.items():
                entity_location_text += f"- {entity} → {location}\n"

        # Extract environmental features
        env_features = {
            "terrain": features.get("terrain_type", []),
            "water": features.get("water_bodies", []),
            "sky": features.get("sky_features", []),
            "buildings": features.get("building_density", "unknown"),
            "roads": features.get("road_types", []),
            "vegetation": features.get("vegetation_density", "unknown"),
        }

        # Format business information with more detail
        business_details = []
        for business in features.get("extracted_text", {}).get("business_names", []):
            business_type = features.get("entity_types", {}).get(business, "unknown")
            business_details.append(f"{business} (Type: {business_type})")

        # Format street information
        street_details = []
        for street in features.get("extracted_text", {}).get("street_signs", []):
            street_details.append(street)

        # Extract license plate information with region details
        license_plate_info = features.get("extracted_text", {}).get("license_plate_info", [])

        # Format license plate information
        plate_details = []
        for plate_info in license_plate_info:
            plate_details.append(
                f"Plate: {plate_info['plate_number']} "
                f"(Country: {plate_info.get('country', 'Unknown')}, "
                f"Region: {plate_info.get('region_name', 'Unknown')} [{plate_info.get('region_code', 'Unknown')}])"
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
        
        Business Information:
        {chr(10).join([f"- {business}" for business in business_details]) if business_details else 'No businesses detected'}
        
        Street Information:
        {chr(10).join([f"- {street}" for street in street_details]) if street_details else 'No streets detected'}
        
        Building Information:
        {chr(10).join([f"- {building}" for building in features.get('extracted_text', {}).get('building_info', [])]) if features.get('extracted_text', {}).get('building_info', []) else 'No building info detected'}
        {entity_location_text}
        
        Other Features:
        - Landmarks: {', '.join(features['landmarks']) if features.get('landmarks') else 'None detected'}
        - Architecture: {features.get('architecture_style') or 'Unknown'}
        - Time of Day: {features.get('time_of_day') or 'Unknown'}
        - Weather: {features.get('weather') or 'Unknown'}
        
        Potential Locations:
        {candidates_text}
        
        Please analyze all evidence to determine the most specific location possible, considering:
        1. License plate region codes - these provide exact city information (e.g. 16 = Bursa)
        2. Business names and their known locations
        3. Street names and their city context
        4. Specific districts or neighborhoods within the identified city
        5. Building numbers and addresses
        6. Match between environmental features and local geography
        7. Architectural styles and building patterns
        8. Language and script used in signs
        9. Road types and infrastructure style
        10. Vegetation patterns and climate indicators
        
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

    def _merge_predictions(self, vlm_pred: Dict, osv5m_pred: Dict) -> Dict:
        """Merge predictions from different models"""
        # If VLM is very confident, keep it
        if vlm_pred.get("confidence", 0) > 0.85:
            return vlm_pred

        # If OSV5M is very confident, prefer it
        if osv5m_pred.get("confidence", 0) > 0.85:
            return osv5m_pred

        # Otherwise merge them
        return {
            "name": f"{vlm_pred['name']} / {osv5m_pred['name']}",
            "lat": (vlm_pred["lat"] + osv5m_pred["lat"]) / 2,
            "lon": (vlm_pred["lon"] + osv5m_pred["lon"]) / 2,
            "confidence": max(vlm_pred.get("confidence", 0), osv5m_pred.get("confidence", 0)),
            "source": "hybrid",
            "type": "merged_prediction",
        }
