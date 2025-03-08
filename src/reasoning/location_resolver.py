from typing import Dict, List, Optional

# TODO: requests is currently unused
# import requests
from openai import OpenAI
from src.image_analysis.environment_classifier import EnvironmentType, EnvironmentInfo


class LocationResolver:
    def __init__(self, llm_api_key: str):
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=llm_api_key)
        # Define evidence weights for different environment types
        self.evidence_weights = {
            EnvironmentType.URBAN.name: {"license_plates": 0.9, "business_names": 0.8, "street_signs": 0.8, "building_info": 0.7, "landmarks": 0.6},
            EnvironmentType.SUBURBAN.name: {"street_signs": 0.9, "building_info": 0.8, "business_names": 0.7, "license_plates": 0.6, "landmarks": 0.7},
            EnvironmentType.RURAL.name: {"landmarks": 0.9, "business_names": 0.8, "geographic_features": 0.8, "license_plates": 0.5, "street_signs": 0.6},
            EnvironmentType.INDUSTRIAL.name: {"business_names": 0.9, "building_info": 0.8, "street_signs": 0.7, "license_plates": 0.6, "landmarks": 0.5},
            EnvironmentType.AIRPORT.name: {"building_info": 0.9, "business_names": 0.8, "landmarks": 0.7, "license_plates": 0.4, "street_signs": 0.5},
            EnvironmentType.COASTAL.name: {"landmarks": 0.9, "business_names": 0.8, "geographic_features": 0.8, "street_signs": 0.6, "license_plates": 0.5},
            EnvironmentType.FOREST.name: {"landmarks": 0.9, "geographic_features": 0.9, "business_names": 0.6, "license_plates": 0.3, "street_signs": 0.4},
            EnvironmentType.MOUNTAIN.name: {"landmarks": 0.9, "geographic_features": 0.9, "business_names": 0.6, "license_plates": 0.3, "street_signs": 0.4},
            EnvironmentType.DESERT.name: {"landmarks": 0.9, "geographic_features": 0.9, "business_names": 0.7, "license_plates": 0.4, "street_signs": 0.5},
            EnvironmentType.PARK.name: {"landmarks": 0.9, "business_names": 0.7, "street_signs": 0.6, "license_plates": 0.4, "building_info": 0.5},
            EnvironmentType.HIGHWAY.name: {"street_signs": 0.9, "landmarks": 0.8, "business_names": 0.7, "license_plates": 0.6, "building_info": 0.5},
        }

    def resolve_location(self, features: Dict, candidates: List[Dict], description: str, metadata: Dict = None, osv5m_prediction: Dict = None) -> Dict:
        """Enhanced location resolution considering environment type and location hints"""
        # Extract location hint from metadata or features
        location_hint = metadata.get("location_hint") if metadata else None

        # Get environment type
        env_type = features.get("environment_type", "UNKNOWN")
        env_confidence = features.get("environment_confidence", 0.5)

        # Filter candidates to prioritize those within the hinted region
        if location_hint:
            valid_candidates = [c for c in candidates if self._is_within_region(c, location_hint)]
            if valid_candidates:
                # Boost confidence for candidates within the hinted region
                for c in valid_candidates:
                    c["confidence"] = min(1.0, c.get("confidence", 0.5) * 1.5)
                candidates = valid_candidates

        # Get initial resolution with location hint and environment type
        initial_location = self._initial_resolution(features, candidates, description, location_hint)

        # If we have a location hint and the initial resolution is too general, try to find a more specific match
        if location_hint and "country" in initial_location.get("type", "").lower():
            closest_valid = self._find_closest_valid_candidate(candidates, location_hint)
            if closest_valid:
                initial_location = closest_valid

        # Apply environment-specific refinements
        if initial_location:
            refined_location = self._refine_with_environment_specific_evidence(initial_location, features, env_type, env_confidence)
            if refined_location:
                # Only use refined location if it's within the hinted region
                if location_hint and self._is_within_region(refined_location, location_hint):
                    initial_location = refined_location
                elif not location_hint:
                    initial_location = refined_location

        # If we have a location hint, ensure the final location is specific enough
        if location_hint and initial_location:
            location_parts = location_hint.split(",")
            if len(location_parts) > 1:  # We have a city or more specific hint
                if "country" in initial_location.get("type", "").lower():
                    # Try to find a more specific candidate within the hinted region
                    closest_valid = self._find_closest_valid_candidate(candidates, location_hint)
                    if closest_valid:
                        initial_location = closest_valid
                        # Boost confidence if we found a match within the hinted region
                        initial_location["confidence"] = min(1.0, initial_location["confidence"] * 1.3)

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

    def _refine_with_environment_specific_evidence(self, location: Dict, features: Dict, env_type: str, env_confidence: float) -> Optional[Dict]:
        """Refine location using environment-specific evidence weights"""
        # Get evidence weights for this environment type
        weights = self.evidence_weights.get(env_type, self.evidence_weights.get("UNKNOWN", {}))

        # Calculate weighted confidence for each type of evidence
        evidence_scores = []

        # Check license plates
        if features.get("extracted_text", {}).get("license_plate_info"):
            plate_info = features["extracted_text"]["license_plate_info"][0]
            if plate_info.get("region_name") == location["name"]:
                evidence_scores.append(weights.get("license_plates", 0.5))

        # Check business names
        if any(business in location["name"] for business in features.get("extracted_text", {}).get("business_names", [])):
            evidence_scores.append(weights.get("business_names", 0.5))

        # Check street signs
        if any(street in location["name"] for street in features.get("extracted_text", {}).get("street_signs", [])):
            evidence_scores.append(weights.get("street_signs", 0.5))

        # Check landmarks
        if any(landmark in location["name"] for landmark in features.get("landmarks", [])):
            evidence_scores.append(weights.get("landmarks", 0.5))

        # Check building info
        if any(info in location["name"] for info in features.get("extracted_text", {}).get("building_info", [])):
            evidence_scores.append(weights.get("building_info", 0.5))

        # Check geographic features
        if any(feature in location["name"] for feature in features.get("geographic_features", [])):
            evidence_scores.append(weights.get("geographic_features", 0.5))

        if not evidence_scores:
            return None

        # Calculate new confidence score
        base_confidence = sum(evidence_scores) / len(evidence_scores)
        # Adjust confidence based on environment confidence
        adjusted_confidence = base_confidence * (0.7 + (0.3 * env_confidence))

        if adjusted_confidence > location["confidence"]:
            location["confidence"] = adjusted_confidence
            return location

        return None

    def _build_reasoning_prompt(self, features: Dict, candidates: List[Dict], description: str, location_hint: str = None) -> str:
        """Build prompt for the LLM with improved entity information"""
        candidates_text = self._format_candidates(candidates)
        location_context = ""
        if location_hint:
            location_context = f"""
            CRITICAL: A specific location of '{location_hint}' has been provided.
            You MUST prioritize finding evidence that confirms or refutes this location.
            If evidence supports this location, focus on finding the most specific area WITHIN {location_hint}.
            Only suggest a different location if there is strong contradictory evidence.
            """

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
        {location_context}
        IMPORTANT: Your goal is to identify the most precise location possible - prefer specific neighborhoods or districts over general city names, and cities over countries.

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
        
        Analysis Instructions:
        1. If a specific location hint is provided, start by validating it against the evidence
        2. Look for specific districts or neighborhoods within the suggested area
        3. Use architectural styles and urban patterns to confirm the region
        4. Check if business names and street signs match the expected language/style
        5. Verify if environmental features match the local geography
        6. Consider the building density and road types for area classification
        
        NEVER default to a country-level location if a more specific location hint is provided.
        If the evidence supports the hinted location, your confidence should be at least 0.7.
        Only suggest a different location if you find strong contradictory evidence.
        
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

    def _is_within_region(self, location: Dict, region_hint: str) -> bool:
        """Check if a location is within the hinted region"""
        try:
            # Use Google Maps geocoding to get region bounds
            completion = self.client.chat.completions.create(
                model="google/gemini-2.0-flash-001",
                messages=[
                    {
                        "role": "user",
                        "content": f"""
                    Given these two locations, determine if the first is STRICTLY within the second:
                    Location 1: {location['name']} ({location['lat']}, {location['lon']})
                    Location 2: {region_hint}
                    
                    Consider:
                    1. If region_hint is a city, check if location is STRICTLY within that city's boundaries
                    2. If region_hint is a state/province, check if location is STRICTLY within that state
                    3. If region_hint is a country, check if location is STRICTLY within that country
                    4. If uncertain, return 'false'
                    
                    Return ONLY 'true' or 'false'
                    """,
                    }
                ],
            )
            result = completion.choices[0].message.content.strip().lower()
            return result == "true"
        except Exception as e:
            print(f"Error checking region containment: {e}")
            return False  # Default to False to avoid false positives

    def _find_closest_valid_candidate(self, candidates: List[Dict], region_hint: str) -> Optional[Dict]:
        """Find the closest candidate that is within the hinted region"""
        valid_candidates = []
        for candidate in candidates:
            if self._is_within_region(candidate, region_hint):
                valid_candidates.append(candidate)

        if not valid_candidates:
            return None

        # Return the highest confidence candidate among valid ones
        return max(valid_candidates, key=lambda x: x.get("confidence", 0))
