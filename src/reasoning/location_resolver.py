from typing import Dict, List
import requests
from openai import OpenAI


class LocationResolver:
    def __init__(self, llm_api_key: str):
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=llm_api_key)

    def resolve_location(self, image_features: Dict, location_candidates: List[Dict], image_description: str, location_hint: str = None) -> Dict:
        """Use LLM to reason about the most likely location"""
        prompt = self._build_reasoning_prompt(image_features, location_candidates, image_description, location_hint)

        # Call LLM API
        completion = self.client.chat.completions.create(model="google/gemini-2.0-flash-001", messages=[{"role": "user", "content": prompt}])

        # Parse response
        response = completion.choices[0].message.content
        return self._parse_llm_response(response, location_candidates)

    def _build_reasoning_prompt(self, features: Dict, candidates: List[Dict], description: str, location_hint: str = None) -> str:
        """Build prompt for the LLM"""
        candidates_text = self._format_candidates(candidates)
        location_context = f"\nProvided Location Context: {location_hint}" if location_hint else ""

        return f"""
        Given an image with the following description and features, determine the most likely location.{location_context}

        Image Description: {description}
        
        Extracted Features:
        - Landmarks: {', '.join(features['landmarks']) if features['landmarks'] else 'None detected'}
        - Architecture: {features['architecture_style'] or 'Unknown'}
        - Time of Day: {features['time_of_day'] or 'Unknown'}
        - Vegetation: {', '.join(features['vegetation']) if features['vegetation'] else 'None detected'}
        - Weather: {features['weather'] or 'Unknown'}
        - Geographic Features: {', '.join(features['geographic_features']) if features['geographic_features'] else 'None detected'}
        
        Potential Locations:
        {candidates_text}
        
        Analyze the evidence and determine the most likely location.
        If a location context was provided, give higher confidence to matches within or near that area.
        
        Return your response in this format:
        Location: [name]
        Coordinates: [lat], [lon]
        Confidence: [0-1]
        Reasoning: [explanation]
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
