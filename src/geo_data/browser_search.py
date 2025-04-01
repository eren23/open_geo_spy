from typing import Dict, List, Optional
from browser_use import Agent
import google.generativeai as genai
from config import CONFIG
from langchain_google_genai import ChatGoogleGenerativeAI
import re


class BrowserSearchAdapter:
    def __init__(self):
        genai.configure(api_key=CONFIG.GEMINI_API_KEY)
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=CONFIG.GEMINI_API_KEY, temperature=0.7)

    async def search_locations(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Search for locations using browser-use with Gemini
        """
        try:
            # Create search task with specific location context
            search_task = (
                f"Find exact location and coordinates for: {query}. "
                "Search specifically in Google Maps and business directories. "
                "Return in format: Name: [address], Coordinates: [lat], [lon]"
            )

            agent = Agent(task=search_task, llm=self.llm)

            # Run the agent
            result = await agent.run()

            # Extract location from agent's response
            if isinstance(result, dict) and result.get("done"):
                location = self._parse_location_from_text(result["done"]["text"])
                return [location] if location else []

            return []

        except Exception as e:
            print(f"Browser search error: {e}")
            return []

    def _parse_location_from_text(self, text: str) -> Optional[Dict]:
        """Parse location data from agent's text response"""
        try:
            # for error mitigation, probably a bad idea
            coord_patterns = [
                r"(\d+\.\d+),\s*(\d+\.\d+)",  # decimal format
                r"(\d+)°\s*(\d+)'\s*([0-9.]+)\"?\s*[NS],?\s*(\d+)°\s*(\d+)'\s*([0-9.]+)\"?\s*[EW]",  # DMS format
            ]

            for pattern in coord_patterns:
                match = re.search(pattern, text)
                if match:
                    if len(match.groups()) == 2:  # decimal format
                        lat, lon = map(float, match.groups())
                    else:  # DMS format
                        lat_d, lat_m, lat_s, lon_d, lon_m, lon_s = map(float, match.groups())
                        lat = lat_d + lat_m / 60 + lat_s / 3600
                        lon = lon_d + lon_m / 60 + lon_s / 3600

                    name_parts = text.split("Coordinates:")[0].split(",")
                    name = name_parts[0].strip()
                    address = ",".join(name_parts[1:]).strip() if len(name_parts) > 1 else ""

                    return {
                        "source": "browser",
                        "name": name,
                        "lat": lat,
                        "lon": lon,
                        "confidence": 0.7,
                        "type": "browser_match",
                        "metadata": {"description": text, "source": "browser-use agent", "address": address},
                    }
            return None

        except Exception as e:
            print(f"Error parsing location data: {e}")
            return None
