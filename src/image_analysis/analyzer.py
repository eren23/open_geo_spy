import base64
from openai import OpenAI
import requests
from PIL import Image
import io


class ImageAnalyzer:
    def __init__(self, openrouter_api_key: str, app_name: str, app_url: str):
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_api_key)
        self.headers = {"HTTP-Referer": app_url, "X-Title": app_name}

    def analyze_image(self, image_path: str):
        # Convert image to base64 or get URL
        if image_path.startswith(("http://", "https://")):
            image_url = image_path
        else:
            # Convert local image to base64
            with open(image_path, "rb") as img_file:
                image_data = base64.b64encode(img_file.read()).decode("utf-8")
                image_url = f"data:image/jpeg;base64,{image_data}"

        # Analyze with Gemini
        prompt = """Analyze this image in detail and provide the following information:
        1. Landmarks and notable buildings
        2. Architectural style
        3. Vegetation and natural features
        4. Time of day and lighting conditions
        5. Weather conditions
        6. Any visible text or signs
        7. Geographic indicators (mountains, water bodies, etc.)
        """

        completion = self.client.chat.completions.create(
            extra_headers=self.headers,
            model="google/gemini-2.0-flash-001",
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": image_url}}]}],
        )

        analysis = completion.choices[0].message.content

        # Extract structured features
        features = self._parse_analysis(analysis)
        return features, analysis

    def _parse_analysis(self, analysis: str) -> dict:
        # Parse the analysis text into structured features
        features = {"landmarks": [], "architecture_style": "", "vegetation": [], "time_of_day": "", "weather": "", "text_signs": [], "geographic_features": []}

        # Add basic NLP parsing here
        # For now, returning placeholder data
        return features
