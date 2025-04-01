import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Centralized configuration class"""

    def __init__(self):
        self.OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
        self.GEONAMES_USERNAME = os.getenv("GEONAMES_USERNAME")
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        self.BING_API_KEY = os.getenv("BING_API_KEY")
        self.GOOGLE_SEARCH_CX = os.getenv("GOOGLE_SEARCH_CX")
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        self.APP_NAME = "GeoLocator"
        self.APP_URL = os.getenv("APP_URL", "https://your-app-url.com")
        self.IMAGE_DIR = os.getenv("IMAGES_DIR", "/app/images")
        self.GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
        self.USE_BROWSER = os.getenv("USE_BROWSER", "false").lower() == "true"
        self.BROWSER_API_KEY = os.getenv("BROWSER_API_KEY")


CONFIG = Config()
