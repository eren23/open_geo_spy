import os
from dotenv import load_dotenv

load_dotenv()

CONFIG = {
    "openrouter_api_key": os.getenv("OPENROUTER_API_KEY"),
    "geonames_username": os.getenv("GEONAMES_USERNAME"),
    "app_name": "GeoLocator",
    "app_url": "https://your-app-url.com",
    "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
    "BING_API_KEY": os.getenv("BING_API_KEY"),
    "GOOGLE_SEARCH_CX": os.getenv("GOOGLE_SEARCH_CX"),  # Custom Search Engine ID
    "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
}
