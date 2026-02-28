"""Type-safe configuration using Pydantic BaseSettings.

All settings are loaded from environment variables with validation at startup.
Nested settings group related config together for clarity.
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings

# Load .env into os.environ so flat env var mappings work
load_dotenv()


class Environment(str, Enum):
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


# --- Nested setting groups ---


class LLMSettings(BaseModel):
    """OpenRouter LLM configuration."""

    api_key: str = Field(default="", description="OpenRouter API key")
    base_url: str = "https://openrouter.ai/api/v1"

    # Model tiers
    fast_model: str = "google/gemini-2.5-flash"
    reasoning_model: str = "google/gemini-2.5-pro"
    heavy_model: str = "google/gemini-3.1-pro-preview-20260219"
    verification_model: str = "qwen/qwen3-vl-235b-a22b-instruct"
    budget_model: str = "qwen/qwen3-vl-8b-instruct"

    temperature: float = 0.1
    max_retries: int = 3


class BrowserSettings(BaseModel):
    """Browser automation and stealth configuration."""

    enabled: bool = False
    api_key: str = ""
    pool_size: int = 3
    request_delay_min: float = 2.0
    request_delay_max: float = 8.0

    # Stealth
    enable_stealth: bool = True
    enable_canvas_noise: bool = True
    enable_webgl_spoof: bool = True
    tls_impersonation: str = "chrome131"

    # Proxy
    proxy_url: Optional[str] = None

    # CAPTCHA
    capsolver_api_key: str = ""


class GeoSettings(BaseModel):
    """Geolocation service configuration."""

    geonames_username: str = ""
    serper_api_key: str = ""
    mapillary_access_token: str = ""
    brave_api_key: str = ""
    searxng_url: str = ""
    search_providers: list[str] = ["serper"]  # Options: serper, brave, searxng


class MLSettings(BaseModel):
    """ML model configuration."""

    enable_geoclip: bool = True
    enable_streetclip: bool = True
    enable_pigeon: bool = False  # Requires additional setup
    enable_visual_verification: bool = True
    device: str = "cpu"  # "cpu", "cuda", "mps"
    cache_dir: str = os.path.expanduser("~/.cache/open_geo_spy/models")

    # Per-model weights for ensemble scoring
    model_weights: dict[str, float] = {
        "GeoCLIP": 1.0,
        "StreetCLIP": 1.0,
        "VLM Geo": 1.5,
    }


class CacheSettings(BaseModel):
    """Caching configuration."""

    enabled: bool = True
    backend: str = "memory"  # "memory" or "disk"
    disk_path: str = os.path.expanduser("~/.cache/open_geo_spy/api_cache")
    max_memory_entries: int = 1000

    # TTLs per source (seconds)
    serper_ttl: int = 7200  # 2 hours
    osm_ttl: int = 86400  # 24 hours
    browser_ttl: int = 1800  # 30 minutes
    brave_ttl: int = 7200
    searxng_ttl: int = 3600


class CalibrationSettings(BaseModel):
    """Confidence calibration configuration."""

    enabled: bool = False
    data_path: str = os.path.expanduser("~/.cache/open_geo_spy/calibration.json")


class APISettings(BaseModel):
    """API server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]
    rate_limit_rpm: int = 60
    max_upload_size_mb: int = 50


# --- Main settings ---


class Settings(BaseSettings):
    """Root settings loaded from environment variables.

    Environment variables are mapped with these prefixes:
      - LLM__*        -> llm.*
      - BROWSER__*    -> browser.*
      - GEO__*        -> geo.*
      - ML__*         -> ml.*
      - API__*        -> api.*
    Or flat env vars for common ones (see aliases below).
    """

    environment: Environment = Environment.DEV
    debug: bool = False
    app_name: str = "OpenGeoSpy"
    image_dir: str = os.getenv("IMAGES_DIR", "./images")

    # Nested groups
    llm: LLMSettings = LLMSettings()
    browser: BrowserSettings = BrowserSettings()
    geo: GeoSettings = GeoSettings()
    ml: MLSettings = MLSettings()
    api: APISettings = APISettings()
    cache: CacheSettings = CacheSettings()
    calibration: CalibrationSettings = CalibrationSettings()

    model_config = {
        "env_prefix": "",
        "env_nested_delimiter": "__",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @field_validator("image_dir")
    @classmethod
    def ensure_image_dir(cls, v: str) -> str:
        os.makedirs(v, exist_ok=True)
        return v

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Support flat env vars for backward compatibility
        self._load_flat_env_vars()

    def _load_flat_env_vars(self):
        """Map legacy flat env vars to nested settings."""
        mappings = {
            "OPENROUTER_API_KEY": ("llm", "api_key"),
            "GEONAMES_USERNAME": ("geo", "geonames_username"),
            "SERPER_API_KEY": ("geo", "serper_api_key"),
            "USE_BROWSER": ("browser", "enabled"),
            "BROWSER_API_KEY": ("browser", "api_key"),
            "MAPILLARY_ACCESS_TOKEN": ("geo", "mapillary_access_token"),
        }
        for env_var, (group, field) in mappings.items():
            val = os.getenv(env_var)
            if val is not None:
                group_obj = getattr(self, group)
                current = getattr(group_obj, field)
                # Only override if the nested value is empty/default
                if not current or current == "" or current is False:
                    if field == "enabled":
                        val = val.lower() == "true"
                    object.__setattr__(group_obj, field, val)


@lru_cache
def get_settings() -> Settings:
    """Cached singleton settings instance."""
    return Settings()
