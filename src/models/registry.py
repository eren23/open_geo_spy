"""Hot-pluggable model registry.

Models register themselves via ``@ModelRegistry.register`` and are
discovered at startup by the ML ensemble agent.
"""

from __future__ import annotations

from typing import Optional, Type

from loguru import logger

from src.config.settings import Settings
from src.models.base import GeoModel, ModelCapability, ModelInfo


class ModelRegistry:
    """Singleton registry for geo models."""

    _models: dict[str, Type[GeoModel]] = {}
    _instances: dict[str, GeoModel] = {}

    @classmethod
    def register(cls, model_cls: Type[GeoModel]) -> Type[GeoModel]:
        """Class decorator to register a GeoModel implementation.

        Usage::

            @ModelRegistry.register
            class MyModel(GeoModel):
                ...
        """
        # Instantiate temporarily to get info
        try:
            tmp = object.__new__(model_cls)
            # Some models need __init__ to set defaults, but info() should
            # work at the class level or with minimal init.
            name = model_cls.__name__
        except Exception:
            name = model_cls.__name__

        cls._models[name] = model_cls
        logger.debug("Registered model: {}", name)
        return model_cls

    @classmethod
    def get_all(cls) -> dict[str, Type[GeoModel]]:
        """Return all registered model classes."""
        return dict(cls._models)

    @classmethod
    def get_enabled(cls, settings: Settings) -> list[GeoModel]:
        """Return instantiated models that are enabled in settings.

        Uses the ``ml`` settings block to decide which models to load.
        """
        enabled: list[GeoModel] = []

        model_toggle = {
            "GeoCLIPAdapter": settings.ml.enable_geoclip,
            "StreetCLIPAdapter": settings.ml.enable_streetclip,
            "VLMGeoAdapter": True,  # Always on (API-based)
        }

        for name, model_cls in cls._models.items():
            is_on = model_toggle.get(name, True)
            if not is_on:
                logger.debug("Model {} disabled by settings", name)
                continue

            if name not in cls._instances:
                try:
                    instance = model_cls(settings=settings)
                    cls._instances[name] = instance
                    logger.info("Instantiated model: {}", name)
                except Exception as e:
                    logger.warning("Failed to instantiate {}: {}", name, e)
                    continue

            enabled.append(cls._instances[name])

        return enabled

    @classmethod
    def get_by_capability(cls, cap: ModelCapability, settings: Optional[Settings] = None) -> list[GeoModel]:
        """Return enabled models that have a given capability."""
        if settings is None:
            # Return from already-instantiated models
            return [
                m for m in cls._instances.values()
                if cap in m.info().capabilities
            ]

        return [
            m for m in cls.get_enabled(settings)
            if cap in m.info().capabilities
        ]

    @classmethod
    def clear(cls) -> None:
        """Reset registry (useful for tests)."""
        cls._models.clear()
        cls._instances.clear()
