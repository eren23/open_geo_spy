"""ML Ensemble Agent - runs multiple geolocation models in parallel.

Combines predictions from GeoCLIP, StreetCLIP, and VLM geo-reasoning
with real confidence based on model agreement, not random values.
"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger
from openai import AsyncOpenAI

from src.config.settings import Settings
from src.evidence.chain import Evidence, EvidenceChain, EvidenceSource
from src.models import vlm_geo
from src.utils.geo_math import country_level_agreement, geographic_spread


class MLEnsembleAgent:
    """Runs geolocation ML models in parallel and aggregates with real confidence."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AsyncOpenAI(
            base_url=settings.llm.base_url,
            api_key=settings.llm.api_key,
        )

        # Lazy-loaded ML models
        self._geoclip = None
        self._streetclip = None

    @property
    def geoclip(self):
        if self._geoclip is None and self.settings.ml.enable_geoclip:
            try:
                from src.models.geoclip_predictor import GeoCLIPPredictor
                self._geoclip = GeoCLIPPredictor(device=self.settings.ml.device)
            except Exception as e:
                logger.warning("GeoCLIP unavailable: {}", e)
        return self._geoclip

    @property
    def streetclip(self):
        if self._streetclip is None and self.settings.ml.enable_streetclip:
            try:
                from src.models.streetclip_predictor import StreetCLIPPredictor
                self._streetclip = StreetCLIPPredictor(device=self.settings.ml.device)
            except Exception as e:
                logger.warning("StreetCLIP unavailable: {}", e)
        return self._streetclip

    @property
    def streetclip_model_and_processor(self):
        """Expose loaded StreetCLIP model/processor for sharing with other agents."""
        if self._streetclip and self._streetclip.model:
            return (self._streetclip.model, self._streetclip.processor)
        return None

    async def predict(
        self,
        image_path: str,
        feature_evidence: EvidenceChain | None = None,
    ) -> EvidenceChain:
        """Run all ML models in parallel and return aggregated evidence.

        Args:
            image_path: Path to the image
            feature_evidence: Optional evidence from feature extraction (provides context)
        """
        chain = EvidenceChain()
        logger.info("Starting ML ensemble prediction")

        # Build additional context from prior evidence
        context = ""
        if feature_evidence:
            context = feature_evidence.to_prompt_context()

        # Run models in parallel
        tasks = {}

        # VLM geo reasoning (always available - uses API)
        tasks["vlm_geo"] = vlm_geo.predict_location(
            image_path, self.client, self.settings.llm.reasoning_model, context
        )

        # GeoCLIP (local model)
        if self.geoclip:
            tasks["geoclip"] = asyncio.to_thread(self.geoclip.predict, image_path, 5)

        # StreetCLIP (local model)
        if self.streetclip:
            tasks["streetclip"] = asyncio.to_thread(self.streetclip.predict_country, image_path, 5)

        # Gather results
        task_names = list(tasks.keys())
        results_list = await asyncio.gather(*tasks.values(), return_exceptions=True)
        results = dict(zip(task_names, results_list))

        # Process VLM geo
        vlm_result = results.get("vlm_geo")
        if isinstance(vlm_result, dict):
            chain.add_many(vlm_geo.to_evidence(vlm_result))
            logger.info("VLM geo: country={}, conf={}", vlm_result.get("country"), vlm_result.get("confidence"))
        elif isinstance(vlm_result, Exception):
            logger.error("VLM geo failed: {}", vlm_result)

        # Process GeoCLIP
        geoclip_result = results.get("geoclip")
        if isinstance(geoclip_result, list) and geoclip_result and self.geoclip:
            chain.add_many(self.geoclip.to_evidence(geoclip_result))
            logger.info("GeoCLIP: top=({:.2f}, {:.2f}), conf={:.3f}",
                        geoclip_result[0]["lat"], geoclip_result[0]["lon"], geoclip_result[0]["confidence"])
        elif isinstance(geoclip_result, Exception):
            logger.error("GeoCLIP failed: {}", geoclip_result)

        # Process StreetCLIP
        streetclip_result = results.get("streetclip")
        if isinstance(streetclip_result, list) and streetclip_result and self.streetclip:
            chain.add_many(self.streetclip.to_evidence(streetclip_result))
            logger.info("StreetCLIP: top={}, conf={:.3f}",
                        streetclip_result[0]["country"], streetclip_result[0]["confidence"])
        elif isinstance(streetclip_result, Exception):
            logger.error("StreetCLIP failed: {}", streetclip_result)

        # Calculate ensemble agreement and add meta-evidence
        ensemble_meta = self._calculate_ensemble_confidence(results)
        if ensemble_meta:
            chain.add(ensemble_meta)

        logger.info(
            "ML ensemble complete: {} evidences, agreement={:.2f}",
            len(chain.evidences),
            chain.agreement_score(),
        )

        return chain

    def _calculate_ensemble_confidence(self, results: dict[str, Any]) -> Evidence | None:
        """Calculate real confidence from model agreement.

        - All models agree on country -> 0.9+ confidence
        - 3/4 agree -> 0.7-0.9
        - Split opinions -> 0.4-0.6
        - Plus geographic distance between predictions as secondary metric
        """
        countries = []
        coords = []

        # VLM geo
        vlm = results.get("vlm_geo")
        if isinstance(vlm, dict) and vlm.get("country"):
            countries.append(vlm["country"])
            if vlm.get("latitude") is not None:
                coords.append((vlm["latitude"], vlm["longitude"]))

        # GeoCLIP
        geoclip = results.get("geoclip")
        if isinstance(geoclip, list) and geoclip:
            coords.append((geoclip[0]["lat"], geoclip[0]["lon"]))
            # GeoCLIP doesn't directly predict country, but we could reverse geocode

        # StreetCLIP
        streetclip = results.get("streetclip")
        if isinstance(streetclip, list) and streetclip:
            countries.append(streetclip[0]["country"])

        if not countries and not coords:
            return None

        # Country agreement
        country_agree = country_level_agreement(countries) if countries else 0.0

        # Geographic spread
        spread = geographic_spread(coords) if len(coords) >= 2 else 0.0
        if spread < 50:
            geo_agree = 1.0
        elif spread < 200:
            geo_agree = 0.7
        elif spread < 500:
            geo_agree = 0.4
        else:
            geo_agree = 0.2

        # Combined confidence
        active_models = sum(1 for v in results.values() if not isinstance(v, (Exception, type(None))))
        if active_models == 0:
            return None

        if countries and coords:
            confidence = 0.5 * country_agree + 0.5 * geo_agree
        elif countries:
            confidence = country_agree
        else:
            confidence = geo_agree

        return Evidence(
            source=EvidenceSource.REASONING,
            content=(
                f"Ensemble agreement: {active_models} models active, "
                f"country_agreement={country_agree:.2f}, "
                f"geo_spread={spread:.0f}km, "
                f"countries={countries}"
            ),
            confidence=confidence,
            metadata={
                "type": "ensemble_meta",
                "active_models": active_models,
                "country_agreement": country_agree,
                "geo_spread_km": spread,
                "countries_predicted": countries,
            },
        )
