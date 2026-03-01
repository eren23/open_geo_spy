"""ML Ensemble Agent - runs multiple geolocation models in parallel.

Discovers models via :class:`ModelRegistry` and aggregates predictions
with real confidence based on model agreement.
"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger
from openai import AsyncOpenAI

from src.config.settings import Settings, get_scoring_config
from src.evidence.chain import Evidence, EvidenceChain, EvidenceSource
from src.models.base import ModelCapability
from src.models.registry import ModelRegistry
from src.scoring.scorer import GeoScorer
from src.utils.geo_math import country_level_agreement, geographic_spread

# Importing adapters triggers registration
import src.models.adapters  # noqa: F401


class MLEnsembleAgent:
    """Runs geolocation ML models in parallel and aggregates with real confidence."""

    def __init__(self, settings: Settings, scorer: GeoScorer | None = None):
        self.settings = settings
        self.scorer = scorer or GeoScorer(get_scoring_config())
        self.client = AsyncOpenAI(
            base_url=settings.llm.base_url,
            api_key=settings.llm.api_key,
        )

    @property
    def streetclip_model_and_processor(self):
        """Expose loaded StreetCLIP model/processor for sharing with other agents."""
        from src.models.adapters import StreetCLIPAdapter

        for inst in ModelRegistry._instances.values():
            if isinstance(inst, StreetCLIPAdapter):
                return inst.model_and_processor
        return None

    async def predict(
        self,
        image_path: str,
        feature_evidence: EvidenceChain | None = None,
        candidate_cities: list[str] | None = None,
    ) -> EvidenceChain:
        """Run all enabled ML models in parallel and return aggregated evidence."""
        chain = EvidenceChain()
        logger.info("Starting ML ensemble prediction")

        # Build context from prior evidence
        context: dict[str, Any] = {}
        if feature_evidence:
            context["evidence_text"] = feature_evidence.to_prompt_context()
        # Pass candidate cities for StreetCLIP city prediction
        if candidate_cities:
            context["candidate_cities"] = candidate_cities

        # Discover enabled models from registry
        models = ModelRegistry.get_enabled(self.settings)
        if not models:
            logger.warning("No ML models enabled")
            return chain

        logger.info("Running {} models: {}", len(models), [m.info().name for m in models])

        # Run all models in parallel
        tasks = {m.info().name: m.predict(image_path, context) for m in models}
        task_names = list(tasks.keys())
        results_list = await asyncio.gather(*tasks.values(), return_exceptions=True)
        results = dict(zip(task_names, results_list))

        # Collect evidence from each model, applying ensemble weights
        for model in models:
            name = model.info().name
            result = results.get(name)

            if isinstance(result, Exception):
                logger.error("{} failed: {}", name, result)
                continue

            if isinstance(result, list) and result:
                evidences = model.to_evidence(result)

                # Store per-model weight as metadata for ensemble voting (Fix D:
                # don't multiply confidence directly — it inflates values to 1.0)
                weight = self.settings.ml.model_weights.get(name, model.info().default_weight)
                for ev in evidences:
                    ev.metadata["model_weight"] = weight

                added = chain.add_many(evidences)
                logger.info("{}: {} predictions -> {} new evidence (weight={:.1f})", name, len(result), added, weight)
            else:
                logger.debug("{}: no predictions", name)

        # Calculate ensemble agreement meta-evidence
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
        """Calculate real confidence from model agreement using per-model top-1 weighted voting."""
        countries = []
        country_weights = []
        coords = []

        for name, result in results.items():
            if isinstance(result, Exception) or not result:
                continue

            # Look up model weight for weighted voting
            weight = self.settings.ml.model_weights.get(name, 1.0)

            # Only use the top prediction per model (Fix C: prevents top-K
            # predictions from giving one model disproportionate voting power)
            if isinstance(result, list):
                top_pred = None
                for pred in result:
                    if isinstance(pred, dict):
                        top_pred = pred
                        break
                if top_pred:
                    if top_pred.get("country"):
                        countries.append(top_pred["country"])
                        country_weights.append(weight)
                    lat = top_pred.get("lat") or top_pred.get("latitude")
                    lon = top_pred.get("lon") or top_pred.get("longitude")
                    if lat is not None and lon is not None:
                        coords.append((float(lat), float(lon)))

        if not countries and not coords:
            return None

        # Weighted country agreement: weight each vote by model weight
        if countries:
            weighted_votes: dict[str, float] = {}
            for country, w in zip(countries, country_weights):
                weighted_votes[country] = weighted_votes.get(country, 0) + w
            total_weight = sum(country_weights)
            country_agree = max(weighted_votes.values()) / total_weight if total_weight > 0 else 0.0
        else:
            country_agree = 0.0

        spread = geographic_spread(coords) if len(coords) >= 2 else 0.0
        geo_agree = self.scorer.geo_agreement_score(spread)

        active_models = sum(1 for v in results.values() if not isinstance(v, (Exception, type(None))))
        if active_models == 0:
            return None

        if countries and coords:
            confidence = self.scorer.blend_ensemble(country_agree, geo_agree)
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
