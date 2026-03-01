"""Candidate Verification Agent - visual comparison of location candidates.

Generates multiple candidate locations from existing evidence, fetches
reference images for each candidate, and scores them against the query
image using StreetCLIP embeddings. Produces VISUAL_MATCH evidence that
helps distinguish between same-category candidates (e.g., two hotels
in the same city).
"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from src.config.settings import Settings, get_scoring_config
from src.evidence.chain import Evidence, EvidenceChain, EvidenceSource
from src.geo.mapillary_client import MapillaryClient
from src.geo.serper_client import SerperClient
from src.models.visual_similarity import VisualSimilarityScorer
from src.scoring.scorer import GeoScorer


class CandidateVerificationAgent:
    """Generates candidate locations and visually verifies them against the query image."""

    def __init__(
        self,
        settings: Settings,
        streetclip_model: Any | None = None,
        streetclip_processor: Any | None = None,
        scorer: GeoScorer | None = None,
    ):
        self.settings = settings
        self.geo_scorer = scorer or GeoScorer(get_scoring_config())
        self._scorer = VisualSimilarityScorer(
            device=settings.ml.device,
            model=streetclip_model,
            processor=streetclip_processor,
        )

        # Serper for image search
        self._serper = (
            SerperClient(settings.geo.serper_api_key)
            if settings.geo.serper_api_key
            else None
        )

        # Mapillary (optional, only if token provided)
        self._mapillary = (
            MapillaryClient(settings.geo.mapillary_access_token)
            if settings.geo.mapillary_access_token
            else None
        )

    async def verify_candidates(
        self,
        image_path: str,
        evidence_chain: EvidenceChain,
        features: dict,
        ocr_result: dict,
    ) -> EvidenceChain:
        """Run full candidate verification pipeline.

        1. Generate candidates from evidence
        2. Fetch reference images for each candidate
        3. Score all references against query image
        4. Return VISUAL_MATCH evidence

        Returns:
            EvidenceChain with VISUAL_MATCH evidence entries.
        """
        result_chain = EvidenceChain()

        # Step 1: Generate candidates
        candidates = self._generate_candidates(evidence_chain, features, ocr_result)
        if not candidates:
            logger.info("No candidates generated for visual verification")
            return result_chain

        logger.info("Generated {} candidate queries for visual verification", len(candidates))

        # Step 2: Fetch reference images for all candidates in parallel
        refs_by_candidate = await self._fetch_reference_images(candidates)
        total_refs = sum(len(refs) for refs in refs_by_candidate.values())
        logger.info(
            "Fetched {} reference images across {} candidates",
            total_refs,
            len(refs_by_candidate),
        )

        if total_refs == 0:
            return result_chain

        # Step 3: Flatten all refs and score against query
        all_refs = []
        for name, refs in refs_by_candidate.items():
            for ref in refs:
                all_refs.append({
                    "url": ref["url"],
                    "candidate_name": name,
                    "title": ref.get("title", ""),
                })

        scored = await self._scorer.score_candidates(image_path, all_refs)

        if not scored:
            return result_chain

        # Step 4: Group by candidate, take best score per candidate
        best_by_candidate: dict[str, dict] = {}
        for item in scored:
            name = item["candidate_name"]
            if name not in best_by_candidate or item["similarity"] > best_by_candidate[name]["similarity"]:
                best_by_candidate[name] = item

        # Convert to VISUAL_MATCH evidence
        for name, best in sorted(
            best_by_candidate.items(),
            key=lambda x: x[1]["similarity"],
            reverse=True,
        ):
            confidence = self.geo_scorer.similarity_to_confidence(best["similarity"])
            candidate_info = next(
                (c for c in candidates if c["name"] == name), {}
            )

            evidence = Evidence(
                source=EvidenceSource.VISUAL_MATCH,
                content=(
                    f"Visual match for '{name}': similarity={best['similarity']:.3f} "
                    f"(ref: {best.get('title', best['url'][:60])})"
                ),
                confidence=confidence,
                latitude=candidate_info.get("lat"),
                longitude=candidate_info.get("lon"),
                country=candidate_info.get("country"),
                city=candidate_info.get("city"),
                url=best.get("url"),
                metadata={
                    "type": "visual_match",
                    "candidate_name": name,
                    "similarity": best["similarity"],
                    "ref_url": best["url"],
                    "ref_title": best.get("title", ""),
                },
            )
            result_chain.add(evidence)

        if best_by_candidate:
            top_name = max(best_by_candidate, key=lambda k: best_by_candidate[k]["similarity"])
            top_sim = best_by_candidate[top_name]["similarity"]
            logger.info("Best visual match: '{}' (sim={:.3f})", top_name, top_sim)

        return result_chain

    def _generate_candidates(
        self,
        evidence_chain: EvidenceChain,
        features: dict,
        ocr_result: dict,
    ) -> list[dict]:
        """Generate candidate locations using 4 strategies.

        Each candidate: {name, query, lat?, lon?, country?, city?}
        """
        candidates: dict[str, dict] = {}  # name -> candidate dict

        # Extract common location context from evidence
        countries = evidence_chain.country_predictions
        top_country = max(set(countries), key=countries.count) if countries else ""
        cities = [e.city for e in evidence_chain.evidences if e.city]
        top_city = max(set(cities), key=cities.count) if cities else ""

        if isinstance(ocr_result, dict):
            all_texts = (
                ocr_result.get("business_names", [])
                + ocr_result.get("street_signs", [])
                + ocr_result.get("informational", [])
                + ocr_result.get("building_info", [])
            )
            ocr_text = " ".join(all_texts)
        else:
            ocr_text = ""
        business_names = ocr_result.get("business_names", []) if isinstance(ocr_result, dict) else []

        # Strategy A: OCR business names + city/country from ML
        for bname in business_names[:3]:
            location_suffix = " ".join(filter(None, [top_city, top_country]))
            query = f"{bname} {location_suffix}".strip()
            if query and bname not in candidates:
                candidates[bname] = {
                    "name": bname,
                    "query": f"{query} exterior photo",
                    "country": top_country or None,
                    "city": top_city or None,
                }

        # Strategy B: Category search for generic OCR text
        category = _extract_category(ocr_text)
        if category and top_city:
            cat_name = f"{category}s in {top_city}"
            if cat_name not in candidates:
                candidates[cat_name] = {
                    "name": cat_name,
                    "query": f"{category}s in {top_city} {top_country} exterior".strip(),
                    "country": top_country or None,
                    "city": top_city or None,
                }

        # Strategy C: Landmark names from visual features
        landmarks = features.get("landmarks", []) if isinstance(features, dict) else []
        if isinstance(landmarks, str):
            landmarks = [landmarks]
        for lm in landmarks[:2]:
            if lm and lm not in candidates:
                location_suffix = " ".join(filter(None, [top_city, top_country]))
                candidates[lm] = {
                    "name": lm,
                    "query": f"{lm} {location_suffix} exterior photo".strip(),
                    "country": top_country or None,
                    "city": top_city or None,
                }

        # Strategy D: Top evidence by confidence (e.g., from Serper/OSM with specific names)
        for e in evidence_chain.top_evidence(10):
            if e.source in (EvidenceSource.SERPER, EvidenceSource.OSM, EvidenceSource.GOOGLE_MAPS):
                title = e.metadata.get("title", "")
                if title and len(title) > 3 and title not in candidates:
                    candidates[title] = {
                        "name": title,
                        "query": f"{title} exterior photo",
                        "lat": e.latitude,
                        "lon": e.longitude,
                        "country": e.country or top_country or None,
                        "city": e.city or top_city or None,
                    }

            if len(candidates) >= self.geo_scorer.max_candidates:
                break

        return list(candidates.values())[:self.geo_scorer.max_candidates]

    async def _fetch_reference_images(
        self, candidates: list[dict]
    ) -> dict[str, list[dict]]:
        """Fetch reference images for all candidates in parallel.

        Returns: {candidate_name: [{url, title}]}
        """
        tasks = {c["name"]: self._fetch_for_candidate(c) for c in candidates}
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        refs_by_candidate: dict[str, list[dict]] = {}
        for name, result in zip(tasks.keys(), results):
            if isinstance(result, list):
                refs_by_candidate[name] = result
            else:
                logger.debug("Failed to fetch refs for '{}': {}", name, result)
                refs_by_candidate[name] = []

        return refs_by_candidate

    async def _fetch_for_candidate(self, candidate: dict) -> list[dict]:
        """Fetch reference images for a single candidate.

        Uses Serper image search (primary) + Mapillary (if coords available).
        """
        refs: list[dict] = []

        # Primary: Serper image search
        if self._serper:
            try:
                images = await self._serper.search_images(
                    candidate["query"], num_results=self.geo_scorer.max_refs_per_candidate
                )
                for img in images:
                    url = img.get("imageUrl") or img.get("thumbnailUrl")
                    if url:
                        refs.append({
                            "url": url,
                            "title": img.get("title", ""),
                        })
            except Exception as e:
                logger.debug("Serper image search failed for '{}': {}", candidate["name"], e)

        # Supplement: Mapillary (if we have coords)
        if self._mapillary and candidate.get("lat") and candidate.get("lon"):
            try:
                mapillary_imgs = await self._mapillary.search_nearby(
                    candidate["lat"], candidate["lon"], radius_m=self.geo_scorer.mapillary_radius_m, limit=2
                )
                for mimg in mapillary_imgs:
                    url = mimg.get("image_url") or mimg.get("thumb_url")
                    if url:
                        refs.append({
                            "url": url,
                            "title": f"Mapillary street view near {candidate['name']}",
                        })
            except Exception as e:
                logger.debug("Mapillary search failed for '{}': {}", candidate["name"], e)

        return refs[:self.geo_scorer.max_refs_per_candidate]

    async def close(self):
        """Clean up resources."""
        if self._serper:
            await self._serper.close()
        if self._mapillary:
            await self._mapillary.close()


def _extract_category(text: str) -> str | None:
    """Extract a location category from OCR text (Hotel, Restaurant, etc.)."""
    if not text:
        return None
    categories = [
        "Hotel", "Restaurant", "Cafe", "Bar", "Museum", "Church",
        "Station", "Airport", "Hospital", "School", "University",
        "Theater", "Theatre", "Gallery", "Shop", "Store", "Bank",
        "Pharmacy", "Market", "Park", "Library",
    ]
    text_lower = text.lower()
    for cat in categories:
        if cat.lower() in text_lower:
            return cat
    return None
