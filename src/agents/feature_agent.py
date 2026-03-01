"""Feature Extraction Agent - runs EXIF, VLM, and OCR in parallel.

This agent extracts all visual and metadata features from an image,
producing Evidence objects for the orchestrator.
"""

from __future__ import annotations

import asyncio

from loguru import logger
from openai import AsyncOpenAI

from src.config.settings import Settings
from src.evidence.chain import Evidence, EvidenceChain, EvidenceSource
from src.extraction import features as visual_features
from src.extraction import ocr
from src.extraction.metadata import MetadataExtractor


class FeatureExtractionAgent:
    """Runs EXIF + VLM visual analysis + OCR in parallel."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AsyncOpenAI(
            base_url=settings.llm.base_url,
            api_key=settings.llm.api_key,
        )
        self.metadata_extractor = MetadataExtractor()
        self.fast_model = settings.llm.fast_model

    async def extract(
        self,
        image_path: str,
        location_hint: str | None = None,
    ) -> EvidenceChain:
        """Extract all features from image, returning an evidence chain.

        Runs three extraction tasks in parallel:
        1. EXIF metadata extraction
        2. VLM visual feature analysis
        3. VLM OCR text extraction
        """
        chain = EvidenceChain()
        logger.info("Starting feature extraction for {}", image_path)

        # Add user hint as evidence if provided
        if location_hint:
            chain.add(
                Evidence(
                    source=EvidenceSource.USER_HINT,
                    content=f"User location hint: {location_hint}",
                    confidence=0.8,
                    metadata={"hint": location_hint},
                )
            )

        # Run all extractions in parallel
        metadata_task = asyncio.to_thread(self.metadata_extractor.extract_metadata, image_path)
        features_task = visual_features.extract_visual_features(image_path, self.client, self.fast_model)
        ocr_task = ocr.extract_text(image_path, self.client, self.fast_model)

        results = await asyncio.gather(metadata_task, features_task, ocr_task, return_exceptions=True)

        # Process metadata
        if isinstance(results[0], dict):
            metadata = results[0]
            chain.add_many(self.metadata_extractor.to_evidence(metadata))
            logger.info("Metadata extracted: GPS={}", metadata.get("gps_coordinates"))
        elif isinstance(results[0], Exception):
            logger.error("Metadata extraction failed: {}", results[0])
            metadata = {}

        # Process visual features
        if isinstance(results[1], dict):
            feat = results[1]
            chain.add_many(visual_features.to_evidence(feat))
            logger.info(
                "Visual features: env={}, landmarks={}, country_clues={}",
                feat.get("environment_type"),
                len(feat.get("landmarks", [])),
                len(feat.get("country_clues", [])),
            )
        elif isinstance(results[1], Exception):
            logger.error("Feature extraction failed: {}", results[1])
            feat = {}

        # Process OCR
        if isinstance(results[2], dict):
            ocr_result = results[2]
            chain.add_many(ocr.to_evidence(ocr_result))
            total_text = sum(len(v) for v in ocr_result.values() if isinstance(v, list))
            logger.info("OCR extracted {} text items", total_text)
        elif isinstance(results[2], Exception):
            logger.error("OCR extraction failed: {}", results[2])
            ocr_result = {}

        logger.info(
            "Feature extraction complete: {} evidences, agreement={:.2f}",
            len(chain.evidences),
            chain.agreement_score(),
        )

        return chain

    async def extract_with_raw(
        self,
        image_path: str,
        location_hint: str | None = None,
    ) -> tuple[EvidenceChain, dict, dict, dict]:
        """Like extract() but also returns the raw feature dicts.

        Returns: (evidence_chain, metadata, visual_features, ocr_result)
        """
        chain = EvidenceChain()

        if location_hint:
            chain.add(
                Evidence(
                    source=EvidenceSource.USER_HINT,
                    content=f"User location hint: {location_hint}",
                    confidence=0.8,
                    metadata={"hint": location_hint},
                )
            )

        metadata_task = asyncio.to_thread(self.metadata_extractor.extract_metadata, image_path)
        features_task = visual_features.extract_visual_features(image_path, self.client, self.fast_model)
        ocr_task = ocr.extract_text(image_path, self.client, self.fast_model)

        results = await asyncio.gather(metadata_task, features_task, ocr_task, return_exceptions=True)

        metadata = results[0] if isinstance(results[0], dict) else {}
        feat = results[1] if isinstance(results[1], dict) else {}
        ocr_result = results[2] if isinstance(results[2], dict) else {}

        if metadata:
            chain.add_many(self.metadata_extractor.to_evidence(metadata))
        if feat:
            chain.add_many(visual_features.to_evidence(feat))
        if ocr_result:
            chain.add_many(ocr.to_evidence(ocr_result))

        return chain, metadata, feat, ocr_result
