"""StreetCLIP-based visual similarity scoring.

Reuses the already-loaded StreetCLIP model to compute image-to-image
cosine similarity. Used by the CandidateVerificationAgent to compare
the query image against reference photos of candidate locations.
"""

from __future__ import annotations

import asyncio
from io import BytesIO
from typing import Any

import httpx
import numpy as np
import torch
from loguru import logger
from PIL import Image


class VisualSimilarityScorer:
    """Scores visual similarity between a query image and reference images using StreetCLIP embeddings."""

    def __init__(
        self,
        device: str = "cpu",
        model: Any | None = None,
        processor: Any | None = None,
    ):
        self.device = device
        self.model = model
        self.processor = processor

    def _ensure_loaded(self):
        """Lazy-load StreetCLIP if not provided via constructor."""
        if self.model is not None:
            return
        try:
            from transformers import CLIPModel, CLIPProcessor

            model_name = "geolocal/StreetCLIP"
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("VisualSimilarityScorer loaded StreetCLIP on {}", self.device)
        except Exception as e:
            logger.error("Failed to load StreetCLIP for visual similarity: {}", e)
            raise

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Encode a PIL image to an L2-normalized 768-dim CLIP embedding."""
        self._ensure_loaded()

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            features = self.model.get_image_features(**inputs)

        embedding = features[0].cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    @staticmethod
    def cosine_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """Cosine similarity between two L2-normalized embeddings."""
        return float(np.dot(emb_a, emb_b))

    async def score_candidates(
        self,
        query_image_path: str,
        reference_image_urls: list[dict],
    ) -> list[dict]:
        """Download reference images, encode all, and return sorted by similarity.

        Args:
            query_image_path: Path to the query image file.
            reference_image_urls: List of dicts with keys: url, candidate_name, (optional) title.

        Returns:
            Sorted list of dicts: {url, candidate_name, title, similarity}.
        """
        self._ensure_loaded()

        # Encode query image
        try:
            query_image = Image.open(query_image_path).convert("RGB")
            query_emb = await asyncio.to_thread(self.encode_image, query_image)
        except Exception as e:
            logger.error("Failed to encode query image: {}", e)
            return []

        # Download reference images in parallel
        ref_images = await _download_images_parallel(reference_image_urls)

        if not ref_images:
            logger.warning("No reference images could be downloaded")
            return []

        # Encode all reference images
        results = []
        for ref in ref_images:
            try:
                ref_emb = await asyncio.to_thread(self.encode_image, ref["image"])
                sim = self.cosine_similarity(query_emb, ref_emb)
                results.append({
                    "url": ref["url"],
                    "candidate_name": ref["candidate_name"],
                    "title": ref.get("title", ""),
                    "similarity": sim,
                })
            except Exception as e:
                logger.debug("Failed to encode reference image {}: {}", ref["url"][:80], e)

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results


async def _download_image(
    client: httpx.AsyncClient, url: str
) -> Image.Image | None:
    """Download a single image from URL, return PIL Image or None."""
    try:
        resp = await client.get(url, follow_redirects=True)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "image" not in content_type and not url.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        ):
            return None
        return Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


async def _download_images_parallel(
    refs: list[dict],
    max_concurrent: int = 10,
    timeout: float = 8.0,
) -> list[dict]:
    """Download multiple images in parallel with concurrency limit."""
    semaphore = asyncio.Semaphore(max_concurrent)
    results = []

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:

        async def _download_one(ref: dict) -> dict | None:
            async with semaphore:
                img = await _download_image(client, ref["url"])
                if img is None:
                    return None
                return {
                    "url": ref["url"],
                    "candidate_name": ref["candidate_name"],
                    "title": ref.get("title", ""),
                    "image": img,
                }

        tasks = [_download_one(ref) for ref in refs]
        downloaded = await asyncio.gather(*tasks, return_exceptions=True)

        for item in downloaded:
            if isinstance(item, dict):
                results.append(item)

    return results
