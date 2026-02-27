"""StreetCLIP zero-shot geolocation model.

Uses CLIP-based zero-shot classification against country/city labels
to predict location without any fine-tuning on geo data.
"""

from __future__ import annotations

import torch
from loguru import logger
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.evidence.chain import Evidence, EvidenceSource

# Top countries for zero-shot classification
COUNTRIES = [
    "Afghanistan", "Albania", "Algeria", "Argentina", "Australia", "Austria",
    "Bangladesh", "Belgium", "Bolivia", "Brazil", "Bulgaria", "Cambodia",
    "Cameroon", "Canada", "Chile", "China", "Colombia", "Croatia",
    "Czech Republic", "Denmark", "Dominican Republic", "Ecuador", "Egypt",
    "Estonia", "Ethiopia", "Finland", "France", "Germany", "Ghana", "Greece",
    "Guatemala", "Hungary", "Iceland", "India", "Indonesia", "Iran", "Iraq",
    "Ireland", "Israel", "Italy", "Jamaica", "Japan", "Jordan", "Kazakhstan",
    "Kenya", "Latvia", "Lebanon", "Lithuania", "Malaysia", "Mexico",
    "Mongolia", "Morocco", "Myanmar", "Nepal", "Netherlands", "New Zealand",
    "Nigeria", "North Korea", "Norway", "Pakistan", "Panama", "Peru",
    "Philippines", "Poland", "Portugal", "Romania", "Russia", "Saudi Arabia",
    "Senegal", "Serbia", "Singapore", "Slovakia", "Slovenia", "South Africa",
    "South Korea", "Spain", "Sri Lanka", "Sweden", "Switzerland", "Taiwan",
    "Tanzania", "Thailand", "Tunisia", "Turkey", "Uganda", "Ukraine",
    "United Arab Emirates", "United Kingdom", "United States", "Uruguay",
    "Uzbekistan", "Venezuela", "Vietnam",
]


class StreetCLIPPredictor:
    """StreetCLIP zero-shot country/city prediction."""

    MODEL_NAME = "geolocal/StreetCLIP"

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None
        self.processor = None

    def _ensure_loaded(self):
        if self.model is not None:
            return

        try:
            self.model = CLIPModel.from_pretrained(self.MODEL_NAME)
            self.processor = CLIPProcessor.from_pretrained(self.MODEL_NAME)
            self.model.to(self.device)
            self.model.eval()
            logger.info("StreetCLIP loaded on {}", self.device)
        except Exception as e:
            logger.error("Failed to load StreetCLIP: {}", e)
            raise

    def predict_country(self, image_path: str, top_k: int = 5) -> list[dict]:
        """Predict country from image using zero-shot classification.

        Returns: [{"country": str, "confidence": float}, ...]
        """
        self._ensure_loaded()

        try:
            image = Image.open(image_path).convert("RGB")
            labels = [f"a street view photo from {c}" for c in COUNTRIES]

            inputs = self.processor(
                text=labels,
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits_per_image[0]
                probs = logits.softmax(dim=0)

            top_indices = probs.argsort(descending=True)[:top_k]
            results = []
            for idx in top_indices:
                results.append({
                    "country": COUNTRIES[idx],
                    "confidence": float(probs[idx]),
                })

            return results

        except Exception as e:
            logger.error("StreetCLIP prediction failed: {}", e)
            return []

    def predict_city(self, image_path: str, country: str, cities: list[str]) -> list[dict]:
        """Predict city within a country using zero-shot classification.

        Args:
            image_path: Path to image
            country: The predicted country
            cities: List of candidate cities to classify against

        Returns: [{"city": str, "confidence": float}, ...]
        """
        if not cities:
            return []

        self._ensure_loaded()

        try:
            image = Image.open(image_path).convert("RGB")
            labels = [f"a photo from {city}, {country}" for city in cities]

            inputs = self.processor(
                text=labels,
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits_per_image[0]
                probs = logits.softmax(dim=0)

            results = []
            for i, city in enumerate(cities):
                results.append({
                    "city": city,
                    "confidence": float(probs[i]),
                })

            return sorted(results, key=lambda x: x["confidence"], reverse=True)

        except Exception as e:
            logger.error("StreetCLIP city prediction failed: {}", e)
            return []

    def to_evidence(self, country_preds: list[dict], city_preds: list[dict] | None = None) -> list[Evidence]:
        """Convert predictions to Evidence objects."""
        evidences = []

        for pred in country_preds[:3]:
            evidences.append(
                Evidence(
                    source=EvidenceSource.STREETCLIP,
                    content=f"StreetCLIP country: {pred['country']}",
                    confidence=pred["confidence"],
                    country=pred["country"],
                    metadata={"model": "streetclip", "type": "country"},
                )
            )

        if city_preds:
            for pred in city_preds[:3]:
                evidences.append(
                    Evidence(
                        source=EvidenceSource.STREETCLIP,
                        content=f"StreetCLIP city: {pred['city']}",
                        confidence=pred["confidence"],
                        city=pred["city"],
                        metadata={"model": "streetclip", "type": "city"},
                    )
                )

        return evidences
