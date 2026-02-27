"""GeoCLIP GPS prediction model.

GeoCLIP (NeurIPS '23) uses CLIP + continuous GPS encoding for geolocation.
Outputs direct lat/lon coordinates with a meaningful confidence score.
"""

from __future__ import annotations

from typing import Optional

import torch
from loguru import logger
from src.evidence.chain import Evidence, EvidenceSource


class GeoCLIPPredictor:
    """GeoCLIP-based continuous GPS prediction."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None
        self.processor = None

    def _ensure_loaded(self):
        """Lazy-load the model."""
        if self.model is not None:
            return

        try:
            from geoclip import GeoCLIP

            self.model = GeoCLIP()
            self.model.to(self.device)
            self.model.eval()
            logger.info("GeoCLIP model loaded on {}", self.device)
        except ImportError:
            logger.warning("geoclip not installed. Install with: pip install geoclip")
            raise
        except Exception as e:
            logger.error("Failed to load GeoCLIP: {}", e)
            raise

    def predict(self, image_path: str, top_k: int = 5) -> list[dict]:
        """Predict GPS coordinates from image.

        Returns list of predictions sorted by confidence:
        [{"lat": float, "lon": float, "confidence": float}, ...]
        """
        self._ensure_loaded()

        try:
            with torch.no_grad():
                top_pred_gps, top_pred_prob = self.model.predict(image_path, top_k=top_k)

            predictions = []
            for i in range(len(top_pred_gps)):
                lat = float(top_pred_gps[i][0])
                lon = float(top_pred_gps[i][1])
                prob = float(top_pred_prob[i])

                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    predictions.append({
                        "lat": lat,
                        "lon": lon,
                        "confidence": prob,
                    })

            return predictions

        except Exception as e:
            logger.error("GeoCLIP prediction failed: {}", e)
            return []

    def to_evidence(self, predictions: list[dict]) -> list[Evidence]:
        """Convert predictions to Evidence objects."""
        evidences = []
        for i, pred in enumerate(predictions):
            evidences.append(
                Evidence(
                    source=EvidenceSource.GEOCLIP,
                    content=f"GeoCLIP prediction #{i+1}: ({pred['lat']:.4f}, {pred['lon']:.4f})",
                    confidence=pred["confidence"],
                    latitude=pred["lat"],
                    longitude=pred["lon"],
                    metadata={"rank": i + 1, "model": "geoclip"},
                )
            )
        return evidences
