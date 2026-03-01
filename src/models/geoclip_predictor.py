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
        self._cached_gps_features = None

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
        """Predict GPS coordinates from image."""
        self._ensure_loaded()

        try:
            with torch.no_grad():
                try:
                    top_pred_gps, top_pred_prob = self.model.predict(image_path, top_k=top_k)
                except (TypeError, RuntimeError) as e:
                    if "BaseModelOutputWithPooling" in str(e) or "must be Tensor" in str(e):
                        logger.warning("GeoCLIP transformers incompatibility, attempting patched predict: {}", e)
                        top_pred_gps, top_pred_prob = self._patched_predict(image_path, top_k)
                    else:
                        raise

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

    def _patched_predict(self, image_path: str, top_k: int = 5):
        """Patched predict that handles transformers BaseModelOutputWithPooling.

        When transformers>=4.40, CLIP outputs structured objects instead of raw tensors.
        We extract .pooler_output or the first element to get the actual tensor.
        """
        from PIL import Image as PILImage

        image = PILImage.open(image_path)

        # Access internal components
        img_proc = self.model.image_encoder.preprocess
        img_tensor = img_proc(image).unsqueeze(0).to(self.model.device if hasattr(self.model, 'device') else 'cpu')

        with torch.no_grad():
            img_output = self.model.image_encoder.CLIP.get_image_features(pixel_values=img_tensor)
            # Handle structured output
            if hasattr(img_output, 'pooler_output'):
                img_output = img_output.pooler_output
            elif hasattr(img_output, 'last_hidden_state'):
                img_output = img_output.last_hidden_state[:, 0]
            # If it's already a tensor, use as-is

            # Use cached GPS features or compute once
            if self._cached_gps_features is None:
                gps_gallery = self.model.gps_gallery
                location_encoder = self.model.location_encoder

                gps_features = location_encoder(gps_gallery)
                if hasattr(gps_features, 'pooler_output'):
                    gps_features = gps_features.pooler_output

                # Normalize GPS features (location_encoder doesn't normalize)
                gps_features = gps_features / gps_features.norm(dim=-1, keepdim=True)
                self._cached_gps_features = gps_features
            else:
                gps_features = self._cached_gps_features

            # img_output from get_image_features() is already L2-normalized

            # Compute similarity
            similarity = (img_output @ gps_features.T).squeeze(0)
            probs = similarity.softmax(dim=0)

            top_k_indices = probs.topk(top_k).indices
            top_pred_prob = probs[top_k_indices]
            top_pred_gps = gps_gallery[top_k_indices]

        return top_pred_gps, top_pred_prob

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
