"""Confidence calibration using isotonic regression.

Maps raw confidence scores from the pipeline to calibrated probabilities
that better reflect the actual likelihood of being correct.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
from loguru import logger


class ConfidenceCalibrator:
    """Isotonic regression calibrator for pipeline confidence scores."""

    def __init__(self, data_path: Optional[str] = None):
        self._fitted = False
        self._x: list[float] = []  # Raw scores
        self._y: list[float] = []  # Calibrated values (monotonic)
        self._data_path = data_path

        if data_path and os.path.exists(data_path):
            self.load(data_path)

    def fit(self, raw_scores: list[float], actual_correct: list[bool]) -> None:
        """Fit the calibrator on ground truth data.

        Args:
            raw_scores: Pipeline confidence scores [0, 1].
            actual_correct: Whether the prediction was actually correct
                            (e.g., within 50km of ground truth).
        """
        from sklearn.isotonic import IsotonicRegression

        ir = IsotonicRegression(out_of_bounds="clip")
        x = np.array(raw_scores, dtype=np.float64)
        y = np.array(actual_correct, dtype=np.float64)

        ir.fit(x, y)
        self._x = ir.X_thresholds_.tolist()
        self._y = ir.y_thresholds_.tolist()
        self._fitted = True

        logger.info("Calibrator fitted on {} samples", len(raw_scores))

        if self._data_path:
            self.save(self._data_path)

    def calibrate(self, raw: float) -> float:
        """Map a raw confidence to a calibrated value.

        If not fitted, returns the raw score unchanged.
        """
        if not self._fitted or not self._x:
            return raw

        # Simple linear interpolation on the isotonic curve
        if raw <= self._x[0]:
            return self._y[0]
        if raw >= self._x[-1]:
            return self._y[-1]

        for i in range(len(self._x) - 1):
            if self._x[i] <= raw <= self._x[i + 1]:
                t = (raw - self._x[i]) / (self._x[i + 1] - self._x[i] + 1e-10)
                return self._y[i] + t * (self._y[i + 1] - self._y[i])

        return raw

    def calibrate_candidates(self, candidates: list[dict]) -> list[dict]:
        """Apply calibration to a list of candidate dicts in-place."""
        for c in candidates:
            if "confidence" in c:
                c["raw_confidence"] = c["confidence"]
                c["confidence"] = round(self.calibrate(c["confidence"]), 4)
        return candidates

    def save(self, path: str) -> None:
        """Save calibration data to JSON."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({"x": self._x, "y": self._y}, f)
        logger.info("Calibration data saved to {}", path)

    def load(self, path: str) -> None:
        """Load calibration data from JSON."""
        try:
            with open(path) as f:
                data = json.load(f)
            self._x = data["x"]
            self._y = data["y"]
            self._fitted = bool(self._x)
            logger.info("Calibration data loaded from {}", path)
        except Exception as e:
            logger.warning("Failed to load calibration data: {}", e)

    @property
    def is_fitted(self) -> bool:
        return self._fitted
