"""Rule-based noise classifier (v1) — no ML training."""

from __future__ import annotations

import numpy as np

from .base import ClassificationResult, NoiseClassifierBase
from .categories import NOISE_CLASSES, UNKNOWN_CLASS
from .features import (
    DEFAULT_HOP_MS,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_WINDOW_MS,
    AggregatedFeatures,
    extract_features,
    sanitize_audio,
)


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


class NoiseClassifier(NoiseClassifierBase):
    """
    Deterministic rule-based noise classifier for defence categories.

    Accepts arbitrary-length mono float32 chunks, accumulates them for
    analysis, and returns class scores plus a predicted label. This is a
    lightweight analysis primitive — not a trained ML model.
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        *,
        window_ms: float = DEFAULT_WINDOW_MS,
        hop_ms: float = DEFAULT_HOP_MS,
        confidence_threshold: float = 0.35,
        silence_threshold_db: float = -55.0,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")

        if confidence_threshold <= 0.0 or confidence_threshold >= 1.0:
            raise ValueError(
                "confidence_threshold must be in (0, 1)."
            )

        self._sample_rate = int(sample_rate)
        self._window_ms = float(window_ms)
        self._hop_ms = float(hop_ms)
        self._confidence_threshold = float(confidence_threshold)
        self._silence_threshold_db = float(silence_threshold_db)
        self._buffer = np.empty(0, dtype=np.float32)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def class_names(self) -> tuple[str, ...]:
        return NOISE_CLASSES

    def reset(self) -> None:
        self._buffer = np.empty(0, dtype=np.float32)

    def pending_samples(self) -> int:
        return len(self._buffer)

    def process_chunk(
        self,
        audio: np.ndarray,
    ) -> None:
        """Append a mono chunk to the internal analysis buffer."""

        chunk = sanitize_audio(audio)

        if chunk.size == 0:
            return

        if self._buffer.size == 0:
            self._buffer = chunk.copy()
        else:
            self._buffer = np.concatenate((self._buffer, chunk))

    def classify_buffered(self) -> ClassificationResult:
        """Classify all audio currently held in the internal buffer."""

        return self.classify(self._buffer)

    def classify(self, audio: np.ndarray) -> ClassificationResult:
        """Classify a mono audio segment in one shot."""

        audio = sanitize_audio(audio)

        features = extract_features(
            audio,
            self._sample_rate,
            window_ms=self._window_ms,
            hop_ms=self._hop_ms,
            silence_threshold_db=self._silence_threshold_db,
        )

        scores = self._category_scores(features)
        probabilities = self._normalize_scores(scores)
        predicted = self._select_class(probabilities, features)
        confidence = max(probabilities.values()) if probabilities else 0.0

        return ClassificationResult(
            predicted_class=predicted,
            scores=scores,
            probabilities=probabilities,
            features=features,
            confidence=float(confidence),
        )

    def _category_scores(
        self,
        features: AggregatedFeatures,
    ) -> dict[str, float]:
        if features.silence_ratio >= 0.95 or features.rms_db <= self._silence_threshold_db:
            return {
                "uav_drone": 0.0,
                "vehicle_engine": 0.0,
                "impulsive_firearms": 0.0,
                "unknown": 1.0,
            }

        broadband_like = (
            features.spectral_flatness > 0.7
            and features.low_band_ratio < 0.25
            and features.impulsive_index < 0.1
        )

        if broadband_like:
            return {
                "uav_drone": 0.0,
                "vehicle_engine": 0.0,
                "impulsive_firearms": 0.0,
                "unknown": 1.0,
            }

        impulsive = (
            0.45 * _clamp(features.impulsive_index / 0.25)
            + 0.30 * _clamp((features.crest_factor - 3.5) / 4.0)
            + 0.15 * _clamp(features.spectral_flatness / 0.35)
            + 0.10 * _clamp(features.peak_envelope / 0.5)
        )

        engine = (
            0.40 * _clamp((features.low_band_ratio - 0.45) / 0.35)
            + 0.20 * _clamp(1.0 - features.zero_crossing_rate / 0.12)
            + 0.20 * _clamp(1.0 - features.rms_variability / 0.8)
            + 0.10 * _clamp(1.0 - features.crest_factor / 5.0)
            + 0.10 * _clamp(features.spectral_centroid_hz / 1200.0)
        )

        drone = (
            0.30 * _clamp(features.tonal_index / 0.35)
            + 0.25 * _clamp(features.mid_band_ratio / 0.45)
            + 0.20 * _clamp(features.modulation_index / 0.35)
            + 0.15 * _clamp(1.0 - features.impulsive_index / 0.2)
            + 0.10 * _clamp(features.spectral_centroid_hz / 2500.0)
        )

        unknown = (
            0.40 * _clamp(features.silence_ratio / 0.5)
            + 0.30 * _clamp(1.0 - max(impulsive, engine, drone))
            + 0.30 * _clamp(features.spectral_flatness)
        )

        return {
            "uav_drone": float(drone),
            "vehicle_engine": float(engine),
            "impulsive_firearms": float(impulsive),
            "unknown": float(unknown),
        }

    def _normalize_scores(
        self,
        scores: dict[str, float],
    ) -> dict[str, float]:
        values = np.array(
            [scores[name] for name in NOISE_CLASSES],
            dtype=np.float64,
        )
        values = np.maximum(values, 0.0)
        total = float(np.sum(values))

        if total <= 1e-20:
            uniform = 1.0 / len(NOISE_CLASSES)

            return {name: uniform for name in NOISE_CLASSES}

        normalized = values / total

        return {
            name: float(normalized[index])
            for index, name in enumerate(NOISE_CLASSES)
        }

    def _select_class(
        self,
        probabilities: dict[str, float],
        features: AggregatedFeatures,
    ) -> str:
        if (
            features.silence_ratio >= 0.95
            or features.rms_db <= self._silence_threshold_db
        ):
            return UNKNOWN_CLASS

        ranked = sorted(
            probabilities.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        best_class, best_prob = ranked[0]
        second_prob = ranked[1][1] if len(ranked) > 1 else 0.0

        if best_class == UNKNOWN_CLASS:
            return UNKNOWN_CLASS

        if best_prob < self._confidence_threshold:
            return UNKNOWN_CLASS

        if (best_prob - second_prob) < 0.08:
            return UNKNOWN_CLASS

        return best_class
