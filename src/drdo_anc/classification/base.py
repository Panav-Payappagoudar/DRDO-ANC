"""Shared classifier API for noise analysis modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .features import AggregatedFeatures


@dataclass(frozen=True)
class ClassificationResult:
    """Classifier output for one audio segment."""

    predicted_class: str
    scores: dict[str, float]
    probabilities: dict[str, float]
    features: AggregatedFeatures
    confidence: float = 0.0


class NoiseClassifierBase(ABC):
    """Common mono-chunk classifier interface used by v1 and v2."""

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def class_names(self) -> tuple[str, ...]:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def pending_samples(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def process_chunk(self, audio: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def classify_buffered(self) -> ClassificationResult:
        raise NotImplementedError

    @abstractmethod
    def classify(self, audio: np.ndarray) -> ClassificationResult:
        raise NotImplementedError
