"""Noise classification analysis primitives."""

from .base import ClassificationResult, NoiseClassifierBase
from .categories import (
    DEFENCE_NOISE_CATEGORIES,
    NOISE_CLASSES,
    UNKNOWN_CLASS,
)
from .classifier import NoiseClassifier
from .features import (
    DEFAULT_HOP_MS,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_WINDOW_MS,
    FEATURE_NAMES,
    FEATURE_VERSION,
    AggregatedFeatures,
    extract_features,
    feature_vector,
)
from .supervised import SupervisedNoiseClassifier

__all__ = [
    "AggregatedFeatures",
    "ClassificationResult",
    "DEFENCE_NOISE_CATEGORIES",
    "DEFAULT_HOP_MS",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_WINDOW_MS",
    "FEATURE_NAMES",
    "FEATURE_VERSION",
    "NOISE_CLASSES",
    "NoiseClassifier",
    "NoiseClassifierBase",
    "SupervisedNoiseClassifier",
    "UNKNOWN_CLASS",
    "extract_features",
    "feature_vector",
]
