"""Supervised lightweight noise classifier (v2)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

from .base import ClassificationResult, NoiseClassifierBase
from .categories import DEFENCE_NOISE_CATEGORIES, NOISE_CLASSES, UNKNOWN_CLASS
from .features import (
    DEFAULT_HOP_MS,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_WINDOW_MS,
    FEATURE_NAMES,
    FEATURE_VERSION,
    extract_features,
    feature_vector,
    sanitize_audio,
)

MODEL_BUNDLE_VERSION = "noise-classifier-v2-bundle-1"
DEFAULT_MODEL_FILENAME = "model.joblib"
DEFAULT_META_FILENAME = "model_meta.json"


@dataclass(frozen=True)
class SupervisedModelMeta:
    """Serializable metadata stored alongside a trained v2 model."""

    bundle_version: str
    model_name: str
    feature_version: str
    feature_names: tuple[str, ...]
    class_names: tuple[str, ...]
    sample_rate: int
    window_ms: float
    hop_ms: float
    silence_threshold_db: float
    seed: int
    training_config: dict[str, Any]
    split_info: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["feature_names"] = list(self.feature_names)
        payload["class_names"] = list(self.class_names)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SupervisedModelMeta":
        return cls(
            bundle_version=str(payload["bundle_version"]),
            model_name=str(payload["model_name"]),
            feature_version=str(payload["feature_version"]),
            feature_names=tuple(payload["feature_names"]),
            class_names=tuple(payload["class_names"]),
            sample_rate=int(payload["sample_rate"]),
            window_ms=float(payload["window_ms"]),
            hop_ms=float(payload["hop_ms"]),
            silence_threshold_db=float(payload["silence_threshold_db"]),
            seed=int(payload["seed"]),
            training_config=dict(payload.get("training_config", {})),
            split_info=dict(payload.get("split_info", {})),
        )


class SupervisedNoiseClassifier(NoiseClassifierBase):
    """
    Lightweight supervised noise classifier using ``noise-features-v1``.

    Trained on the three labelled defence categories. ``unknown`` is not a
    trained class; it is only emitted when optional confidence gating is
    enabled or the buffer is silent.
    """

    def __init__(
        self,
        estimator: BaseEstimator | ClassifierMixin | Pipeline,
        *,
        class_names: tuple[str, ...] = DEFENCE_NOISE_CATEGORIES,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        window_ms: float = DEFAULT_WINDOW_MS,
        hop_ms: float = DEFAULT_HOP_MS,
        silence_threshold_db: float = -55.0,
        confidence_threshold: float | None = None,
        model_name: str = "supervised",
        meta: SupervisedModelMeta | None = None,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")

        if not class_names:
            raise ValueError("class_names must be non-empty.")

        self._estimator = estimator
        self._labelled_classes = tuple(class_names)
        self._sample_rate = int(sample_rate)
        self._window_ms = float(window_ms)
        self._hop_ms = float(hop_ms)
        self._silence_threshold_db = float(silence_threshold_db)
        self._confidence_threshold = confidence_threshold
        self._model_name = model_name
        self._meta = meta
        self._buffer = np.empty(0, dtype=np.float32)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def class_names(self) -> tuple[str, ...]:
        return NOISE_CLASSES

    @property
    def labelled_class_names(self) -> tuple[str, ...]:
        return self._labelled_classes

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def meta(self) -> SupervisedModelMeta | None:
        return self._meta

    @property
    def estimator(self) -> BaseEstimator | ClassifierMixin | Pipeline:
        return self._estimator

    def reset(self) -> None:
        self._buffer = np.empty(0, dtype=np.float32)

    def pending_samples(self) -> int:
        return len(self._buffer)

    def process_chunk(self, audio: np.ndarray) -> None:
        chunk = sanitize_audio(audio)

        if chunk.size == 0:
            return

        if self._buffer.size == 0:
            self._buffer = chunk.copy()
        else:
            self._buffer = np.concatenate((self._buffer, chunk))

    def classify_buffered(self) -> ClassificationResult:
        return self.classify(self._buffer)

    def classify(self, audio: np.ndarray) -> ClassificationResult:
        audio = sanitize_audio(audio)
        features = extract_features(
            audio,
            self._sample_rate,
            window_ms=self._window_ms,
            hop_ms=self._hop_ms,
            silence_threshold_db=self._silence_threshold_db,
        )

        if (
            features.silence_ratio >= 0.95
            or features.rms_db <= self._silence_threshold_db
            or features.num_frames == 0
        ):
            probabilities = {name: 0.0 for name in NOISE_CLASSES}
            probabilities[UNKNOWN_CLASS] = 1.0
            return ClassificationResult(
                predicted_class=UNKNOWN_CLASS,
                scores=dict(probabilities),
                probabilities=probabilities,
                features=features,
                confidence=1.0,
            )

        vector = feature_vector(features).reshape(1, -1)
        vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)

        if hasattr(self._estimator, "predict_proba"):
            proba = np.asarray(
                self._estimator.predict_proba(vector),
                dtype=np.float64,
            )[0]
            estimator_classes = [
                str(label) for label in self._estimator.classes_
            ]
        else:
            predicted = str(self._estimator.predict(vector)[0])
            estimator_classes = list(self._labelled_classes)
            proba = np.zeros(len(estimator_classes), dtype=np.float64)
            if predicted in estimator_classes:
                proba[estimator_classes.index(predicted)] = 1.0
            else:
                proba[:] = 1.0 / len(estimator_classes)

        labelled_probs = {
            name: 0.0 for name in self._labelled_classes
        }
        for index, name in enumerate(estimator_classes):
            if name in labelled_probs:
                labelled_probs[name] = float(proba[index])

        total = float(sum(labelled_probs.values()))
        if total <= 1e-20:
            uniform = 1.0 / len(self._labelled_classes)
            labelled_probs = {
                name: uniform for name in self._labelled_classes
            }
        else:
            labelled_probs = {
                name: value / total
                for name, value in labelled_probs.items()
            }

        probabilities = {name: 0.0 for name in NOISE_CLASSES}
        for name, value in labelled_probs.items():
            probabilities[name] = float(value)

        ranked = sorted(
            labelled_probs.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        predicted, confidence = ranked[0]

        if (
            self._confidence_threshold is not None
            and confidence < self._confidence_threshold
        ):
            predicted = UNKNOWN_CLASS
            probabilities[UNKNOWN_CLASS] = float(1.0 - confidence)
            labelled_total = sum(
                probabilities[name] for name in self._labelled_classes
            )
            if labelled_total > 0:
                scale = confidence / labelled_total
                for name in self._labelled_classes:
                    probabilities[name] *= scale

        return ClassificationResult(
            predicted_class=predicted,
            scores=dict(probabilities),
            probabilities=probabilities,
            features=features,
            confidence=float(confidence),
        )

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        joblib.dump(self._estimator, directory / DEFAULT_MODEL_FILENAME)

        meta = self._meta
        if meta is None:
            meta = SupervisedModelMeta(
                bundle_version=MODEL_BUNDLE_VERSION,
                model_name=self._model_name,
                feature_version=FEATURE_VERSION,
                feature_names=FEATURE_NAMES,
                class_names=self._labelled_classes,
                sample_rate=self._sample_rate,
                window_ms=self._window_ms,
                hop_ms=self._hop_ms,
                silence_threshold_db=self._silence_threshold_db,
                seed=-1,
                training_config={},
                split_info={},
            )

        (directory / DEFAULT_META_FILENAME).write_text(
            json.dumps(meta.to_dict(), indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        directory: Path,
        *,
        confidence_threshold: float | None = None,
    ) -> "SupervisedNoiseClassifier":
        directory = Path(directory)
        model_path = directory / DEFAULT_MODEL_FILENAME
        meta_path = directory / DEFAULT_META_FILENAME

        if not model_path.exists():
            raise FileNotFoundError(f"Missing model file: {model_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {meta_path}")

        estimator = joblib.load(model_path)
        meta = SupervisedModelMeta.from_dict(
            json.loads(meta_path.read_text(encoding="utf-8"))
        )

        if meta.feature_version != FEATURE_VERSION:
            raise ValueError(
                f"Incompatible feature version {meta.feature_version}; "
                f"expected {FEATURE_VERSION}."
            )

        if tuple(meta.feature_names) != FEATURE_NAMES:
            raise ValueError(
                "Serialized feature names do not match FEATURE_NAMES."
            )

        return cls(
            estimator,
            class_names=meta.class_names,
            sample_rate=meta.sample_rate,
            window_ms=meta.window_ms,
            hop_ms=meta.hop_ms,
            silence_threshold_db=meta.silence_threshold_db,
            confidence_threshold=confidence_threshold,
            model_name=meta.model_name,
            meta=meta,
        )
