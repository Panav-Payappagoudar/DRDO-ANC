"""Training and evaluation utilities for Noise Classifier v2."""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from drdo_anc.dataset.zip_manifest_dataset import ZipManifestDataset

from .categories import DEFENCE_NOISE_CATEGORIES
from .classifier import NoiseClassifier
from .evaluation import CorpusClip, build_defence_noise_corpus
from .features import (
    DEFAULT_HOP_MS,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_WINDOW_MS,
    FEATURE_NAMES,
    FEATURE_VERSION,
    extract_features,
    feature_vector,
)
from .source_identity import recording_source_id
from .supervised import (
    MODEL_BUNDLE_VERSION,
    SupervisedModelMeta,
    SupervisedNoiseClassifier,
)


DEFAULT_SEED = 42
DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15
TRAINING_RULES_VERSION = "noise-classifier-v2-train-v1"


@dataclass(frozen=True)
class LabeledFeatureRow:
    clip_id: str
    source_id: str
    recording_id: str
    label: str
    features: np.ndarray
    num_samples: int
    sample_rate: int


@dataclass(frozen=True)
class DatasetSplit:
    train: tuple[LabeledFeatureRow, ...]
    validation: tuple[LabeledFeatureRow, ...]
    test: tuple[LabeledFeatureRow, ...]
    seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float

    def class_distribution(self) -> dict[str, dict[str, int]]:
        return {
            "train": _count_labels(self.train),
            "validation": _count_labels(self.validation),
            "test": _count_labels(self.test),
        }

    def recording_ids(self, split_name: str) -> set[str]:
        rows = {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }[split_name]
        return {row.recording_id for row in rows}


@dataclass
class CandidateResult:
    model_name: str
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    model_size_bytes: int
    estimator: Any


@dataclass
class TrainingReport:
    rules_version: str
    seed: int
    feature_version: str
    class_names: tuple[str, ...]
    class_distribution: dict[str, dict[str, int]]
    split_info: dict[str, Any]
    candidates: list[dict[str, Any]] = field(default_factory=list)
    selected_model: str | None = None
    selection_criterion: str = "validation_macro_f1"
    v1_comparison: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _count_labels(rows: Iterable[LabeledFeatureRow]) -> dict[str, int]:
    counts = Counter(row.label for row in rows)
    return {
        label: int(counts.get(label, 0))
        for label in DEFENCE_NOISE_CATEGORIES
    }


def _finite_feature_vector(features) -> np.ndarray:
    vector = feature_vector(features)
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)


def extract_labeled_features(
    clips: tuple[CorpusClip, ...] | list[CorpusClip],
    dataset: ZipManifestDataset,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    window_ms: float = DEFAULT_WINDOW_MS,
    hop_ms: float = DEFAULT_HOP_MS,
) -> list[LabeledFeatureRow]:
    """Extract deterministic feature vectors for labelled corpus clips."""

    rows: list[LabeledFeatureRow] = []

    for clip in clips:
        audio, audio_sr = dataset.load_audio(clip.source)
        if audio_sr != sample_rate:
            raise ValueError(
                f"Sample rate mismatch for {clip.clip_id}: "
                f"{audio_sr} != {sample_rate}"
            )

        features = extract_features(
            audio,
            sample_rate,
            window_ms=window_ms,
            hop_ms=hop_ms,
        )
        recording_id = recording_source_id(clip.source, clip.ground_truth)

        rows.append(
            LabeledFeatureRow(
                clip_id=clip.clip_id,
                source_id=clip.source.sample_id,
                recording_id=recording_id,
                label=clip.ground_truth,
                features=_finite_feature_vector(features),
                num_samples=int(audio.size),
                sample_rate=audio_sr,
            )
        )

    return rows


def stratified_recording_split(
    rows: list[LabeledFeatureRow],
    *,
    seed: int = DEFAULT_SEED,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
) -> DatasetSplit:
    """
    Deterministic stratified split with recording-level grouping.

    All clips sharing a ``recording_id`` are assigned to the same split.
    Ratios are applied per class over unique recordings.
    """

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-9:
        raise ValueError("train/val/test ratios must sum to 1.0.")

    by_recording: dict[str, list[LabeledFeatureRow]] = defaultdict(list)
    recording_label: dict[str, str] = {}

    for row in rows:
        by_recording[row.recording_id].append(row)
        previous = recording_label.get(row.recording_id)
        if previous is not None and previous != row.label:
            raise ValueError(
                f"Recording {row.recording_id} has conflicting labels "
                f"{previous} vs {row.label}."
            )
        recording_label[row.recording_id] = row.label

    recordings_by_label: dict[str, list[str]] = {
        label: [] for label in DEFENCE_NOISE_CATEGORIES
    }
    for recording_id, label in recording_label.items():
        recordings_by_label[label].append(recording_id)

    rng = np.random.default_rng(seed)
    assignment: dict[str, str] = {}

    for label in DEFENCE_NOISE_CATEGORIES:
        ids = sorted(recordings_by_label[label])
        if not ids:
            raise ValueError(f"No recordings available for class {label}.")

        order = rng.permutation(len(ids))
        shuffled = [ids[index] for index in order]
        n = len(shuffled)
        n_test = max(1, int(round(n * test_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        if n_test + n_val >= n:
            # Keep at least one training recording when class is tiny.
            n_test = min(n_test, max(1, n // 5))
            n_val = min(n_val, max(1, n // 5))
            if n_test + n_val >= n:
                n_test = 1 if n >= 3 else 0
                n_val = 1 if n >= 2 else 0

        test_ids = shuffled[:n_test]
        val_ids = shuffled[n_test : n_test + n_val]
        train_ids = shuffled[n_test + n_val :]

        for recording_id in train_ids:
            assignment[recording_id] = "train"
        for recording_id in val_ids:
            assignment[recording_id] = "validation"
        for recording_id in test_ids:
            assignment[recording_id] = "test"

    train_rows: list[LabeledFeatureRow] = []
    val_rows: list[LabeledFeatureRow] = []
    test_rows: list[LabeledFeatureRow] = []

    for row in sorted(rows, key=lambda item: item.clip_id):
        split_name = assignment[row.recording_id]
        if split_name == "train":
            train_rows.append(row)
        elif split_name == "validation":
            val_rows.append(row)
        else:
            test_rows.append(row)

    split = DatasetSplit(
        train=tuple(train_rows),
        validation=tuple(val_rows),
        test=tuple(test_rows),
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    # Leakage guard.
    train_ids = split.recording_ids("train")
    val_ids = split.recording_ids("validation")
    test_ids = split.recording_ids("test")
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise RuntimeError("Recording leakage detected across splits.")

    return split


def _rows_to_xy(
    rows: tuple[LabeledFeatureRow, ...] | list[LabeledFeatureRow],
) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        return (
            np.empty((0, len(FEATURE_NAMES)), dtype=np.float64),
            np.empty((0,), dtype=object),
        )

    x = np.vstack([row.features for row in rows]).astype(np.float64)
    y = np.asarray([row.label for row in rows], dtype=object)
    return x, y


def build_candidate_estimators(seed: int) -> dict[str, Pipeline]:
    """CPU-friendly candidate models with class-weight imbalance handling."""

    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=4000,
                        solver="lbfgs",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=None,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=300,
                        max_depth=None,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    class_names: tuple[str, ...] = DEFENCE_NOISE_CATEGORIES,
    inference_seconds: list[float] | None = None,
    audio_seconds: list[float] | None = None,
) -> dict[str, Any]:
    labels = list(class_names)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    per_class = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    }

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_recall_fscore_support(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )[0]
        ),
        "macro_recall": float(
            precision_recall_fscore_support(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )[1]
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "per_class": per_class,
        "confusion_matrix": {
            truth: {
                pred: int(matrix[i, j])
                for j, pred in enumerate(labels)
            }
            for i, truth in enumerate(labels)
        },
    }

    if inference_seconds:
        sorted_times = sorted(inference_seconds)
        p95_index = min(
            len(sorted_times) - 1,
            max(0, int(np.ceil(0.95 * len(sorted_times)) - 1)),
        )
        metrics["mean_inference_s"] = float(np.mean(inference_seconds))
        metrics["p95_inference_s"] = float(sorted_times[p95_index])
        if audio_seconds:
            rtfs = [
                duration / audio_s
                for duration, audio_s in zip(inference_seconds, audio_seconds)
                if audio_s > 0.0
            ]
            metrics["mean_rtf"] = float(np.mean(rtfs)) if rtfs else 0.0

    return metrics


def _predict_rows(
    estimator: Pipeline,
    rows: tuple[LabeledFeatureRow, ...] | list[LabeledFeatureRow],
) -> tuple[np.ndarray, np.ndarray, list[float], list[float]]:
    x, y_true = _rows_to_xy(rows)
    inference_seconds: list[float] = []
    audio_seconds: list[float] = []
    predictions: list[str] = []

    for index, row in enumerate(rows):
        start = time.perf_counter()
        pred = estimator.predict(x[index : index + 1])[0]
        elapsed = time.perf_counter() - start
        predictions.append(str(pred))
        inference_seconds.append(elapsed)
        audio_seconds.append(row.num_samples / max(row.sample_rate, 1))

    return y_true, np.asarray(predictions, dtype=object), inference_seconds, audio_seconds


def evaluate_v1_on_rows(
    rows: tuple[LabeledFeatureRow, ...] | list[LabeledFeatureRow],
    dataset: ZipManifestDataset,
    clips_by_id: dict[str, CorpusClip],
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> dict[str, Any]:
    """Run the preserved rule-based v1 classifier on the same clip set."""

    classifier = NoiseClassifier(sample_rate=sample_rate)
    y_true: list[str] = []
    y_pred: list[str] = []
    inference_seconds: list[float] = []
    audio_seconds: list[float] = []

    for row in rows:
        clip = clips_by_id[row.clip_id]
        audio, audio_sr = dataset.load_audio(clip.source)
        if audio_sr != sample_rate:
            raise ValueError(
                f"Sample rate mismatch for v1 comparison on {row.clip_id}"
            )

        start = time.perf_counter()
        result = classifier.classify(audio)
        elapsed = time.perf_counter() - start

        y_true.append(row.label)
        y_pred.append(result.predicted_class)
        inference_seconds.append(elapsed)
        audio_seconds.append(audio.size / max(audio_sr, 1))

    # Map unknown predictions as incorrect vs labelled classes via metrics
    # that only score exact label matches among defence categories.
    labelled_pred = [
        pred if pred in DEFENCE_NOISE_CATEGORIES else "__unknown__"
        for pred in y_pred
    ]
    metrics = evaluate_predictions(
        np.asarray(y_true, dtype=object),
        np.asarray(labelled_pred, dtype=object),
        class_names=DEFENCE_NOISE_CATEGORIES,
        inference_seconds=inference_seconds,
        audio_seconds=audio_seconds,
    )
    metrics["unknown_rate"] = float(
        sum(1 for pred in y_pred if pred == "unknown") / max(len(y_pred), 1)
    )
    metrics["raw_predicted_distribution"] = dict(Counter(y_pred))
    return metrics


def train_and_select_models(
    split: DatasetSplit,
    *,
    seed: int = DEFAULT_SEED,
    output_dir: Path,
    dataset: ZipManifestDataset,
    clips: tuple[CorpusClip, ...] | list[CorpusClip],
) -> tuple[SupervisedNoiseClassifier, TrainingReport, CandidateResult]:
    """Train candidates, select by validation macro F1, evaluate on test."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train = _rows_to_xy(split.train)
    candidates = build_candidate_estimators(seed)
    results: list[CandidateResult] = []

    for model_name, estimator in candidates.items():
        estimator.fit(x_train, y_train)

        y_val_true, y_val_pred, val_times, val_audio = _predict_rows(
            estimator,
            split.validation,
        )
        val_metrics = evaluate_predictions(
            y_val_true,
            y_val_pred,
            inference_seconds=val_times,
            audio_seconds=val_audio,
        )

        y_test_true, y_test_pred, test_times, test_audio = _predict_rows(
            estimator,
            split.test,
        )
        test_metrics = evaluate_predictions(
            y_test_true,
            y_test_pred,
            inference_seconds=test_times,
            audio_seconds=test_audio,
        )

        model_path = output_dir / f"candidate_{model_name}.joblib"
        joblib.dump(estimator, model_path)
        size_bytes = model_path.stat().st_size

        results.append(
            CandidateResult(
                model_name=model_name,
                validation_metrics=val_metrics,
                test_metrics=test_metrics,
                model_size_bytes=size_bytes,
                estimator=estimator,
            )
        )

    selected = max(
        results,
        key=lambda item: (
            item.validation_metrics["macro_f1"],
            item.test_metrics["macro_f1"],
            -item.model_size_bytes,
        ),
    )

    clips_by_id = {clip.clip_id: clip for clip in clips}
    v1_metrics = evaluate_v1_on_rows(
        split.test,
        dataset,
        clips_by_id,
    )

    split_info = {
        "seed": seed,
        "train_ratio": split.train_ratio,
        "val_ratio": split.val_ratio,
        "test_ratio": split.test_ratio,
        "num_train_clips": len(split.train),
        "num_validation_clips": len(split.validation),
        "num_test_clips": len(split.test),
        "num_train_recordings": len(split.recording_ids("train")),
        "num_validation_recordings": len(split.recording_ids("validation")),
        "num_test_recordings": len(split.recording_ids("test")),
        "class_distribution": split.class_distribution(),
        "grouping": (
            "recording_source_id: ESC-50 clip id / firearm UUID / "
            "unique sample_id for drone"
        ),
    }

    report = TrainingReport(
        rules_version=TRAINING_RULES_VERSION,
        seed=seed,
        feature_version=FEATURE_VERSION,
        class_names=DEFENCE_NOISE_CATEGORIES,
        class_distribution=split.class_distribution(),
        split_info=split_info,
        candidates=[
            {
                "model_name": item.model_name,
                "validation_metrics": item.validation_metrics,
                "test_metrics": item.test_metrics,
                "model_size_bytes": item.model_size_bytes,
            }
            for item in results
        ],
        selected_model=selected.model_name,
        selection_criterion="validation_macro_f1",
        v1_comparison=v1_metrics,
    )

    meta = SupervisedModelMeta(
        bundle_version=MODEL_BUNDLE_VERSION,
        model_name=selected.model_name,
        feature_version=FEATURE_VERSION,
        feature_names=FEATURE_NAMES,
        class_names=DEFENCE_NOISE_CATEGORIES,
        sample_rate=DEFAULT_SAMPLE_RATE,
        window_ms=DEFAULT_WINDOW_MS,
        hop_ms=DEFAULT_HOP_MS,
        silence_threshold_db=-55.0,
        seed=seed,
        training_config={
            "class_weight": "balanced / balanced_subsample",
            "candidates": list(candidates),
            "imbalance_strategy": "sklearn class_weight; no class deletion",
            "selection_criterion": "validation_macro_f1",
            "rules_version": TRAINING_RULES_VERSION,
        },
        split_info=split_info,
    )

    classifier = SupervisedNoiseClassifier(
        selected.estimator,
        class_names=DEFENCE_NOISE_CATEGORIES,
        sample_rate=DEFAULT_SAMPLE_RATE,
        model_name=selected.model_name,
        meta=meta,
    )

    selected_dir = output_dir / "selected_model"
    classifier.save(selected_dir)

    return classifier, report, selected


def save_feature_cache(
    path: Path,
    rows: list[LabeledFeatureRow],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        clip_ids=np.asarray([row.clip_id for row in rows], dtype=object),
        source_ids=np.asarray([row.source_id for row in rows], dtype=object),
        recording_ids=np.asarray(
            [row.recording_id for row in rows],
            dtype=object,
        ),
        labels=np.asarray([row.label for row in rows], dtype=object),
        features=np.vstack([row.features for row in rows]),
        num_samples=np.asarray(
            [row.num_samples for row in rows],
            dtype=np.int64,
        ),
        sample_rates=np.asarray(
            [row.sample_rate for row in rows],
            dtype=np.int64,
        ),
        feature_names=np.asarray(FEATURE_NAMES, dtype=object),
        feature_version=np.asarray([FEATURE_VERSION], dtype=object),
    )


def load_feature_cache(path: Path) -> list[LabeledFeatureRow]:
    payload = np.load(Path(path), allow_pickle=True)
    rows: list[LabeledFeatureRow] = []

    for index in range(len(payload["clip_ids"])):
        rows.append(
            LabeledFeatureRow(
                clip_id=str(payload["clip_ids"][index]),
                source_id=str(payload["source_ids"][index]),
                recording_id=str(payload["recording_ids"][index]),
                label=str(payload["labels"][index]),
                features=np.asarray(
                    payload["features"][index],
                    dtype=np.float64,
                ),
                num_samples=int(payload["num_samples"][index]),
                sample_rate=int(payload["sample_rates"][index]),
            )
        )

    return rows


def build_training_corpus(
    metadata_path: Path,
) -> tuple[CorpusClip, ...]:
    return build_defence_noise_corpus(
        metadata_path,
        categories=DEFENCE_NOISE_CATEGORIES,
    )
