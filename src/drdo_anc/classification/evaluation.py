"""Benchmark and real-corpus evaluation helpers for the noise classifier."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

import numpy as np

from drdo_anc.benchmark.case import BenchmarkCase
from drdo_anc.benchmark.config import STREAMING_CHUNK_SIZES
from drdo_anc.benchmark.evaluation_manifest import EvaluationManifest
from drdo_anc.dataset.manifest import (
    load_metadata_rows,
    row_to_source_sample,
)
from drdo_anc.dataset.source_pool import (
    DEVELOPMENT_NOISE_CATEGORIES,
    NOISE_CATEGORY_SOURCES,
    is_noise_source,
    source_sample_id_from_row,
)
from drdo_anc.dataset.source_sample import SourceSample
from drdo_anc.dataset.zip_manifest_dataset import ZipManifestDataset

from .categories import DEFENCE_NOISE_CATEGORIES, NOISE_CLASSES, UNKNOWN_CLASS
from .classifier import NoiseClassifier
from .features import DEFAULT_SAMPLE_RATE
from .base import ClassificationResult


CORPUS_EVAL_RULES_VERSION = "noise-classifier-corpus-v1"
CORPUS_EVAL_SPLIT_NAME = "sih26_defence_noise_corpus"


@dataclass(frozen=True)
class ClassifierCaseResult:
    case_id: str
    noise_source_id: str
    ground_truth: str
    predicted_class: str
    probabilities: dict[str, float]
    inference_s: float
    num_samples: int
    sample_rate: int


@dataclass
class ClassifierBenchmarkReport:
    manifest_rules_version: str
    manifest_split_name: str
    case_results: list[ClassifierCaseResult] = field(
        default_factory=list,
    )

    def confusion_matrix(self) -> dict[str, dict[str, int]]:
        matrix: dict[str, dict[str, int]] = {
            truth: {pred: 0 for pred in NOISE_CLASSES}
            for truth in DEFENCE_NOISE_CATEGORIES
        }

        for result in self.case_results:
            matrix[result.ground_truth][result.predicted_class] += 1

        return matrix

    def per_class_metrics(self) -> dict[str, dict[str, float]]:
        return _per_class_metrics(self.case_results)

    def overall_accuracy(self) -> float:
        return _labelled_accuracy(self.case_results)

    def unknown_predictions(self) -> list[ClassifierCaseResult]:
        return [
            result
            for result in self.case_results
            if result.predicted_class == UNKNOWN_CLASS
        ]

    def timing_summary(self) -> dict[str, float]:
        return _timing_summary(self.case_results)


@dataclass(frozen=True)
class CorpusClip:
    """One labelled noise clip in a deterministic corpus evaluation set."""

    clip_id: str
    ground_truth: str
    source: SourceSample


@dataclass(frozen=True)
class CorpusCaseResult:
    clip_id: str
    noise_source_id: str
    ground_truth: str
    predicted_class: str
    probabilities: dict[str, float]
    confidence: float
    top2_margin: float
    inference_s: float
    num_samples: int
    sample_rate: int


@dataclass
class CorpusEvaluationReport:
    rules_version: str
    split_name: str
    case_results: list[CorpusCaseResult] = field(default_factory=list)

    def clips_per_class(self) -> dict[str, int]:
        counts = Counter(result.ground_truth for result in self.case_results)
        return {
            label: int(counts.get(label, 0))
            for label in DEFENCE_NOISE_CATEGORIES
        }

    def predicted_class_distribution(self) -> dict[str, int]:
        counts = Counter(
            result.predicted_class for result in self.case_results
        )
        return {
            label: int(counts.get(label, 0))
            for label in NOISE_CLASSES
        }

    def confusion_matrix(self) -> dict[str, dict[str, int]]:
        matrix: dict[str, dict[str, int]] = {
            truth: {pred: 0 for pred in NOISE_CLASSES}
            for truth in DEFENCE_NOISE_CATEGORIES
        }

        for result in self.case_results:
            matrix[result.ground_truth][result.predicted_class] += 1

        return matrix

    def per_class_metrics(self) -> dict[str, dict[str, float]]:
        return _per_class_metrics(self.case_results)

    def overall_accuracy(self) -> float:
        """Fraction of labelled clips predicted as the true class.

        ``unknown`` is never counted as correct for a labelled noise class.
        """

        return _labelled_accuracy(self.case_results)

    def macro_f1(self) -> float:
        metrics = self.per_class_metrics()
        if not metrics:
            return 0.0

        return mean(item["f1"] for item in metrics.values())

    def unknown_rate(self) -> float:
        if not self.case_results:
            return 0.0

        unknown = sum(
            1
            for result in self.case_results
            if result.predicted_class == UNKNOWN_CLASS
        )
        return unknown / len(self.case_results)

    def timing_summary(self) -> dict[str, float]:
        return _timing_summary(self.case_results)

    def qualitative_examples(
        self,
        *,
        limit: int = 12,
    ) -> list[dict[str, float | str]]:
        """Representative examples covering correct, wrong, and unknown."""

        by_bucket: dict[str, list[CorpusCaseResult]] = {
            "correct": [],
            "wrong": [],
            "unknown": [],
        }

        for result in self.case_results:
            if result.predicted_class == UNKNOWN_CLASS:
                by_bucket["unknown"].append(result)
            elif result.predicted_class == result.ground_truth:
                by_bucket["correct"].append(result)
            else:
                by_bucket["wrong"].append(result)

        for bucket in by_bucket.values():
            bucket.sort(key=lambda item: (item.ground_truth, item.clip_id))

        selected: list[CorpusCaseResult] = []
        seen: set[str] = set()

        def _take_round_robin(
            candidates: list[CorpusCaseResult],
            quota: int,
        ) -> None:
            by_truth: dict[str, list[CorpusCaseResult]] = {
                label: [] for label in DEFENCE_NOISE_CATEGORIES
            }
            for result in candidates:
                by_truth[result.ground_truth].append(result)

            taken = 0
            index = 0
            while taken < quota:
                progressed = False
                for label in DEFENCE_NOISE_CATEGORIES:
                    pool = by_truth[label]
                    if index >= len(pool):
                        continue
                    candidate = pool[index]
                    if candidate.clip_id in seen:
                        continue
                    selected.append(candidate)
                    seen.add(candidate.clip_id)
                    taken += 1
                    progressed = True
                    if taken >= quota:
                        break
                if not progressed:
                    break
                index += 1

        per_bucket = max(1, limit // 3)
        for bucket_name in ("correct", "wrong", "unknown"):
            _take_round_robin(by_bucket[bucket_name], per_bucket)

        if len(selected) < limit:
            leftovers = [
                result
                for result in self.case_results
                if result.clip_id not in seen
            ]
            leftovers.sort(key=lambda item: (item.ground_truth, item.clip_id))
            for result in leftovers:
                if len(selected) >= limit:
                    break
                selected.append(result)
                seen.add(result.clip_id)

        return [
            {
                "clip_id": result.clip_id,
                "actual": result.ground_truth,
                "predicted": result.predicted_class,
                "confidence": result.confidence,
                "top2_margin": result.top2_margin,
            }
            for result in selected[:limit]
        ]

    def systematic_misclassifications(
        self,
        *,
        top_n: int = 10,
    ) -> list[dict[str, int | str]]:
        """Most frequent actual→predicted error pairs (excluding unknowns)."""

        pairs = Counter(
            (result.ground_truth, result.predicted_class)
            for result in self.case_results
            if result.predicted_class != result.ground_truth
            and result.predicted_class != UNKNOWN_CLASS
        )

        ranked = sorted(
            pairs.items(),
            key=lambda item: (-item[1], item[0][0], item[0][1]),
        )

        return [
            {
                "actual": actual,
                "predicted": predicted,
                "count": count,
            }
            for (actual, predicted), count in ranked[:top_n]
        ]


def _confidence_and_margin(
    probabilities: dict[str, float],
) -> tuple[float, float]:
    ranked = sorted(probabilities.values(), reverse=True)
    confidence = float(ranked[0]) if ranked else 0.0
    second = float(ranked[1]) if len(ranked) > 1 else 0.0
    return confidence, confidence - second


def _labelled_accuracy(results: list) -> float:
    if not results:
        return 0.0

    correct = sum(
        1
        for result in results
        if result.ground_truth == result.predicted_class
    )
    return correct / len(results)


def _per_class_metrics(results: list) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}

    for truth in DEFENCE_NOISE_CATEGORIES:
        predicted_positive = [
            result
            for result in results
            if result.predicted_class == truth
        ]
        actual_positive = [
            result
            for result in results
            if result.ground_truth == truth
        ]
        true_positive = sum(
            1
            for result in results
            if result.ground_truth == truth
            and result.predicted_class == truth
        )

        precision = (
            true_positive / len(predicted_positive)
            if predicted_positive
            else 0.0
        )
        recall = (
            true_positive / len(actual_positive)
            if actual_positive
            else 0.0
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0.0
            else 0.0
        )

        metrics[truth] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(len(actual_positive)),
        }

    return metrics


def _timing_summary(results: list) -> dict[str, float]:
    if not results:
        return {}

    durations = [result.inference_s for result in results]
    audio_seconds = [
        result.num_samples / result.sample_rate
        for result in results
        if result.sample_rate > 0
    ]
    sorted_durations = sorted(durations)
    p95_index = min(
        len(sorted_durations) - 1,
        max(0, int(np.ceil(0.95 * len(sorted_durations)) - 1)),
    )

    return {
        "mean_inference_s": mean(durations),
        "p95_inference_s": float(sorted_durations[p95_index]),
        "total_inference_s": float(sum(durations)),
        "mean_audio_s": mean(audio_seconds) if audio_seconds else 0.0,
        "mean_rtf": (
            mean(
                duration / audio_s
                for duration, audio_s in zip(durations, audio_seconds)
                if audio_s > 0.0
            )
            if audio_seconds
            else 0.0
        ),
    }


def required_archives_for_defence_noise(
    metadata_path: Path,
    *,
    categories: tuple[str, ...] = DEVELOPMENT_NOISE_CATEGORIES,
) -> dict[str, set[str]]:
    """Map each defence category to the archive ZIP names it requires."""

    rows = load_metadata_rows(metadata_path)
    required: dict[str, set[str]] = {category: set() for category in categories}

    for category in categories:
        dataset_source = NOISE_CATEGORY_SOURCES[category]
        for row in rows:
            if not is_noise_source(row):
                continue
            if row["dataset_source"] != dataset_source:
                continue
            required[category].add(row["archive_name"])

    return required


def find_missing_corpus_archives(
    metadata_path: Path,
    *,
    archive_dir: Path | None = None,
    repo_id: str | None = None,
    categories: tuple[str, ...] = DEVELOPMENT_NOISE_CATEGORIES,
) -> list[str]:
    """
    Return missing archive names required for real defence-noise evaluation.

    Local ``archive_dir`` is checked first. When ``repo_id`` is set, an
    archive is considered available if it exists in the Hugging Face cache
    (no network download is triggered by this check).
    """

    required = required_archives_for_defence_noise(
        metadata_path,
        categories=categories,
    )
    all_archives = sorted(
        {
            archive
            for archives in required.values()
            for archive in archives
        }
    )

    missing: list[str] = []

    for archive_name in all_archives:
        if archive_dir is not None:
            if (archive_dir / archive_name).exists():
                continue
            missing.append(archive_name)
            continue

        if repo_id is None:
            missing.append(archive_name)
            continue

        if not _archive_present_in_hf_cache(repo_id, archive_name):
            missing.append(archive_name)

    return missing


def _archive_present_in_hf_cache(repo_id: str, archive_name: str) -> bool:
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return False

    cache = scan_cache_dir()

    for repo in cache.repos:
        if repo.repo_id != repo_id or repo.repo_type != "dataset":
            continue

        for revision in repo.revisions:
            for cached_file in revision.files:
                if cached_file.file_name == archive_name:
                    return True

    return False


def build_defence_noise_corpus(
    metadata_path: Path,
    *,
    categories: tuple[str, ...] = DEVELOPMENT_NOISE_CATEGORIES,
    max_per_class: int | None = None,
) -> tuple[CorpusClip, ...]:
    """
    Build a deterministic labelled evaluation set of real noise clips.

    Clips are selected from metadata only (no audio I/O), sorted by
    ``sample_id``, optionally capped per class with ``max_per_class``.
    """

    if max_per_class is not None and max_per_class <= 0:
        raise ValueError("max_per_class must be positive when provided.")

    rows = load_metadata_rows(metadata_path)
    clips: list[CorpusClip] = []

    for category in categories:
        dataset_source = NOISE_CATEGORY_SOURCES[category]
        category_rows = sorted(
            [
                row
                for row in rows
                if is_noise_source(row)
                and row["dataset_source"] == dataset_source
            ],
            key=source_sample_id_from_row,
        )

        if max_per_class is not None:
            category_rows = category_rows[:max_per_class]

        for index, row in enumerate(category_rows, start=1):
            source = row_to_source_sample(row)
            clips.append(
                CorpusClip(
                    clip_id=f"{category}_{index:05d}",
                    ground_truth=category,
                    source=source,
                )
            )

    return tuple(clips)


def classify_audio_streaming(
    classifier: NoiseClassifier,
    audio: np.ndarray,
    *,
    chunk_sizes: tuple[int, ...] = STREAMING_CHUNK_SIZES,
) -> tuple[ClassificationResult, float]:
    """Classify mono audio with the project streaming chunk cycle."""

    classifier.reset()
    start = time.perf_counter()

    offset = 0
    chunk_index = 0

    while offset < audio.size:
        chunk_size = chunk_sizes[chunk_index % len(chunk_sizes)]
        classifier.process_chunk(audio[offset : offset + chunk_size])
        offset += chunk_size
        chunk_index += 1

    result = classifier.classify_buffered()
    elapsed = time.perf_counter() - start
    return result, elapsed


def classify_case_noise(
    classifier: NoiseClassifier,
    dataset: ZipManifestDataset,
    case: BenchmarkCase,
    *,
    chunk_sizes: tuple[int, ...] = STREAMING_CHUNK_SIZES,
) -> tuple[ClassificationResult, float]:
    """
    Classify the benchmark noise source for one case using streaming chunks.

    Ground-truth labels come from ``case.noise_category``. Audio is loaded
    from the deterministic noise source referenced by the case.
    """

    noise, sample_rate = dataset.load_audio(case.noise_source)

    if sample_rate != classifier.sample_rate:
        raise ValueError(
            f"Sample rate mismatch for {case.case_id}: "
            f"noise={sample_rate}, classifier={classifier.sample_rate}"
        )

    return classify_audio_streaming(
        classifier,
        noise,
        chunk_sizes=chunk_sizes,
    )


def run_classifier_benchmark(
    manifest: EvaluationManifest,
    dataset: ZipManifestDataset,
    *,
    classifier: NoiseClassifier | None = None,
    chunk_sizes: tuple[int, ...] = STREAMING_CHUNK_SIZES,
) -> ClassifierBenchmarkReport:
    """Evaluate the classifier across all manifest cases."""

    if classifier is None:
        classifier = NoiseClassifier()

    report = ClassifierBenchmarkReport(
        manifest_rules_version=manifest.rules_version,
        manifest_split_name=manifest.split_name,
    )

    for case in manifest.cases:
        noise, sample_rate = dataset.load_audio(case.noise_source)
        result, elapsed = classify_audio_streaming(
            classifier,
            noise,
            chunk_sizes=chunk_sizes,
        )

        report.case_results.append(
            ClassifierCaseResult(
                case_id=case.case_id,
                noise_source_id=case.noise_source.sample_id,
                ground_truth=case.noise_category,
                predicted_class=result.predicted_class,
                probabilities=result.probabilities,
                inference_s=elapsed,
                num_samples=int(noise.size),
                sample_rate=sample_rate,
            )
        )

    return report


def run_noise_corpus_evaluation(
    clips: tuple[CorpusClip, ...] | list[CorpusClip],
    dataset: ZipManifestDataset,
    *,
    classifier: NoiseClassifier | None = None,
    chunk_sizes: tuple[int, ...] = STREAMING_CHUNK_SIZES,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    rules_version: str = CORPUS_EVAL_RULES_VERSION,
    split_name: str = CORPUS_EVAL_SPLIT_NAME,
) -> CorpusEvaluationReport:
    """Run NoiseClassifier over a deterministic real-noise corpus set."""

    if classifier is None:
        classifier = NoiseClassifier(sample_rate=sample_rate)

    report = CorpusEvaluationReport(
        rules_version=rules_version,
        split_name=split_name,
    )

    for clip in clips:
        audio, audio_sr = dataset.load_audio(clip.source)

        if audio_sr != classifier.sample_rate:
            raise ValueError(
                f"Sample rate mismatch for {clip.clip_id}: "
                f"audio={audio_sr}, classifier={classifier.sample_rate}"
            )

        result, elapsed = classify_audio_streaming(
            classifier,
            audio,
            chunk_sizes=chunk_sizes,
        )
        confidence, margin = _confidence_and_margin(result.probabilities)

        report.case_results.append(
            CorpusCaseResult(
                clip_id=clip.clip_id,
                noise_source_id=clip.source.sample_id,
                ground_truth=clip.ground_truth,
                predicted_class=result.predicted_class,
                probabilities=result.probabilities,
                confidence=confidence,
                top2_margin=margin,
                inference_s=elapsed,
                num_samples=int(audio.size),
                sample_rate=audio_sr,
            )
        )

    return report
