"""Offline noise-aware enhancement experiment runner."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
import torch
from df import enhance

from drdo_anc.benchmark.case import BenchmarkCase
from drdo_anc.benchmark.evaluation_manifest import EvaluationManifest
from drdo_anc.benchmark.manifest_benchmark import resample_mixture_for_enhancer
from drdo_anc.benchmark.mixture import MixtureGenerator
from drdo_anc.classification.base import NoiseClassifierBase
from drdo_anc.evaluation import evaluate_pair

from .classical import spectral_subtraction
from .strategies import (
    DEFAULT_STRATEGY_MAP,
    STRATEGY_CLASSICAL,
    STRATEGY_DF3_FULL,
    STRATEGY_NOISY,
    EnhancementStrategy,
    select_strategy,
)


@dataclass
class OfflineDF3Session:
    """Offline DeepFilterNet session used only by this experiment."""

    model: Any
    df_state: Any
    sample_rate_hz: int
    name: str


def load_offline_df3_session() -> OfflineDF3Session:
    """
    Load DeepFilterNet for offline ``df.enhance`` calls.

    Uses the public ``df`` API only (including documented ``atten_lim_db``).
    Does not modify ``DeepFilterNetEnhancer`` or the model registry.
    Compatible with both 3-tuple and 4-tuple ``init_df()`` return values.
    """

    from df import init_df

    loaded = init_df()
    if len(loaded) == 4:
        model, df_state, suffix, _epoch = loaded
    elif len(loaded) == 3:
        model, df_state, suffix = loaded
    else:
        raise RuntimeError(
            f"Unexpected init_df() return length: {len(loaded)}"
        )

    sample_rate = int(df_state.sr())
    if sample_rate != 48_000:
        raise ValueError(
            f"Expected DeepFilterNet sample rate 48000 Hz, got {sample_rate}"
        )

    return OfflineDF3Session(
        model=model,
        df_state=df_state,
        sample_rate_hz=sample_rate,
        name=str(suffix),
    )


EXPERIMENT_VERSION = "noise-aware-enhancement-v1"


@dataclass(frozen=True)
class SystemCaseMetrics:
    system_id: str
    strategy_id: str
    snr: float | None
    si_sdr: float | None
    stoi: float | None
    pesq: float | None
    inference_s: float | None
    rtf: float | None
    status: str = "success"
    error: str | None = None


@dataclass(frozen=True)
class NoiseAwareCaseResult:
    case_id: str
    true_noise_class: str
    snr_db: float
    predicted_noise_class: str
    classifier_confidence: float
    adaptive_strategy_id: str
    adaptive_fallback_reason: str | None
    classification_correct: bool
    systems: dict[str, SystemCaseMetrics]


@dataclass
class NoiseAwareExperimentReport:
    experiment_version: str
    manifest_rules_version: str
    manifest_split_name: str
    classifier_model_name: str
    strategy_map: dict[str, str]
    case_results: list[NoiseAwareCaseResult] = field(default_factory=list)

    def successful_system_metrics(
        self,
        system_id: str,
    ) -> list[SystemCaseMetrics]:
        metrics: list[SystemCaseMetrics] = []
        for case in self.case_results:
            result = case.systems.get(system_id)
            if result is not None and result.status == "success":
                metrics.append(result)
        return metrics

    def summary_for_system(self, system_id: str) -> dict[str, float]:
        return _aggregate_metrics(self.successful_system_metrics(system_id))

    def summary_by_noise_category(
        self,
        system_id: str,
    ) -> dict[str, dict[str, float]]:
        groups: dict[str, list[SystemCaseMetrics]] = defaultdict(list)
        for case in self.case_results:
            result = case.systems.get(system_id)
            if result is not None and result.status == "success":
                groups[case.true_noise_class].append(result)
        return {
            category: _aggregate_metrics(values)
            for category, values in sorted(groups.items())
        }

    def summary_by_snr(
        self,
        system_id: str,
    ) -> dict[str, dict[str, float]]:
        groups: dict[float, list[SystemCaseMetrics]] = defaultdict(list)
        for case in self.case_results:
            result = case.systems.get(system_id)
            if result is not None and result.status == "success":
                groups[case.snr_db].append(result)
        return {
            _format_snr_key(snr): _aggregate_metrics(values)
            for snr, values in sorted(groups.items())
        }

    def compare_systems(
        self,
        baseline_id: str,
        candidate_id: str,
        metric: str = "si_sdr",
    ) -> dict[str, Any]:
        deltas: list[float] = []
        improved = 0
        degraded = 0
        tied = 0
        per_category: dict[str, list[float]] = defaultdict(list)

        for case in self.case_results:
            base = case.systems.get(baseline_id)
            cand = case.systems.get(candidate_id)
            if (
                base is None
                or cand is None
                or base.status != "success"
                or cand.status != "success"
            ):
                continue
            base_value = getattr(base, metric)
            cand_value = getattr(cand, metric)
            if base_value is None or cand_value is None:
                continue
            delta = float(cand_value - base_value)
            deltas.append(delta)
            per_category[case.true_noise_class].append(delta)
            if delta > 1e-9:
                improved += 1
            elif delta < -1e-9:
                degraded += 1
            else:
                tied += 1

        return {
            "metric": metric,
            "baseline": baseline_id,
            "candidate": candidate_id,
            "num_compared": len(deltas),
            "mean_difference": mean(deltas) if deltas else 0.0,
            "median_difference": median(deltas) if deltas else 0.0,
            "improved": improved,
            "degraded": degraded,
            "tied": tied,
            "per_category_mean_difference": {
                category: mean(values) if values else 0.0
                for category, values in sorted(per_category.items())
            },
        }

    def classification_summary(self) -> dict[str, Any]:
        total = len(self.case_results)
        correct = sum(1 for case in self.case_results if case.classification_correct)
        return {
            "num_cases": total,
            "accuracy": correct / total if total else 0.0,
            "predicted_distribution": _count(
                case.predicted_noise_class for case in self.case_results
            ),
            "true_distribution": _count(
                case.true_noise_class for case in self.case_results
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_version": self.experiment_version,
            "manifest_rules_version": self.manifest_rules_version,
            "manifest_split_name": self.manifest_split_name,
            "classifier_model_name": self.classifier_model_name,
            "strategy_map": self.strategy_map,
            "classification_summary": self.classification_summary(),
            "summaries": {
                system_id: self.summary_for_system(system_id)
                for system_id in (
                    "noisy",
                    "classical_spectral_subtraction",
                    "df3_baseline",
                    "adaptive",
                )
            },
            "summaries_by_noise_category": {
                system_id: self.summary_by_noise_category(system_id)
                for system_id in ("df3_baseline", "adaptive")
            },
            "summaries_by_snr": {
                system_id: self.summary_by_snr(system_id)
                for system_id in ("df3_baseline", "adaptive")
            },
            "comparisons": {
                metric: self.compare_systems("df3_baseline", "adaptive", metric)
                for metric in ("si_sdr", "stoi", "pesq", "snr")
            },
            "case_results": [
                {
                    "case_id": case.case_id,
                    "true_noise_class": case.true_noise_class,
                    "snr_db": case.snr_db,
                    "predicted_noise_class": case.predicted_noise_class,
                    "classifier_confidence": case.classifier_confidence,
                    "adaptive_strategy_id": case.adaptive_strategy_id,
                    "adaptive_fallback_reason": case.adaptive_fallback_reason,
                    "classification_correct": case.classification_correct,
                    "systems": {
                        key: asdict(value) for key, value in case.systems.items()
                    },
                }
                for case in self.case_results
            ],
        }

    def save_json(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def _count(values) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[str(value)] += 1
    return dict(sorted(counts.items()))


def _format_snr_key(snr_db: float) -> str:
    if snr_db == int(snr_db):
        sign = "+" if snr_db > 0 else ""
        return f"{sign}{int(snr_db)} dB"
    return f"{snr_db} dB"


def _aggregate_metrics(results: list[SystemCaseMetrics]) -> dict[str, float]:
    if not results:
        return {}

    summary: dict[str, float] = {}
    for key in ("snr", "si_sdr", "stoi", "pesq"):
        values = [
            getattr(result, key)
            for result in results
            if getattr(result, key) is not None
        ]
        if values:
            summary[f"mean_{key}"] = mean(values)
            summary[f"median_{key}"] = median(values)

    rtfs = [result.rtf for result in results if result.rtf is not None]
    if rtfs:
        summary["mean_rtf"] = mean(rtfs)
        summary["median_rtf"] = median(rtfs)

    inferences = [
        result.inference_s
        for result in results
        if result.inference_s is not None
    ]
    if inferences:
        summary["mean_inference_s"] = mean(inferences)
        summary["median_inference_s"] = median(inferences)

    summary["num_cases"] = float(len(results))
    return summary


def _apply_strategy(
    strategy: EnhancementStrategy,
    noisy_16k: np.ndarray,
    noisy_48k: np.ndarray,
    *,
    df3_session: OfflineDF3Session | None,
) -> tuple[np.ndarray, int]:
    """
    Apply one strategy.

    Returns enhanced mono float32 and the sample rate of the returned audio.
    DF3 paths return 48 kHz; classical/noisy return 16 kHz.
    """

    if strategy.method == "passthrough":
        return noisy_16k.astype(np.float32, copy=True), 16_000

    if strategy.method == "classical_spectral_subtraction":
        return spectral_subtraction(noisy_16k, 16_000), 16_000

    if strategy.method == "df3":
        if df3_session is None:
            raise RuntimeError("DF3 session is not loaded.")

        audio = torch.from_numpy(noisy_48k).float()
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        enhanced = enhance(
            df3_session.model,
            df3_session.df_state,
            audio,
            atten_lim_db=strategy.atten_lim_db,
        )
        array = (
            enhanced.detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
        if array.ndim == 2:
            array = array.squeeze(0)
        return array, 48_000

    raise ValueError(f"Unsupported strategy method: {strategy.method}")


def _evaluate_estimate(
    clean_16k: np.ndarray,
    estimate: np.ndarray,
    estimate_sample_rate: int,
) -> dict[str, float]:
    if estimate_sample_rate == 16_000:
        clean = clean_16k
        est = estimate
        sample_rate = 16_000
    elif estimate_sample_rate == 48_000:
        clean, _, sample_rate = resample_mixture_for_enhancer(
            clean_16k,
            clean_16k,
            16_000,
            48_000,
        )
        est = estimate
    else:
        raise ValueError(f"Unexpected estimate sample rate: {estimate_sample_rate}")

    length = min(len(clean), len(est))
    return evaluate_pair(clean[:length], est[:length], sample_rate)


def run_noise_aware_experiment(
    manifest: EvaluationManifest,
    mixture_generator: MixtureGenerator,
    classifier: NoiseClassifierBase,
    *,
    df3_session: OfflineDF3Session,
    cases: tuple[BenchmarkCase, ...] | None = None,
    strategy_map: dict[str, EnhancementStrategy] | None = None,
) -> NoiseAwareExperimentReport:
    """
    Run the offline controlled experiment on deterministic benchmark cases.

    Systems evaluated on every case:
      * noisy
      * classical_spectral_subtraction
      * df3_baseline (atten_lim_db=None)
      * adaptive (classifier → category strategy using documented atten_lim_db)
    """

    selected_cases = cases or manifest.cases
    mapping = strategy_map or DEFAULT_STRATEGY_MAP
    report = NoiseAwareExperimentReport(
        experiment_version=EXPERIMENT_VERSION,
        manifest_rules_version=manifest.rules_version,
        manifest_split_name=manifest.split_name,
        classifier_model_name=getattr(classifier, "model_name", classifier.__class__.__name__),
        strategy_map={
            key: value.strategy_id for key, value in mapping.items()
        },
    )

    systems_fixed = {
        "noisy": STRATEGY_NOISY,
        "classical_spectral_subtraction": STRATEGY_CLASSICAL,
        "df3_baseline": STRATEGY_DF3_FULL,
    }

    for index, case in enumerate(selected_cases, start=1):
        print(
            f"Case {index}/{len(selected_cases)}: {case.case_id}",
            flush=True,
        )
        mixture = mixture_generator.generate(case)
        clean_16k = mixture.clean.astype(np.float32, copy=False)
        noisy_16k = mixture.noisy.astype(np.float32, copy=False)

        # Classifier operates at its native rate (16 kHz corpus convention).
        classifier.reset()
        classification = classifier.classify(noisy_16k)
        selection = select_strategy(
            classification.predicted_class,
            classification.confidence,
            strategy_map=mapping,
        )

        _, noisy_48k, _ = resample_mixture_for_enhancer(
            clean_16k,
            noisy_16k,
            mixture.sample_rate,
            df3_session.sample_rate_hz,
        )

        system_metrics: dict[str, SystemCaseMetrics] = {}

        for system_id, strategy in systems_fixed.items():
            system_metrics[system_id] = _run_one_system(
                system_id=system_id,
                strategy=strategy,
                clean_16k=clean_16k,
                noisy_16k=noisy_16k,
                noisy_48k=noisy_48k,
                df3_session=df3_session,
            )

        system_metrics["adaptive"] = _run_one_system(
            system_id="adaptive",
            strategy=selection.strategy,
            clean_16k=clean_16k,
            noisy_16k=noisy_16k,
            noisy_48k=noisy_48k,
            df3_session=df3_session,
        )

        report.case_results.append(
            NoiseAwareCaseResult(
                case_id=case.case_id,
                true_noise_class=case.noise_category,
                snr_db=float(case.snr_db),
                predicted_noise_class=selection.predicted_class,
                classifier_confidence=float(selection.confidence),
                adaptive_strategy_id=selection.strategy.strategy_id,
                adaptive_fallback_reason=selection.fallback_reason,
                classification_correct=(
                    selection.predicted_class == case.noise_category
                ),
                systems=system_metrics,
            )
        )

    return report


def _run_one_system(
    *,
    system_id: str,
    strategy: EnhancementStrategy,
    clean_16k: np.ndarray,
    noisy_16k: np.ndarray,
    noisy_48k: np.ndarray,
    df3_session: OfflineDF3Session,
) -> SystemCaseMetrics:
    try:
        start = time.perf_counter()
        estimate, estimate_sr = _apply_strategy(
            strategy,
            noisy_16k,
            noisy_48k,
            df3_session=df3_session,
        )
        inference_s = time.perf_counter() - start
        metrics = _evaluate_estimate(clean_16k, estimate, estimate_sr)
        audio_s = len(noisy_16k) / 16_000.0
        rtf = inference_s / audio_s if audio_s > 0 else None
        return SystemCaseMetrics(
            system_id=system_id,
            strategy_id=strategy.strategy_id,
            snr=metrics["snr"],
            si_sdr=metrics["si_sdr"],
            stoi=metrics["stoi"],
            pesq=metrics["pesq"],
            inference_s=inference_s,
            rtf=rtf,
        )
    except Exception as exc:
        return SystemCaseMetrics(
            system_id=system_id,
            strategy_id=strategy.strategy_id,
            snr=None,
            si_sdr=None,
            stoi=None,
            pesq=None,
            inference_s=None,
            rtf=None,
            status="failed",
            error=str(exc),
        )
