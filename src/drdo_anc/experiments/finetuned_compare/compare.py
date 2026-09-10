"""Head-to-head comparison of two ManifestBenchmarkReport payloads."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

EXPERIMENT_VERSION = "dfn3-finetuned-compare-v1"
COMPARISON_METRICS = ("si_sdr", "stoi", "pesq", "snr")
PRETRAINED_MODEL_NAME = "DeepFilterNet3"
FINETUNED_MODEL_NAME = "DeepFilterNet3-Finetuned"


def load_benchmark_report(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _successful_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in report.get("case_results", [])
        if row.get("status") == "success"
    ]


def _index_rows(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["case_id"]), str(row["mode"]))
        indexed[key] = row
    return indexed


def _summarize_deltas(deltas: list[float]) -> dict[str, float | int]:
    improved = sum(1 for value in deltas if value > 1e-9)
    degraded = sum(1 for value in deltas if value < -1e-9)
    tied = len(deltas) - improved - degraded
    return {
        "num_compared": len(deltas),
        "mean_difference": mean(deltas) if deltas else 0.0,
        "median_difference": median(deltas) if deltas else 0.0,
        "improved": improved,
        "degraded": degraded,
        "tied": tied,
    }


def compare_benchmark_reports(
    baseline_report: dict[str, Any],
    candidate_report: dict[str, Any],
    *,
    baseline_name: str = PRETRAINED_MODEL_NAME,
    candidate_name: str = FINETUNED_MODEL_NAME,
    metrics: tuple[str, ...] = COMPARISON_METRICS,
) -> dict[str, Any]:
    """Pair successful rows by (case_id, mode) and compute candidate − baseline."""

    baseline_rows = _index_rows(_successful_rows(baseline_report))
    candidate_rows = _index_rows(_successful_rows(candidate_report))
    shared_keys = sorted(set(baseline_rows) & set(candidate_rows))

    metric_summaries: dict[str, dict[str, Any]] = {}
    by_mode: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    by_snr: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    by_category: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    for metric in metrics:
        overall: list[float] = []
        mode_groups: dict[str, list[float]] = defaultdict(list)
        snr_groups: dict[str, list[float]] = defaultdict(list)
        category_groups: dict[str, list[float]] = defaultdict(list)

        for case_id, mode in shared_keys:
            base = baseline_rows[(case_id, mode)]
            cand = candidate_rows[(case_id, mode)]
            base_value = base.get(metric)
            cand_value = cand.get(metric)
            if base_value is None or cand_value is None:
                continue
            delta = float(cand_value) - float(base_value)
            overall.append(delta)
            mode_groups[mode].append(delta)
            snr_key = f"{float(base['snr_db']):.0f}dB"
            snr_groups[snr_key].append(delta)
            category_groups[str(base["noise_category"])].append(delta)

        metric_summaries[metric] = _summarize_deltas(overall)
        for mode, values in sorted(mode_groups.items()):
            by_mode[mode][metric] = _summarize_deltas(values)
        for snr_key, values in sorted(snr_groups.items()):
            by_snr[snr_key][metric] = _summarize_deltas(values)
        for category, values in sorted(category_groups.items()):
            by_category[category][metric] = _summarize_deltas(values)

    si_sdr = metric_summaries.get("si_sdr", {})
    recommendation = (
        "do_not_replace_pretrained"
        if float(si_sdr.get("mean_difference", 0.0)) <= 0.0
        else "candidate_beats_pretrained_on_mean_si_sdr"
    )

    return {
        "experiment_version": EXPERIMENT_VERSION,
        "baseline_model": baseline_name,
        "candidate_model": candidate_name,
        "num_paired_rows": len(shared_keys),
        "baseline_successful_rows": len(baseline_rows),
        "candidate_successful_rows": len(candidate_rows),
        "metrics": metric_summaries,
        "by_mode": dict(by_mode),
        "by_snr": dict(by_snr),
        "by_noise_category": dict(by_category),
        "recommendation": recommendation,
    }


def save_comparison_report(report: dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
