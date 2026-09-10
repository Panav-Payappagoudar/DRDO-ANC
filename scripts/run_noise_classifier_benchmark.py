"""Run the rule-based noise classifier over the development benchmark."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import replace
import urllib.request
from pathlib import Path

from drdo_anc.benchmark import (
    build_development_manifest,
)
from drdo_anc.benchmark.manifest_benchmark import (
    build_manifest_dataset,
    select_smoke_cases,
    validate_development_manifest,
)
from drdo_anc.classification import NOISE_CLASSES
from drdo_anc.classification.evaluation import run_classifier_benchmark
from drdo_anc.dataset.manifest import (
    SIH26_METADATA_FILENAME,
    SIH26_REPO_ID,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "classifier_results"


def _resolve_metadata_path(metadata_path: Path | None) -> Path:
    if metadata_path is not None:
        return metadata_path.resolve()

    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=SIH26_REPO_ID,
            repo_type="dataset",
            filename=SIH26_METADATA_FILENAME,
        )

        return Path(downloaded)
    except Exception:
        url = (
            "https://huggingface.co/datasets/"
            f"{SIH26_REPO_ID}/resolve/main/"
            f"{SIH26_METADATA_FILENAME}"
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / SIH26_METADATA_FILENAME
            with urllib.request.urlopen(url, timeout=180) as resp:
                target.write_bytes(resp.read())

            persistent = (
                PROJECT_ROOT
                / "data"
                / "cache"
                / SIH26_METADATA_FILENAME
            )
            persistent.parent.mkdir(parents=True, exist_ok=True)
            persistent.write_bytes(target.read_bytes())

            return persistent


def _print_report(report) -> None:
    print("\n" + "=" * 70)
    print("NOISE CLASSIFIER BENCHMARK")
    print("=" * 70)
    print(f"Manifest: {report.manifest_rules_version} / {report.manifest_split_name}")
    print(f"Cases:    {len(report.case_results)}")
    print(f"Accuracy: {report.overall_accuracy():.4f}")

    print("\nConfusion matrix (rows=ground truth, cols=predicted):")
    matrix = report.confusion_matrix()
    header = "ground_truth".ljust(20) + "".join(
        label.rjust(18) for label in NOISE_CLASSES
    )
    print(header)
    print("-" * len(header))

    for truth, row in matrix.items():
        counts = "".join(str(row[label]).rjust(18) for label in NOISE_CLASSES)
        print(f"{truth.ljust(20)}{counts}")

    print("\nPer-class metrics:")
    for label, metrics in report.per_class_metrics().items():
        print(
            f"  {label}: "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"f1={metrics['f1']:.4f}, "
            f"support={int(metrics['support'])}"
        )

    timing = report.timing_summary()
    if timing:
        print("\nTiming:")
        print(f"  mean_inference_s: {timing['mean_inference_s']:.6f}")
        print(f"  total_inference_s: {timing['total_inference_s']:.6f}")
        print(f"  mean_rtf: {timing['mean_rtf']:.6f}")

    unknown_cases = report.unknown_predictions()
    print(f"\nUnknown predictions: {len(unknown_cases)}")

    if unknown_cases:
        print("Cases classified as unknown:")
        for result in unknown_cases:
            print(
                f"  {result.case_id} "
                f"(truth={result.ground_truth}, "
                f"noise_source={result.noise_source_id})"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the rule-based noise classifier on the "
            "deterministic development benchmark."
        ),
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=None,
        help="Path to metadata.csv (defaults to HF download/cache).",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=None,
        help="Directory containing dataset ZIP archives.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run only the approved 2-case smoke subset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for JSON report output.",
    )

    args = parser.parse_args()

    metadata_path = _resolve_metadata_path(args.metadata_path)
    manifest = build_development_manifest(metadata_path)
    validate_development_manifest(manifest)

    if args.smoke:
        manifest = replace(
            manifest,
            cases=select_smoke_cases(manifest),
        )

    dataset = build_manifest_dataset(
        manifest,
        metadata_path,
        archive_dir=args.archive_dir,
    )

    report = run_classifier_benchmark(manifest, dataset)

    _print_report(report)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "noise_classifier_v1_report.json"
    payload = {
        "classifier": "noise_classifier_v1",
        "manifest_rules_version": report.manifest_rules_version,
        "manifest_split_name": report.manifest_split_name,
        "num_cases": len(report.case_results),
        "overall_accuracy": report.overall_accuracy(),
        "confusion_matrix": report.confusion_matrix(),
        "per_class_metrics": report.per_class_metrics(),
        "timing": report.timing_summary(),
        "unknown_case_ids": [
            result.case_id for result in report.unknown_predictions()
        ],
        "case_results": [
            {
                "case_id": result.case_id,
                "noise_source_id": result.noise_source_id,
                "ground_truth": result.ground_truth,
                "predicted_class": result.predicted_class,
                "probabilities": result.probabilities,
                "inference_s": result.inference_s,
                "num_samples": result.num_samples,
                "sample_rate": result.sample_rate,
            }
            for result in report.case_results
        ],
    }

    output_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote report: {output_path}")


if __name__ == "__main__":
    main()
