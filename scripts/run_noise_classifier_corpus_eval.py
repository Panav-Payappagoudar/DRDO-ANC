"""Evaluate Noise Classifier v1 on the real SIH-26 defence-noise corpus."""

from __future__ import annotations

import argparse
import json
import tempfile
import urllib.request
from pathlib import Path

from drdo_anc.classification import NOISE_CLASSES
from drdo_anc.classification.evaluation import (
    CORPUS_EVAL_RULES_VERSION,
    CORPUS_EVAL_SPLIT_NAME,
    build_defence_noise_corpus,
    find_missing_corpus_archives,
    run_noise_corpus_evaluation,
)
from drdo_anc.dataset.manifest import (
    SIH26_METADATA_FILENAME,
    SIH26_REPO_ID,
)
from drdo_anc.dataset.source_pool import DEVELOPMENT_NOISE_CATEGORIES
from drdo_anc.dataset.zip_manifest_dataset import ZipManifestDataset


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
                PROJECT_ROOT / "data" / "cache" / SIH26_METADATA_FILENAME
            )
            persistent.parent.mkdir(parents=True, exist_ok=True)
            persistent.write_bytes(target.read_bytes())
            return persistent


def _print_report(report) -> None:
    print("\n" + "=" * 70)
    print("NOISE CLASSIFIER REAL CORPUS EVALUATION")
    print("=" * 70)
    print(f"Rules:    {report.rules_version}")
    print(f"Split:    {report.split_name}")
    print(f"Clips:    {len(report.case_results)}")
    print(f"Accuracy: {report.overall_accuracy():.4f}  (unknown != correct)")
    print(f"Macro F1: {report.macro_f1():.4f}")
    print(f"Unknown:  {report.unknown_rate():.4f}")

    print("\nClips per class:")
    for label, count in report.clips_per_class().items():
        print(f"  {label}: {count}")

    print("\nPredicted-class distribution:")
    for label, count in report.predicted_class_distribution().items():
        print(f"  {label}: {count}")

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
        print(f"  p95_inference_s:  {timing['p95_inference_s']:.6f}")
        print(f"  total_inference_s:{timing['total_inference_s']:.6f}")
        print(f"  mean_rtf:         {timing['mean_rtf']:.6f}")

    print("\nSystematic misclassifications (excluding unknown):")
    mis = report.systematic_misclassifications()
    if not mis:
        print("  (none)")
    else:
        for item in mis:
            print(
                f"  {item['actual']} -> {item['predicted']}: "
                f"{item['count']}"
            )

    print("\nQualitative examples (actual | predicted | confidence | top-2 margin):")
    print(
        f"{'actual':20}{'predicted':20}{'confidence':>12}{'margin':>12}  clip_id"
    )
    for example in report.qualitative_examples():
        print(
            f"{str(example['actual']):20}"
            f"{str(example['predicted']):20}"
            f"{float(example['confidence']):12.4f}"
            f"{float(example['top2_margin']):12.4f}  "
            f"{example['clip_id']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Noise Classifier v1 on real SIH-26 "
            "uav_drone / vehicle_engine / impulsive_firearms recordings."
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
        help="Local directory of SIH-26 ZIP archives (optional).",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Optional deterministic cap per class (sorted by sample_id).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for JSON report output.",
    )
    args = parser.parse_args()

    metadata_path = _resolve_metadata_path(args.metadata_path)
    if not metadata_path.exists():
        raise SystemExit(
            f"Missing metadata.csv — cannot evaluate real corpus.\n"
            f"Expected at: {metadata_path}"
        )

    repo_id = None if args.archive_dir is not None else SIH26_REPO_ID
    missing = find_missing_corpus_archives(
        metadata_path,
        archive_dir=args.archive_dir,
        repo_id=repo_id,
        categories=DEVELOPMENT_NOISE_CATEGORIES,
    )

    if missing:
        raise SystemExit(
            "Cannot evaluate real SIH-26 corpus — required archives missing:\n"
            + "\n".join(f"  - {name}" for name in missing)
            + "\nProvide --archive-dir with those ZIPs or download them into "
            "the Hugging Face cache for "
            f"{SIH26_REPO_ID}."
        )

    clips = build_defence_noise_corpus(
        metadata_path,
        categories=DEVELOPMENT_NOISE_CATEGORIES,
        max_per_class=args.max_per_class,
    )

    if not clips:
        raise SystemExit(
            "No defence-noise clips found in metadata for categories: "
            + ", ".join(DEVELOPMENT_NOISE_CATEGORIES)
        )

    dataset = ZipManifestDataset(
        metadata_path=metadata_path,
        repo_id=repo_id,
        archive_dir=args.archive_dir,
    )

    print(
        f"Evaluating {len(clips)} real clips "
        f"({CORPUS_EVAL_RULES_VERSION} / {CORPUS_EVAL_SPLIT_NAME})..."
    )

    report = run_noise_corpus_evaluation(clips, dataset)
    _print_report(report)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "noise_classifier_v1_real_corpus_report.json"
    payload = {
        "classifier": "noise_classifier_v1",
        "evaluation": "real_sih26_defence_noise_corpus",
        "rules_version": report.rules_version,
        "split_name": report.split_name,
        "num_clips": len(report.case_results),
        "clips_per_class": report.clips_per_class(),
        "predicted_class_distribution": report.predicted_class_distribution(),
        "overall_accuracy": report.overall_accuracy(),
        "macro_f1": report.macro_f1(),
        "unknown_rate": report.unknown_rate(),
        "confusion_matrix": report.confusion_matrix(),
        "per_class_metrics": report.per_class_metrics(),
        "timing": report.timing_summary(),
        "systematic_misclassifications": report.systematic_misclassifications(),
        "qualitative_examples": report.qualitative_examples(),
        "case_results": [
            {
                "clip_id": result.clip_id,
                "noise_source_id": result.noise_source_id,
                "ground_truth": result.ground_truth,
                "predicted_class": result.predicted_class,
                "probabilities": result.probabilities,
                "confidence": result.confidence,
                "top2_margin": result.top2_margin,
                "inference_s": result.inference_s,
                "num_samples": result.num_samples,
                "sample_rate": result.sample_rate,
            }
            for result in report.case_results
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote report: {output_path}")


if __name__ == "__main__":
    main()
