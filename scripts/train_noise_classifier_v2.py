"""Train and evaluate Noise Classifier v2 on the real SIH-26 corpus."""

from __future__ import annotations

import argparse
import json
import tempfile
import urllib.request
from pathlib import Path

from drdo_anc.classification.evaluation import find_missing_corpus_archives
from drdo_anc.classification.training import (
    DEFAULT_SEED,
    TRAINING_RULES_VERSION,
    extract_labeled_features,
    load_feature_cache,
    save_feature_cache,
    stratified_recording_split,
    train_and_select_models,
)
from drdo_anc.dataset.manifest import (
    SIH26_METADATA_FILENAME,
    SIH26_REPO_ID,
)
from drdo_anc.dataset.source_pool import DEVELOPMENT_NOISE_CATEGORIES
from drdo_anc.dataset.zip_manifest_dataset import ZipManifestDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "classifier_results" / "noise_classifier_v2"
)


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


def _print_metrics(title: str, metrics: dict) -> None:
    print(f"\n{title}")
    print(
        f"  accuracy={metrics['accuracy']:.4f}  "
        f"macro_P={metrics['macro_precision']:.4f}  "
        f"macro_R={metrics['macro_recall']:.4f}  "
        f"macro_F1={metrics['macro_f1']:.4f}"
    )
    for label, values in metrics["per_class"].items():
        print(
            f"  {label}: P={values['precision']:.4f} "
            f"R={values['recall']:.4f} F1={values['f1']:.4f} "
            f"n={values['support']}"
        )
    if "mean_inference_s" in metrics:
        print(
            f"  mean_inference_s={metrics['mean_inference_s']:.6f}  "
            f"p95={metrics.get('p95_inference_s', 0.0):.6f}  "
            f"mean_rtf={metrics.get('mean_rtf', 0.0):.6f}"
        )
    if "unknown_rate" in metrics:
        print(f"  unknown_rate={metrics['unknown_rate']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train Noise Classifier v2 candidates on real SIH-26 "
            "defence-noise recordings and select by validation macro F1."
        ),
    )
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument("--archive-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=None,
        help="Optional NPZ feature cache path (read/write).",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Optional deterministic per-class clip cap for smoke runs.",
    )
    args = parser.parse_args()

    metadata_path = _resolve_metadata_path(args.metadata_path)
    if not metadata_path.exists():
        raise SystemExit(f"Missing metadata.csv: {metadata_path}")

    repo_id = None if args.archive_dir is not None else SIH26_REPO_ID
    missing = find_missing_corpus_archives(
        metadata_path,
        archive_dir=args.archive_dir,
        repo_id=repo_id,
        categories=DEVELOPMENT_NOISE_CATEGORIES,
    )
    if missing:
        raise SystemExit(
            "Cannot train v2 — required SIH-26 archives missing:\n"
            + "\n".join(f"  - {name}" for name in missing)
        )

    from drdo_anc.classification.evaluation import build_defence_noise_corpus

    clips = build_defence_noise_corpus(
        metadata_path,
        categories=DEVELOPMENT_NOISE_CATEGORIES,
        max_per_class=args.max_per_class,
    )
    dataset = ZipManifestDataset(
        metadata_path=metadata_path,
        repo_id=repo_id,
        archive_dir=args.archive_dir,
    )

    cache_path = args.feature_cache
    if cache_path is None:
        cache_path = args.output_dir / "features_cache.npz"

    if cache_path.exists() and args.max_per_class is None:
        print(f"Loading feature cache: {cache_path}")
        rows = load_feature_cache(cache_path)
    else:
        print(f"Extracting features for {len(clips)} clips...")
        rows = extract_labeled_features(clips, dataset)
        save_feature_cache(cache_path, rows)
        print(f"Wrote feature cache: {cache_path}")

    split = stratified_recording_split(rows, seed=args.seed)
    print("\nClass distribution:")
    for split_name, counts in split.class_distribution().items():
        print(f"  {split_name}: {counts}")

    print(
        f"\nTraining candidates ({TRAINING_RULES_VERSION}, seed={args.seed})..."
    )
    classifier, report, selected = train_and_select_models(
        split,
        seed=args.seed,
        output_dir=args.output_dir,
        dataset=dataset,
        clips=clips,
    )

    print("\n" + "=" * 70)
    print("NOISE CLASSIFIER V2 TRAINING REPORT")
    print("=" * 70)
    print(f"Selected model: {report.selected_model}")
    print(f"Selection:      {report.selection_criterion}")
    print(f"Saved to:       {args.output_dir / 'selected_model'}")

    for candidate in report.candidates:
        print(f"\nCandidate: {candidate['model_name']}")
        print(f"  size_bytes={candidate['model_size_bytes']}")
        _print_metrics("  validation", candidate["validation_metrics"])
        _print_metrics("  test", candidate["test_metrics"])

    _print_metrics("v1 comparison on same test set", report.v1_comparison)

    report_path = args.output_dir / "training_report.json"
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote report: {report_path}")
    print(
        f"Selected test macro F1={selected.test_metrics['macro_f1']:.4f} "
        f"(v1 macro F1={report.v1_comparison['macro_f1']:.4f})"
    )
    _ = classifier  # loaded/saved selected model


if __name__ == "__main__":
    main()
