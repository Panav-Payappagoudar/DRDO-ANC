"""CLI for the offline noise-aware enhancement experiment."""

from __future__ import annotations

import argparse
import tempfile
import urllib.request
from pathlib import Path

from drdo_anc.benchmark import (
    MixtureGenerator,
    build_development_manifest,
)
from drdo_anc.benchmark.manifest_benchmark import (
    build_manifest_dataset,
    select_smoke_cases,
    validate_development_manifest,
)
from drdo_anc.classification import SupervisedNoiseClassifier
from drdo_anc.dataset.manifest import (
    SIH26_METADATA_FILENAME,
    SIH26_REPO_ID,
)
from drdo_anc.experiments.noise_aware import run_noise_aware_experiment
from drdo_anc.experiments.noise_aware.runner import load_offline_df3_session
from drdo_anc.experiments.noise_aware.strategies import DEFAULT_STRATEGY_MAP


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "classifier_results" / "noise_aware_enhancement_v1"
)
DEFAULT_CLASSIFIER_DIR = (
    PROJECT_ROOT
    / "data"
    / "classifier_results"
    / "noise_classifier_v2"
    / "selected_model"
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


def _print_summary(report) -> None:
    print("\n" + "=" * 70)
    print("NOISE-AWARE ENHANCEMENT EXPERIMENT")
    print("=" * 70)
    print(f"Version:    {report.experiment_version}")
    print(f"Manifest:   {report.manifest_rules_version}")
    print(f"Cases:      {len(report.case_results)}")
    print(f"Classifier: {report.classifier_model_name}")
    print(f"Strategies: {report.strategy_map}")

    clf = report.classification_summary()
    print(
        f"\nClassifier accuracy on benchmark mixtures: "
        f"{clf['accuracy']:.4f}"
    )

    for system_id in (
        "noisy",
        "classical_spectral_subtraction",
        "df3_baseline",
        "adaptive",
    ):
        summary = report.summary_for_system(system_id)
        print(f"\n[{system_id}]")
        if not summary:
            print("  (no successful cases)")
            continue
        for key in (
            "mean_si_sdr",
            "mean_stoi",
            "mean_pesq",
            "mean_snr",
            "median_rtf",
            "mean_rtf",
        ):
            if key in summary:
                print(f"  {key}: {summary[key]:.4f}")

    print("\nBy noise category (SI-SDR mean):")
    for system_id in ("df3_baseline", "adaptive"):
        by_cat = report.summary_by_noise_category(system_id)
        parts = [
            f"{cat}={vals.get('mean_si_sdr', float('nan')):.3f}"
            for cat, vals in by_cat.items()
        ]
        print(f"  {system_id}: " + ", ".join(parts))

    print("\nBy SNR (SI-SDR mean):")
    for system_id in ("df3_baseline", "adaptive"):
        by_snr = report.summary_by_snr(system_id)
        parts = [
            f"{snr}={vals.get('mean_si_sdr', float('nan')):.3f}"
            for snr, vals in by_snr.items()
        ]
        print(f"  {system_id}: " + ", ".join(parts))

    print("\nDF3 baseline vs adaptive:")
    for metric in ("si_sdr", "stoi", "pesq", "snr"):
        cmp_ = report.compare_systems("df3_baseline", "adaptive", metric)
        print(
            f"  {metric}: mean_delta={cmp_['mean_difference']:+.4f}, "
            f"median_delta={cmp_['median_difference']:+.4f}, "
            f"improved={cmp_['improved']}, degraded={cmp_['degraded']}, "
            f"tied={cmp_['tied']}"
        )
        for cat, delta in cmp_["per_category_mean_difference"].items():
            print(f"    {cat}: {delta:+.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Offline noise-aware enhancement experiment "
            "(classifier v2 + DF3 atten_lim_db strategies)."
        ),
    )
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument("--archive-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--classifier-dir",
        type=Path,
        default=DEFAULT_CLASSIFIER_DIR,
        help="Directory containing v2 model.joblib + model_meta.json",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run only the approved 2-case smoke subset.",
    )
    args = parser.parse_args()

    if not args.classifier_dir.exists():
        raise SystemExit(
            f"Classifier model directory not found: {args.classifier_dir}\n"
            "Train v2 first: python scripts/train_noise_classifier_v2.py --seed 42"
        )

    metadata_path = _resolve_metadata_path(args.metadata_path)
    manifest = build_development_manifest(metadata_path)
    validate_development_manifest(manifest)

    cases = select_smoke_cases(manifest) if args.smoke else None

    dataset = build_manifest_dataset(
        manifest,
        metadata_path,
        archive_dir=args.archive_dir,
    )
    mixture_generator = MixtureGenerator(dataset)

    print("Loading Noise Classifier v2...")
    classifier = SupervisedNoiseClassifier.load(args.classifier_dir)

    print("Loading DeepFilterNet3 offline session (df.enhance + atten_lim_db)...")
    df3_session = load_offline_df3_session()
    print(f"DF3 model: {df3_session.name} @ {df3_session.sample_rate_hz} Hz")

    print("\nStrategy map:")
    for label, strategy in DEFAULT_STRATEGY_MAP.items():
        print(
            f"  {label} -> {strategy.strategy_id} "
            f"(atten_lim_db={strategy.atten_lim_db})"
        )

    report = run_noise_aware_experiment(
        manifest,
        mixture_generator,
        classifier,
        df3_session=df3_session,
        cases=cases,
    )

    _print_summary(report)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "noise_aware_enhancement_v1_report.json"
    report.save_json(output_path)
    print(f"\nWrote report: {output_path}")


if __name__ == "__main__":
    main()
