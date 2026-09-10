"""CLI for pretrained vs fine-tuned DeepFilterNet3 head-to-head evaluation."""

from __future__ import annotations

import argparse
import tempfile
import urllib.request
from pathlib import Path

from drdo_anc.benchmark import (
    BenchmarkMode,
    EvaluationManifest,
    MixtureGenerator,
    build_development_manifest,
)
from drdo_anc.benchmark.manifest_benchmark import (
    ManifestBenchmarkRunner,
    build_manifest_dataset,
    select_smoke_cases,
    validate_development_manifest,
)
from drdo_anc.dataset.manifest import (
    SIH26_METADATA_FILENAME,
    SIH26_REPO_ID,
)
from drdo_anc.enhancement import create_enhancer, get_model_config
from drdo_anc.enhancement.finetuned import FINETUNED_MODEL_NAME
from drdo_anc.experiments.finetuned_compare import (
    compare_benchmark_reports,
    load_benchmark_report,
    save_comparison_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "benchmark_results" / "dfn3_finetuned_compare"
)
PRETRAINED_MODEL_NAME = "DeepFilterNet3"


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


def _parse_modes(raw: str) -> tuple[BenchmarkMode, ...]:
    if raw == "both":
        return (BenchmarkMode.OFFLINE, BenchmarkMode.STREAMING)
    return (BenchmarkMode(raw),)


def _run_model(
    model_name: str,
    manifest,
    mixture_generator,
    cases,
    modes: tuple[BenchmarkMode, ...],
    output_path: Path,
):
    model_config = get_model_config(model_name)
    print(f"\nLoading {model_name}...")
    enhancer = create_enhancer(model_name)
    runner = ManifestBenchmarkRunner(
        enhancer,
        mixture_generator,
        streaming_delay_samples=model_config.streaming_delay_samples,
        modes=modes,
    )
    report = runner.run(manifest, cases)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_json(output_path)
    csv_path = output_path.with_suffix(".csv")
    report.save_csv(csv_path)
    print(f"Wrote {output_path}")
    print(f"Wrote {csv_path}")
    return report.to_dict()


def _print_comparison(comparison: dict) -> None:
    print("\n" + "=" * 70)
    print("PRETRAINED vs FINE-TUNED DFN3")
    print("=" * 70)
    print(f"Version:     {comparison['experiment_version']}")
    print(f"Baseline:    {comparison['baseline_model']}")
    print(f"Candidate:   {comparison['candidate_model']}")
    print(f"Paired rows: {comparison['num_paired_rows']}")
    print(f"Decision:    {comparison['recommendation']}")

    for metric, summary in comparison["metrics"].items():
        print(f"\n{metric}:")
        print(f"  mean delta:   {summary['mean_difference']:+.4f}")
        print(f"  median delta: {summary['median_difference']:+.4f}")
        print(
            f"  improved/degraded/tied: "
            f"{summary['improved']}/{summary['degraded']}/{summary['tied']}"
        )

    if comparison.get("by_mode"):
        print("\nBy mode (SI-SDR delta):")
        for mode, metrics in comparison["by_mode"].items():
            si_sdr = metrics.get("si_sdr", {})
            print(
                f"  {mode}: mean delta {si_sdr.get('mean_difference', 0.0):+.4f} "
                f"({si_sdr.get('improved', 0)} improved / "
                f"{si_sdr.get('degraded', 0)} degraded)"
            )

    if comparison.get("by_noise_category"):
        print("\nBy noise category (SI-SDR delta):")
        for category, metrics in comparison["by_noise_category"].items():
            si_sdr = metrics.get("si_sdr", {})
            print(
                f"  {category}: mean delta "
                f"{si_sdr.get('mean_difference', 0.0):+.4f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare registered DeepFilterNet3 against the fine-tuned "
            "epoch-130 artifact on the SIH-26 development manifest. "
            "Does not modify the live pipeline or pretrained enhancer."
        ),
    )
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--archive-dir", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--mode",
        choices=["offline", "streaming", "both"],
        default="both",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run only the 2-case smoke subset.",
    )
    parser.add_argument(
        "--pretrained-report",
        type=Path,
        default=None,
        help="Reuse an existing pretrained ManifestBenchmark JSON.",
    )
    parser.add_argument(
        "--finetuned-report",
        type=Path,
        default=None,
        help="Reuse an existing fine-tuned ManifestBenchmark JSON.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("DRDO-ANC | Fine-tuned DFN3 Head-to-Head")
    print("=" * 70)

    label = "smoke" if args.smoke else "full"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pretrained_path = (
        args.pretrained_report
        or args.output_dir / f"pretrained_{label}.json"
    )
    finetuned_path = (
        args.finetuned_report
        or args.output_dir / f"finetuned_{label}.json"
    )

    pretrained_report = None
    finetuned_report = None

    if args.pretrained_report is not None:
        pretrained_report = load_benchmark_report(args.pretrained_report)
        print(f"Loaded pretrained report: {args.pretrained_report}")
    if args.finetuned_report is not None:
        finetuned_report = load_benchmark_report(args.finetuned_report)
        print(f"Loaded fine-tuned report: {args.finetuned_report}")

    if pretrained_report is None or finetuned_report is None:
        metadata_path = _resolve_metadata_path(args.metadata_path)
        if args.manifest_path is not None:
            manifest = EvaluationManifest.load_json(args.manifest_path)
        else:
            manifest = build_development_manifest(metadata_path)
        validate_development_manifest(manifest)
        cases = select_smoke_cases(manifest) if args.smoke else manifest.cases
        dataset = build_manifest_dataset(
            manifest,
            metadata_path,
            archive_dir=args.archive_dir,
        )
        mixture_generator = MixtureGenerator(dataset)
        modes = _parse_modes(args.mode)

        if pretrained_report is None:
            pretrained_report = _run_model(
                PRETRAINED_MODEL_NAME,
                manifest,
                mixture_generator,
                cases,
                modes,
                pretrained_path,
            )
        if finetuned_report is None:
            finetuned_report = _run_model(
                FINETUNED_MODEL_NAME,
                manifest,
                mixture_generator,
                cases,
                modes,
                finetuned_path,
            )

    comparison = compare_benchmark_reports(
        pretrained_report,
        finetuned_report,
        baseline_name=PRETRAINED_MODEL_NAME,
        candidate_name=FINETUNED_MODEL_NAME,
    )
    comparison_path = args.output_dir / f"comparison_{label}.json"
    save_comparison_report(comparison, comparison_path)
    print(f"\nWrote {comparison_path}")
    _print_comparison(comparison)


if __name__ == "__main__":
    main()
