"""Recording-safe SIH-26 eval of pretrained vs fine-tuned DeepFilterNet3.

Does not modify either enhancer or the 60-case development benchmark.
Does not claim training hold-out unless an explicit train-source list exists.
"""

from __future__ import annotations

import argparse
import tempfile
import urllib.request
from pathlib import Path

from drdo_anc.benchmark import (
    BenchmarkMode,
    MixtureGenerator,
    build_development_manifest,
)
from drdo_anc.benchmark.manifest_benchmark import (
    ManifestBenchmarkRunner,
    build_manifest_dataset,
    select_smoke_cases,
)
from drdo_anc.dataset.manifest import (
    SIH26_METADATA_FILENAME,
    SIH26_REPO_ID,
)
from drdo_anc.enhancement import create_enhancer, get_model_config
from drdo_anc.enhancement.finetuned import FINETUNED_MODEL_NAME
from drdo_anc.experiments.finetuned_compare import (
    compare_benchmark_reports,
    save_comparison_report,
)
from drdo_anc.experiments.finetuned_compare.heldout import (
    EXPERIMENT_VERSION,
    assert_recording_disjoint,
    build_recording_safe_manifest,
    default_output_dir,
    development_blocklist,
    inspect_training_provenance,
    timing_summary,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
    report.save_csv(output_path.with_suffix(".csv"))
    print(f"Wrote {output_path}")
    return report.to_dict()


def _print_report(comparison: dict) -> None:
    provenance = comparison["provenance"]
    print("\n" + "=" * 70)
    print("RECORDING-SAFE SIH-26 EVAL")
    print("=" * 70)
    print(f"Version:     {comparison['experiment_version']}")
    print(f"Holdout:     {provenance['training_holdout_status']}")
    print(f"Claim:       {provenance['training_holdout_claim']}")
    print(f"Paired rows: {comparison['num_paired_rows']}")
    print("Reasons:")
    for reason in provenance["reasons"]:
        print(f"  - {reason}")

    for metric, summary in comparison["metrics"].items():
        print(f"\n{metric}:")
        print(f"  mean delta:   {summary['mean_difference']:+.4f}")
        print(f"  median delta: {summary['median_difference']:+.4f}")
        print(
            f"  improved/degraded/tied: "
            f"{summary['improved']}/{summary['degraded']}/{summary['tied']}"
        )

    print("\nTiming:")
    for name, timing in comparison["timing"].items():
        print(
            f"  {name}: median RTF {timing.get('median_rtf', 0.0):.3f}x "
            f"mean inference {timing.get('mean_inference_s', 0.0):.4f}s"
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

    if comparison.get("by_snr"):
        print("\nBy SNR (SI-SDR delta):")
        for snr_key, metrics in comparison["by_snr"].items():
            si_sdr = metrics.get("si_sdr", {})
            print(
                f"  {snr_key}: mean delta "
                f"{si_sdr.get('mean_difference', 0.0):+.4f}"
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
            "Compare DeepFilterNet3 vs DeepFilterNet3-Finetuned on SIH-26 "
            "recordings that are recording-disjoint from the 60-case "
            "development benchmark. Does not claim training hold-out unless "
            "a training-source list is supplied."
        )
    )
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument("--archive-dir", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--training-manifest",
        type=Path,
        default=None,
        help="Optional JSON listing training sample/recording/speaker IDs.",
    )
    parser.add_argument(
        "--mode",
        choices=["offline", "streaming", "both"],
        default="both",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir or default_output_dir()
    print("=" * 70)
    print("DRDO-ANC | Recording-safe fine-tuned DF3 evaluation")
    print("=" * 70)

    provenance = inspect_training_provenance(
        training_manifest_path=args.training_manifest,
    )
    print(f"Training hold-out status: {provenance.holdout_status}")
    print(f"Training hold-out claim:  {provenance.holdout_claim}")
    for reason in provenance.reasons:
        print(f"  - {reason}")

    metadata_path = _resolve_metadata_path(args.metadata_path)
    development = build_development_manifest(metadata_path)
    manifest = build_recording_safe_manifest(
        metadata_path,
        development,
        provenance=provenance,
    )
    assert_recording_disjoint(development, manifest)
    blocked = development_blocklist(development)
    print(
        f"\nBuilt {len(manifest.cases)} cases "
        f"({manifest.rules_version}); recording-disjoint from sih26-eval-v1."
    )
    print(
        f"Blocked from 60-case set: {len(blocked.speaker_ids)} speakers, "
        f"{len(blocked.noise_recording_ids)} noise recordings."
    )

    cases = select_smoke_cases(manifest) if args.smoke else manifest.cases
    label = "smoke" if args.smoke else "full"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.save_json(output_dir / f"manifest_{label}.json")
    dataset = build_manifest_dataset(
        manifest,
        metadata_path,
        archive_dir=args.archive_dir,
    )
    mixture_generator = MixtureGenerator(dataset)
    modes = _parse_modes(args.mode)

    pretrained_path = output_dir / f"pretrained_{label}.json"
    finetuned_path = output_dir / f"finetuned_{label}.json"
    pretrained_report = _run_model(
        PRETRAINED_MODEL_NAME,
        manifest,
        mixture_generator,
        cases,
        modes,
        pretrained_path,
    )
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
    comparison["experiment_version"] = EXPERIMENT_VERSION
    comparison["provenance"] = provenance.to_dict()
    comparison["disjoint_from_development_benchmark"] = True
    comparison["development_rules_version"] = development.rules_version
    comparison["evaluation_rules_version"] = manifest.rules_version
    comparison["num_cases"] = len(cases)
    comparison["failed_rows"] = {
        PRETRAINED_MODEL_NAME: pretrained_report.get("failed_cases", 0),
        FINETUNED_MODEL_NAME: finetuned_report.get("failed_cases", 0),
    }
    comparison["development_blocklist"] = {
        "num_speakers": len(blocked.speaker_ids),
        "num_clean_sample_ids": len(blocked.clean_sample_ids),
        "num_noise_sample_ids": len(blocked.noise_sample_ids),
        "num_noise_recording_ids": len(blocked.noise_recording_ids),
    }
    comparison["absolute"] = {
        PRETRAINED_MODEL_NAME: pretrained_report.get("summary_overall", {}),
        FINETUNED_MODEL_NAME: finetuned_report.get("summary_overall", {}),
    }
    comparison["absolute_by_snr"] = {
        PRETRAINED_MODEL_NAME: pretrained_report.get("summary_by_snr", {}),
        FINETUNED_MODEL_NAME: finetuned_report.get("summary_by_snr", {}),
    }
    comparison["absolute_by_noise_category"] = {
        PRETRAINED_MODEL_NAME: pretrained_report.get(
            "summary_by_noise_category", {}
        ),
        FINETUNED_MODEL_NAME: finetuned_report.get(
            "summary_by_noise_category", {}
        ),
    }
    comparison["timing"] = {
        PRETRAINED_MODEL_NAME: timing_summary(pretrained_report),
        FINETUNED_MODEL_NAME: timing_summary(finetuned_report),
    }
    comparison_path = output_dir / f"comparison_{label}.json"
    save_comparison_report(comparison, comparison_path)
    print(f"\nWrote {comparison_path}")
    _print_report(comparison)


if __name__ == "__main__":
    main()
