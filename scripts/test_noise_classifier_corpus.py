"""Tests for Noise Classifier real-corpus evaluation helpers."""

from __future__ import annotations

import os

from drdo_anc.classification.evaluation import (
    CORPUS_EVAL_RULES_VERSION,
    CorpusCaseResult,
    CorpusEvaluationReport,
    build_defence_noise_corpus,
    find_missing_corpus_archives,
    run_noise_corpus_evaluation,
)
from drdo_anc.dataset.manifest import SIH26_REPO_ID
from drdo_anc.dataset.source_pool import DEVELOPMENT_NOISE_CATEGORIES
from drdo_anc.dataset.zip_manifest_dataset import ZipManifestDataset

from build_evaluation_fixtures import (
    FIXTURE_DIR,
    METADATA_PATH,
    build_fixtures,
)


def setup_module() -> None:
    build_fixtures()


def test_build_defence_noise_corpus_deterministic() -> None:
    first = build_defence_noise_corpus(METADATA_PATH)
    second = build_defence_noise_corpus(METADATA_PATH)

    assert first == second
    assert len(first) > 0

    counts = {
        category: sum(1 for clip in first if clip.ground_truth == category)
        for category in DEVELOPMENT_NOISE_CATEGORIES
    }
    assert counts["uav_drone"] == 2
    assert counts["impulsive_firearms"] == 2
    assert counts["vehicle_engine"] == 2


def test_build_defence_noise_corpus_max_per_class() -> None:
    clips = build_defence_noise_corpus(METADATA_PATH, max_per_class=1)

    assert len(clips) == 3
    assert {clip.ground_truth for clip in clips} == set(
        DEVELOPMENT_NOISE_CATEGORIES
    )


def test_find_missing_archives_with_local_fixture() -> None:
    missing = find_missing_corpus_archives(
        METADATA_PATH,
        archive_dir=FIXTURE_DIR,
        repo_id=None,
    )
    assert missing == []


def test_find_missing_archives_reports_absent_zips() -> None:
    missing = find_missing_corpus_archives(
        METADATA_PATH,
        archive_dir=FIXTURE_DIR / "does_not_exist",
        repo_id=None,
    )
    assert "test_archive.zip" in missing


def test_unknown_not_counted_as_correct() -> None:
    report = CorpusEvaluationReport(
        rules_version=CORPUS_EVAL_RULES_VERSION,
        split_name="unit",
        case_results=[
            CorpusCaseResult(
                clip_id="a",
                noise_source_id="a",
                ground_truth="uav_drone",
                predicted_class="unknown",
                probabilities={"unknown": 1.0},
                confidence=1.0,
                top2_margin=1.0,
                inference_s=0.01,
                num_samples=16000,
                sample_rate=16000,
            ),
            CorpusCaseResult(
                clip_id="b",
                noise_source_id="b",
                ground_truth="vehicle_engine",
                predicted_class="vehicle_engine",
                probabilities={"vehicle_engine": 1.0},
                confidence=1.0,
                top2_margin=1.0,
                inference_s=0.02,
                num_samples=16000,
                sample_rate=16000,
            ),
        ],
    )

    assert report.overall_accuracy() == 0.5
    assert report.unknown_rate() == 0.5
    assert report.macro_f1() >= 0.0


def test_fixture_corpus_evaluation_runs() -> None:
    clips = build_defence_noise_corpus(METADATA_PATH, max_per_class=1)
    dataset = ZipManifestDataset(
        metadata_path=METADATA_PATH,
        repo_id=None,
        archive_dir=FIXTURE_DIR,
    )

    report = run_noise_corpus_evaluation(clips, dataset)

    assert len(report.case_results) == 3
    assert report.clips_per_class()["uav_drone"] == 1
    assert sum(report.predicted_class_distribution().values()) == 3
    assert 0.0 <= report.overall_accuracy() <= 1.0
    assert 0.0 <= report.unknown_rate() <= 1.0
    timing = report.timing_summary()
    assert "mean_inference_s" in timing
    assert "p95_inference_s" in timing
    assert "mean_rtf" in timing

    examples = report.qualitative_examples(limit=3)
    assert len(examples) == 3
    assert {"actual", "predicted", "confidence", "top2_margin"} <= set(
        examples[0]
    )


def test_integration_real_corpus_smoke() -> None:
    if os.environ.get("SIH26_INTEGRATION") != "1":
        return

    from huggingface_hub import hf_hub_download
    from pathlib import Path
    from drdo_anc.dataset.manifest import (
        SIH26_METADATA_FILENAME,
        SIH26_REPO_ID,
    )

    metadata_path = Path(
        hf_hub_download(
            repo_id=SIH26_REPO_ID,
            repo_type="dataset",
            filename=SIH26_METADATA_FILENAME,
        )
    )

    missing = find_missing_corpus_archives(
        metadata_path,
        archive_dir=None,
        repo_id=SIH26_REPO_ID,
    )
    assert missing == [], f"Missing archives: {missing}"

    clips = build_defence_noise_corpus(metadata_path, max_per_class=2)
    dataset = ZipManifestDataset(
        metadata_path=metadata_path,
        repo_id=SIH26_REPO_ID,
    )
    report = run_noise_corpus_evaluation(clips, dataset)

    assert len(report.case_results) == 6
    assert report.clips_per_class() == {
        "uav_drone": 2,
        "vehicle_engine": 2,
        "impulsive_firearms": 2,
    }


def main() -> None:
    print("=" * 70)
    print("DRDO-ANC | Noise Classifier Corpus Evaluation Tests")
    print("=" * 70)

    build_fixtures()

    tests = [
        test_build_defence_noise_corpus_deterministic,
        test_build_defence_noise_corpus_max_per_class,
        test_find_missing_archives_with_local_fixture,
        test_find_missing_archives_reports_absent_zips,
        test_unknown_not_counted_as_correct,
        test_fixture_corpus_evaluation_runs,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    if os.environ.get("SIH26_INTEGRATION") == "1":
        test_integration_real_corpus_smoke()
        print("PASS: test_integration_real_corpus_smoke")
    else:
        print("SKIP: test_integration_real_corpus_smoke")

    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
