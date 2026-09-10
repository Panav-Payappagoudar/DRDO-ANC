"""Tests for recording-safe fine-tuned DF3 evaluation construction."""

from __future__ import annotations

import tempfile
from pathlib import Path

from drdo_anc.benchmark import build_development_manifest
from drdo_anc.dataset.source_pool import RULES_VERSION
from drdo_anc.experiments.finetuned_compare.heldout import (
    build_recording_safe_manifest,
    inspect_training_provenance,
)

from build_evaluation_fixtures import METADATA_PATH, build_fixtures


def test_training_provenance_is_unverified_without_file_list() -> None:
    provenance = inspect_training_provenance()
    assert provenance.holdout_claim is False
    assert provenance.holdout_status == "unverified"
    assert provenance.training_manifest_path is None
    assert provenance.reasons


def test_recording_safe_manifest_rejects_tiny_fixture() -> None:
    build_fixtures()
    development = build_development_manifest(METADATA_PATH)
    assert development.rules_version == RULES_VERSION
    try:
        build_recording_safe_manifest(METADATA_PATH, development)
    except ValueError as exc:
        message = str(exc).lower()
        assert "disjoint" in message or "speakers" in message
        return
    raise AssertionError("Expected fixture metadata to lack extra speakers.")


def test_development_protocol_unchanged() -> None:
    build_fixtures()
    first = build_development_manifest(METADATA_PATH)
    second = build_development_manifest(METADATA_PATH)
    assert first.rules_version == "sih26-eval-v1"
    assert [case.case_id for case in first.cases] == [
        case.case_id for case in second.cases
    ]
    assert [case.clean_source.sample_id for case in first.cases] == [
        case.clean_source.sample_id for case in second.cases
    ]


def test_explicit_training_manifest_enables_holdout_claim(
    directory: Path,
) -> None:
    manifest_path = directory / "training_sources.json"
    manifest_path.write_text(
        '{"clean_sample_ids": [], "noise_sample_ids": [], '
        '"noise_recording_ids": [], "speaker_ids": []}',
        encoding="utf-8",
    )
    provenance = inspect_training_provenance(
        training_manifest_path=manifest_path,
    )
    assert provenance.holdout_claim is True
    assert provenance.holdout_status == "excluded_listed_training_sources"


def main() -> None:
    print("=" * 70)
    print("DRDO-ANC | Recording-safe fine-tuned DF3 tests")
    print("=" * 70)

    test_training_provenance_is_unverified_without_file_list()
    print("PASS: test_training_provenance_is_unverified_without_file_list")

    test_recording_safe_manifest_rejects_tiny_fixture()
    print("PASS: test_recording_safe_manifest_rejects_tiny_fixture")

    test_development_protocol_unchanged()
    print("PASS: test_development_protocol_unchanged")

    with tempfile.TemporaryDirectory() as tmp_dir:
        test_explicit_training_manifest_enables_holdout_claim(Path(tmp_dir))
        print("PASS: test_explicit_training_manifest_enables_holdout_claim")

    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
