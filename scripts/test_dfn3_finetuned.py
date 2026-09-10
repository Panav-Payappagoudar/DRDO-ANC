"""Tests for fine-tuned DeepFilterNet3 integration and comparison."""

from __future__ import annotations

import json
import os
import tarfile
import tempfile
from pathlib import Path

from drdo_anc.enhancement import (
    FINETUNED_MODEL_NAME,
    FineTunedDeepFilterNetEnhancer,
    create_enhancer,
    get_model_config,
    list_models,
)
from drdo_anc.enhancement.finetuned import (
    REQUIRED_ONNX_MEMBERS,
    ensure_finetuned_onnx_bundle,
    finetuned_export_dir,
    finetuned_onnx_dir,
    resolve_finetuned_artifact_root,
)
from drdo_anc.experiments.finetuned_compare import compare_benchmark_reports


def test_registry_lists_pretrained_and_finetuned() -> None:
    models = list_models()
    assert "DeepFilterNet3" in models
    assert FINETUNED_MODEL_NAME in models

    pretrained = get_model_config("DeepFilterNet3")
    finetuned = get_model_config(FINETUNED_MODEL_NAME)
    assert pretrained.streaming_delay_samples == 1440
    assert finetuned.streaming_delay_samples == 1440
    assert pretrained.name != finetuned.name

    enhancer = create_enhancer(FINETUNED_MODEL_NAME, load=False)
    assert isinstance(enhancer, FineTunedDeepFilterNetEnhancer)
    assert enhancer.name() == FINETUNED_MODEL_NAME


def test_pretrained_factory_unchanged() -> None:
    from drdo_anc.enhancement.deepfilternet import DeepFilterNetEnhancer

    enhancer = create_enhancer("DeepFilterNet3", load=False)
    assert isinstance(enhancer, DeepFilterNetEnhancer)
    assert enhancer.name() == "DeepFilterNet3"


def test_artifact_paths_exist() -> None:
    root = resolve_finetuned_artifact_root()
    export_dir = finetuned_export_dir(root)
    onnx_dir = finetuned_onnx_dir(root)

    assert (export_dir / "config.ini").exists()
    assert (export_dir / "checkpoints" / "model_130.ckpt").exists()
    for name in REQUIRED_ONNX_MEMBERS:
        assert (onnx_dir / name).exists()


def test_onnx_bundle_uses_native_layout() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        source = Path(tmp_dir) / "onnx"
        dest = Path(tmp_dir) / "bundle.tar.gz"
        source.mkdir()
        for name in REQUIRED_ONNX_MEMBERS:
            (source / name).write_bytes(b"placeholder")

        bundle = ensure_finetuned_onnx_bundle(onnx_dir=source, dest=dest)
        assert bundle.exists()
        with tarfile.open(bundle, "r:gz") as archive:
            names = set(archive.getnames())
        assert names == {f"tmp/export/{name}" for name in REQUIRED_ONNX_MEMBERS}


def test_compare_reports_pairs_identical_cases() -> None:
    baseline = {
        "case_results": [
            {
                "case_id": "00001",
                "mode": "offline",
                "noise_category": "uav_drone",
                "snr_db": 0.0,
                "si_sdr": 10.0,
                "stoi": 0.8,
                "pesq": 1.5,
                "snr": 9.0,
                "status": "success",
            },
            {
                "case_id": "00001",
                "mode": "streaming",
                "noise_category": "uav_drone",
                "snr_db": 0.0,
                "si_sdr": 9.0,
                "stoi": 0.7,
                "pesq": 1.4,
                "snr": 8.0,
                "status": "success",
            },
        ]
    }
    candidate = {
        "case_results": [
            {
                "case_id": "00001",
                "mode": "offline",
                "noise_category": "uav_drone",
                "snr_db": 0.0,
                "si_sdr": 11.0,
                "stoi": 0.82,
                "pesq": 1.6,
                "snr": 9.5,
                "status": "success",
            },
            {
                "case_id": "00001",
                "mode": "streaming",
                "noise_category": "uav_drone",
                "snr_db": 0.0,
                "si_sdr": 8.0,
                "stoi": 0.69,
                "pesq": 1.3,
                "snr": 7.5,
                "status": "success",
            },
        ]
    }

    report = compare_benchmark_reports(baseline, candidate)
    assert report["num_paired_rows"] == 2
    assert report["metrics"]["si_sdr"]["mean_difference"] == 0.0
    assert report["metrics"]["si_sdr"]["improved"] == 1
    assert report["metrics"]["si_sdr"]["degraded"] == 1
    assert report["by_mode"]["offline"]["si_sdr"]["mean_difference"] == 1.0
    assert report["by_mode"]["streaming"]["si_sdr"]["mean_difference"] == -1.0
    assert report["recommendation"] == "do_not_replace_pretrained"


def test_compare_reports_recommends_candidate_when_si_sdr_improves() -> None:
    baseline = {
        "case_results": [
            {
                "case_id": "00001",
                "mode": "offline",
                "noise_category": "vehicle_engine",
                "snr_db": 5.0,
                "si_sdr": 10.0,
                "stoi": 0.5,
                "pesq": 1.0,
                "snr": 8.0,
                "status": "success",
            }
        ]
    }
    candidate = json.loads(json.dumps(baseline))
    candidate["case_results"][0]["si_sdr"] = 12.0
    report = compare_benchmark_reports(baseline, candidate)
    assert report["recommendation"] == "candidate_beats_pretrained_on_mean_si_sdr"


def test_finetuned_load_and_process_short_audio() -> None:
    if os.environ.get("DFN3_FINETUNED_INTEGRATION") != "1":
        return

    import torch

    enhancer = FineTunedDeepFilterNetEnhancer()
    enhancer.load()
    assert enhancer.sample_rate() == 48_000
    assert enhancer.name() == FINETUNED_MODEL_NAME

    audio = torch.randn(4800, dtype=torch.float32)
    offline = enhancer.process(audio)
    assert offline.ndim in (1, 2)
    assert torch.isfinite(offline).all()

    streamed = enhancer.process_stream(audio)
    tail = enhancer.flush()
    combined = torch.cat([streamed, tail]) if tail.numel() else streamed
    assert combined.numel() > 0
    assert torch.isfinite(combined).all()
    enhancer.reset()


def main() -> None:
    print("=" * 70)
    print("DRDO-ANC | Fine-tuned DFN3 Tests")
    print("=" * 70)

    tests = [
        test_registry_lists_pretrained_and_finetuned,
        test_pretrained_factory_unchanged,
        test_artifact_paths_exist,
        test_onnx_bundle_uses_native_layout,
        test_compare_reports_pairs_identical_cases,
        test_compare_reports_recommends_candidate_when_si_sdr_improves,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    if os.environ.get("DFN3_FINETUNED_INTEGRATION") == "1":
        test_finetuned_load_and_process_short_audio()
        print("PASS: test_finetuned_load_and_process_short_audio")
    else:
        print("SKIP: test_finetuned_load_and_process_short_audio")

    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
