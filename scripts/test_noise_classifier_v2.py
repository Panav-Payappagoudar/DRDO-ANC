"""Unit tests for Noise Classifier v2 (supervised)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from drdo_anc.classification import (
    NOISE_CLASSES,
    NoiseClassifier,
    extract_features,
    feature_vector,
)
from drdo_anc.classification.features import FEATURE_NAMES, FEATURE_VERSION
from drdo_anc.classification.source_identity import recording_source_id
from drdo_anc.classification.supervised import SupervisedNoiseClassifier
from drdo_anc.classification.training import (
    LabeledFeatureRow,
    build_candidate_estimators,
    evaluate_predictions,
    stratified_recording_split,
)
from drdo_anc.dataset.source_sample import SourceSample
from sklearn.pipeline import Pipeline


SAMPLE_RATE = 16_000
CHUNK_SIZES = (1, 17, 128, 480, 800, 1024)


def _sine(length: int, frequency_hz: float, amplitude: float = 0.3) -> np.ndarray:
    time = np.arange(length, dtype=np.float32) / np.float32(SAMPLE_RATE)
    return (
        amplitude * np.sin(np.float32(2.0 * np.pi * frequency_hz) * time)
    ).astype(np.float32)


def _source(sample_id: str, filename: str, dataset_source: str) -> SourceSample:
    return SourceSample(
        sample_id=sample_id,
        archive_name="test.zip",
        internal_path=f"x/{filename}",
        filename=filename,
        parent_folder="x",
        file_size_bytes=100,
        audio_class="noise",
        dataset_source=dataset_source,
        inferred_subclass="sub",
    )


def _feature_row(
    clip_id: str,
    recording_id: str,
    label: str,
    seed: int,
) -> LabeledFeatureRow:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=len(FEATURE_NAMES)).astype(np.float64)
    # Class-separable means for deterministic learning in tiny tests.
    offset = {
        "uav_drone": 0.0,
        "vehicle_engine": 3.0,
        "impulsive_firearms": -3.0,
    }[label]
    features[0] += offset
    features[6] += offset * 0.2
    return LabeledFeatureRow(
        clip_id=clip_id,
        source_id=f"src/{clip_id}",
        recording_id=recording_id,
        label=label,
        features=features,
        num_samples=8000,
        sample_rate=SAMPLE_RATE,
    )


def _tiny_supervised_classifier() -> SupervisedNoiseClassifier:
    rows: list[LabeledFeatureRow] = []
    labels = ("uav_drone", "vehicle_engine", "impulsive_firearms")
    for class_index, label in enumerate(labels):
        for sample_index in range(12):
            rows.append(
                _feature_row(
                    f"{label}_{sample_index:02d}",
                    f"rec_{label}_{sample_index // 2}",
                    label,
                    seed=1000 + class_index * 100 + sample_index,
                )
            )

    split = stratified_recording_split(rows, seed=7)
    estimator = build_candidate_estimators(7)["logistic_regression"]
    x = np.vstack([row.features for row in split.train])
    y = np.asarray([row.label for row in split.train], dtype=object)
    estimator.fit(x, y)

    return SupervisedNoiseClassifier(
        estimator,
        model_name="logistic_regression",
    )


def test_recording_source_identity_rules() -> None:
    esc_a = _source(
        "Vehicle/1-100210-A-36.wav",
        "1-100210-A-36.wav",
        "Vehicle-Engine-Wind-Electronic-Electrical-Noise",
    )
    esc_b = _source(
        "Vehicle/1-100210-B-36.wav",
        "1-100210-B-36.wav",
        "Vehicle-Engine-Wind-Electronic-Electrical-Noise",
    )
    assert recording_source_id(esc_a, "vehicle_engine") == recording_source_id(
        esc_b,
        "vehicle_engine",
    )

    gun_a = _source(
        "fire/a.wav",
        "ak-47_002_3ab0f1bb-f7fe-4cfe-a78e-11a263f75819_chan1_v0.wav",
        "firearms-audio-dataset-contains-58-guntypes",
    )
    gun_b = _source(
        "fire/b.wav",
        "ak-47_024_3ab0f1bb-f7fe-4cfe-a78e-11a263f75819_chan0_v0.wav",
        "firearms-audio-dataset-contains-58-guntypes",
    )
    assert recording_source_id(gun_a, "impulsive_firearms") == recording_source_id(
        gun_b,
        "impulsive_firearms",
    )

    drone = _source(
        "Drone/drone_1.wav",
        "drone_1764762657604.wav",
        "Drone-Noise-Audio-set",
    )
    assert recording_source_id(drone, "uav_drone").endswith(drone.sample_id)


def test_deterministic_training_split() -> None:
    rows = [
        _feature_row(f"uav_{i}", f"uav_rec_{i}", "uav_drone", i)
        for i in range(20)
    ] + [
        _feature_row(f"veh_{i}", f"veh_rec_{i}", "vehicle_engine", 100 + i)
        for i in range(20)
    ] + [
        _feature_row(f"gun_{i}", f"gun_rec_{i}", "impulsive_firearms", 200 + i)
        for i in range(20)
    ]

    first = stratified_recording_split(rows, seed=42)
    second = stratified_recording_split(rows, seed=42)
    third = stratified_recording_split(rows, seed=43)

    assert [row.clip_id for row in first.train] == [
        row.clip_id for row in second.train
    ]
    assert [row.clip_id for row in first.test] != [
        row.clip_id for row in third.test
    ]
    assert first.class_distribution()["train"]["uav_drone"] > 0


def test_no_source_leakage_between_splits() -> None:
    rows = []
    for label, prefix in (
        ("uav_drone", "u"),
        ("vehicle_engine", "v"),
        ("impulsive_firearms", "g"),
    ):
        for index in range(15):
            # Two clips share a recording id to exercise grouping.
            recording_id = f"{prefix}_rec_{index // 3}"
            rows.append(
                _feature_row(
                    f"{prefix}_{index}",
                    recording_id,
                    label,
                    seed=index + 17,
                )
            )

    split = stratified_recording_split(rows, seed=11)
    train_ids = split.recording_ids("train")
    val_ids = split.recording_ids("validation")
    test_ids = split.recording_ids("test")

    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
    assert not (val_ids & test_ids)


def test_model_save_load_and_prediction_shape(tmp_path: Path | None = None) -> None:
    if tmp_path is None:
        tmp_path = Path("data") / "classifier_results" / "_v2_test_tmp"
        tmp_path.mkdir(parents=True, exist_ok=True)

    classifier = _tiny_supervised_classifier()
    audio = _sine(8_000, 120.0)
    result = classifier.classify(audio)

    assert result.predicted_class in NOISE_CLASSES
    assert set(result.probabilities) == set(NOISE_CLASSES)
    assert np.isclose(sum(result.probabilities.values()), 1.0, atol=1e-6)
    assert result.confidence >= 0.0
    assert feature_vector(result.features).shape == (len(FEATURE_NAMES),)

    classifier.save(tmp_path)
    loaded = SupervisedNoiseClassifier.load(tmp_path)
    assert loaded.meta is not None
    assert loaded.meta.feature_version == FEATURE_VERSION

    again = loaded.classify(audio)
    assert again.predicted_class == result.predicted_class
    assert again.probabilities == result.probabilities


def test_probability_normalization_and_chunks() -> None:
    classifier = _tiny_supervised_classifier()
    audio = _sine(6_000, 90.0) + 0.1 * _sine(6_000, 180.0)

    reference = classifier.classify(audio)
    streaming = _tiny_supervised_classifier()
    offset = 0
    while offset < audio.size:
        chunk = audio[offset : offset + 480]
        streaming.process_chunk(chunk)
        offset += 480
    streamed = streaming.classify_buffered()

    assert streamed.predicted_class == reference.predicted_class
    assert np.isclose(
        sum(streamed.probabilities.values()),
        1.0,
        atol=1e-6,
    )

    for chunk_size in CHUNK_SIZES:
        streaming.reset()
        offset = 0
        while offset < audio.size:
            streaming.process_chunk(audio[offset : offset + chunk_size])
            offset += chunk_size
        result = streaming.classify_buffered()
        assert result.predicted_class == reference.predicted_class


def test_nan_inf_handling() -> None:
    classifier = _tiny_supervised_classifier()
    audio = _sine(4_000, 150.0)
    audio[5] = np.nan
    audio[9] = np.inf
    audio[11] = -np.inf

    result = classifier.classify(audio)
    assert result.predicted_class in NOISE_CLASSES
    assert np.isfinite(feature_vector(result.features)).all()


def test_deterministic_inference() -> None:
    classifier = _tiny_supervised_classifier()
    audio = _sine(5_000, 70.0)
    first = classifier.classify(audio)
    second = classifier.classify(audio)
    assert first.predicted_class == second.predicted_class
    assert first.probabilities == second.probabilities
    assert first.confidence == second.confidence


def test_serialized_model_compatibility(tmp_path: Path | None = None) -> None:
    if tmp_path is None:
        tmp_path = Path("data") / "classifier_results" / "_v2_compat_tmp"
        tmp_path.mkdir(parents=True, exist_ok=True)

    classifier = _tiny_supervised_classifier()
    classifier.save(tmp_path)
    meta = json.loads((tmp_path / "model_meta.json").read_text(encoding="utf-8"))
    assert meta["feature_version"] == FEATURE_VERSION
    assert meta["feature_names"] == list(FEATURE_NAMES)

    loaded = SupervisedNoiseClassifier.load(tmp_path)
    assert isinstance(loaded.estimator, Pipeline)


def test_evaluate_predictions_shape() -> None:
    y_true = np.asarray(
        ["uav_drone", "vehicle_engine", "impulsive_firearms", "uav_drone"],
        dtype=object,
    )
    y_pred = np.asarray(
        ["uav_drone", "vehicle_engine", "uav_drone", "uav_drone"],
        dtype=object,
    )
    metrics = evaluate_predictions(y_true, y_pred)
    assert "macro_f1" in metrics
    assert metrics["confusion_matrix"]["uav_drone"]["uav_drone"] == 2


def test_v1_still_importable() -> None:
    classifier = NoiseClassifier()
    result = classifier.classify(np.zeros(1000, dtype=np.float32))
    assert result.predicted_class in NOISE_CLASSES


def test_integration_real_split_no_leakage() -> None:
    if os.environ.get("SIH26_INTEGRATION") != "1":
        return

    from huggingface_hub import hf_hub_download
    from drdo_anc.classification.evaluation import build_defence_noise_corpus
    from drdo_anc.classification.training import extract_labeled_features
    from drdo_anc.dataset.manifest import (
        SIH26_METADATA_FILENAME,
        SIH26_REPO_ID,
    )
    from drdo_anc.dataset.zip_manifest_dataset import ZipManifestDataset

    metadata_path = Path(
        hf_hub_download(
            repo_id=SIH26_REPO_ID,
            repo_type="dataset",
            filename=SIH26_METADATA_FILENAME,
        )
    )
    clips = build_defence_noise_corpus(metadata_path, max_per_class=8)
    dataset = ZipManifestDataset(
        metadata_path=metadata_path,
        repo_id=SIH26_REPO_ID,
    )
    rows = extract_labeled_features(clips, dataset)
    split = stratified_recording_split(rows, seed=42)
    assert not (
        split.recording_ids("train") & split.recording_ids("test")
    )


def main() -> None:
    print("=" * 70)
    print("DRDO-ANC | Noise Classifier v2 Tests")
    print("=" * 70)

    tests = [
        test_recording_source_identity_rules,
        test_deterministic_training_split,
        test_no_source_leakage_between_splits,
        test_model_save_load_and_prediction_shape,
        test_probability_normalization_and_chunks,
        test_nan_inf_handling,
        test_deterministic_inference,
        test_serialized_model_compatibility,
        test_evaluate_predictions_shape,
        test_v1_still_importable,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    if os.environ.get("SIH26_INTEGRATION") == "1":
        test_integration_real_split_no_leakage()
        print("PASS: test_integration_real_split_no_leakage")
    else:
        print("SKIP: test_integration_real_split_no_leakage")

    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
