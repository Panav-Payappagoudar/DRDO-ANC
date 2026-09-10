"""Unit tests for the rule-based noise classifier (v1)."""

from __future__ import annotations

import numpy as np

from drdo_anc.benchmark import MixtureGenerator, build_development_manifest
from drdo_anc.classification import (
    NOISE_CLASSES,
    UNKNOWN_CLASS,
    NoiseClassifier,
    extract_features,
    feature_vector,
)
from drdo_anc.dataset import ZipManifestDataset

from build_evaluation_fixtures import (
    FIXTURE_DIR,
    METADATA_PATH,
    build_fixtures,
)


CHUNK_SIZES = (1, 17, 128, 480, 800, 1024, 2048)
SAMPLE_RATE = 16_000


def _sine_wave(
    length: int,
    frequency_hz: float,
    *,
    amplitude: float = 0.35,
    phase: float = 0.0,
) -> np.ndarray:
    time = np.arange(length, dtype=np.float32) / np.float32(SAMPLE_RATE)
    return (
        amplitude
        * np.sin(
            np.float32(2.0 * np.pi * frequency_hz) * time
            + np.float32(phase)
        )
    ).astype(np.float32)


def _load_fixture_noise_by_category(category: str) -> np.ndarray:
    manifest = build_development_manifest(METADATA_PATH)
    dataset = ZipManifestDataset(
        metadata_path=METADATA_PATH,
        repo_id=None,
        archive_dir=FIXTURE_DIR,
    )

    for case in manifest.cases:
        if case.noise_category == category:
            audio, sample_rate = dataset.load_audio(case.noise_source)
            assert sample_rate == SAMPLE_RATE
            return audio

    raise AssertionError(f"No fixture case found for category {category}")


def setup_module() -> None:
    build_fixtures()


def test_feature_extraction_shape_and_finiteness() -> None:
    audio = _sine_wave(8_000, 220.0)
    features = extract_features(audio, SAMPLE_RATE)

    vector = feature_vector(features)

    assert vector.shape == (16,)
    assert np.isfinite(vector).all()
    assert features.num_frames > 0


def test_silence_classified_as_unknown() -> None:
    classifier = NoiseClassifier(sample_rate=SAMPLE_RATE)
    result = classifier.classify(np.zeros(8_000, dtype=np.float32))

    assert result.predicted_class == UNKNOWN_CLASS
    assert result.probabilities[UNKNOWN_CLASS] >= result.probabilities["uav_drone"]


def test_speech_like_audio_produces_finite_features() -> None:
    speech_like = (
        0.25 * _sine_wave(8_000, 180.0)
        + 0.20 * _sine_wave(8_000, 360.0, phase=0.3)
        + 0.10 * _sine_wave(8_000, 720.0, phase=0.7)
    ).astype(np.float32)

    classifier = NoiseClassifier(sample_rate=SAMPLE_RATE)
    result = classifier.classify(speech_like)

    assert result.predicted_class in NOISE_CLASSES
    assert np.isclose(sum(result.probabilities.values()), 1.0, atol=1e-6)
    assert np.isfinite(feature_vector(result.features)).all()


def test_stationary_engine_like_signal() -> None:
    engine_like = (
        0.55 * _sine_wave(12_000, 95.0)
        + 0.25 * _sine_wave(12_000, 190.0, phase=0.2)
        + 0.10 * _sine_wave(12_000, 285.0, phase=0.4)
    ).astype(np.float32)

    features = extract_features(engine_like, SAMPLE_RATE)

    assert features.low_band_ratio > 0.5
    assert features.impulsive_index < 0.2


def test_drone_like_signal_from_fixture() -> None:
    audio = _load_fixture_noise_by_category("uav_drone")
    features = extract_features(audio, SAMPLE_RATE)

    assert features.num_frames > 0
    assert np.isfinite(feature_vector(features)).all()


def test_impulsive_signal_features() -> None:
    length = 8_000
    audio = np.zeros(length, dtype=np.float32)
    spike_positions = (400, 1200, 2100, 3500, 5200, 7100)

    for position in spike_positions:
        audio[position] = 0.9

    features = extract_features(audio, SAMPLE_RATE)

    assert features.impulsive_index > 0.0
    assert features.crest_factor > 3.0


def test_arbitrary_chunk_sizes_match_single_shot() -> None:
    audio = _load_fixture_noise_by_category("vehicle_engine")

    single_shot = NoiseClassifier(sample_rate=SAMPLE_RATE)
    reference = single_shot.classify(audio)

    for chunk_size in CHUNK_SIZES:
        streaming = NoiseClassifier(sample_rate=SAMPLE_RATE)
        offset = 0

        while offset < audio.size:
            streaming.process_chunk(audio[offset : offset + chunk_size])
            offset += chunk_size

        streamed = streaming.classify_buffered()

        assert streamed.predicted_class == reference.predicted_class
        assert streamed.probabilities == reference.probabilities


def test_nan_and_inf_handling() -> None:
    audio = _sine_wave(4_000, 150.0)
    corrupted = audio.copy()
    corrupted[10] = np.nan
    corrupted[20] = np.inf
    corrupted[30] = -np.inf

    classifier = NoiseClassifier(sample_rate=SAMPLE_RATE)
    result = classifier.classify(corrupted)

    assert result.predicted_class in NOISE_CLASSES
    assert np.isfinite(feature_vector(result.features)).all()


def test_deterministic_output() -> None:
    audio = _load_fixture_noise_by_category("impulsive_firearms")

    first = NoiseClassifier(sample_rate=SAMPLE_RATE).classify(audio)
    second = NoiseClassifier(sample_rate=SAMPLE_RATE).classify(audio)

    assert first.predicted_class == second.predicted_class
    assert first.probabilities == second.probabilities
    assert np.array_equal(
        feature_vector(first.features),
        feature_vector(second.features),
    )


def test_benchmark_fixture_categories_are_reachable() -> None:
    manifest = build_development_manifest(METADATA_PATH)
    dataset = ZipManifestDataset(
        metadata_path=METADATA_PATH,
        repo_id=None,
        archive_dir=FIXTURE_DIR,
    )
    generator = MixtureGenerator(dataset)

    for case in manifest.cases[:3]:
        mixture = generator.generate(case)
        classifier = NoiseClassifier(sample_rate=mixture.sample_rate)
        result = classifier.classify(mixture.noisy)

        assert result.predicted_class in NOISE_CLASSES


def main() -> None:
    print("=" * 70)
    print("DRDO-ANC | Noise Classifier v1 Tests")
    print("=" * 70)

    build_fixtures()

    tests = [
        test_feature_extraction_shape_and_finiteness,
        test_silence_classified_as_unknown,
        test_speech_like_audio_produces_finite_features,
        test_stationary_engine_like_signal,
        test_drone_like_signal_from_fixture,
        test_impulsive_signal_features,
        test_arbitrary_chunk_sizes_match_single_shot,
        test_nan_and_inf_handling,
        test_deterministic_output,
        test_benchmark_fixture_categories_are_reachable,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
