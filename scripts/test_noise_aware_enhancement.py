"""Tests for the offline noise-aware enhancement experiment."""

from __future__ import annotations

import numpy as np

from drdo_anc.experiments.noise_aware.classical import spectral_subtraction
from drdo_anc.experiments.noise_aware.strategies import (
    DEFAULT_STRATEGY_MAP,
    FALLBACK_STRATEGY,
    STRATEGY_DF3_ATTEN_12,
    STRATEGY_DF3_ATTEN_20,
    STRATEGY_DF3_FULL,
    STRATEGY_NOISY,
    select_strategy,
)


def test_strategy_mapping_known_classes() -> None:
    drone = select_strategy("uav_drone", 0.9)
    engine = select_strategy("vehicle_engine", 0.8)
    guns = select_strategy("impulsive_firearms", 0.7)

    assert drone.strategy.strategy_id == STRATEGY_DF3_FULL.strategy_id
    assert engine.strategy.strategy_id == STRATEGY_DF3_ATTEN_20.strategy_id
    assert guns.strategy.strategy_id == STRATEGY_DF3_ATTEN_12.strategy_id
    assert drone.fallback_reason is None


def test_classifier_to_strategy_mapping_case_insensitive() -> None:
    result = select_strategy("UAV_DRONE", 0.55)
    assert result.predicted_class == "uav_drone"
    assert result.strategy.atten_lim_db is None


def test_deterministic_strategy_selection() -> None:
    first = select_strategy("vehicle_engine", 0.66)
    second = select_strategy("vehicle_engine", 0.66)
    assert first == second
    assert DEFAULT_STRATEGY_MAP["vehicle_engine"].atten_lim_db == 20.0


def test_fallback_unknown_prediction() -> None:
    result = select_strategy("unknown", 0.99)
    assert result.strategy.strategy_id == FALLBACK_STRATEGY.strategy_id
    assert result.fallback_reason == "unknown_prediction"


def test_fallback_malformed_classifier_output() -> None:
    missing = select_strategy(None, 0.5)
    assert missing.fallback_reason == "missing_prediction"
    assert missing.strategy.strategy_id == FALLBACK_STRATEGY.strategy_id

    bad_conf = select_strategy("uav_drone", "not-a-float")
    assert bad_conf.fallback_reason == "malformed_confidence"

    non_finite = select_strategy("uav_drone", float("nan"))
    assert non_finite.fallback_reason == "non_finite_confidence"

    unmapped = select_strategy("spaceship", 0.9)
    assert unmapped.fallback_reason == "unmapped_prediction"
    assert unmapped.strategy == FALLBACK_STRATEGY


def test_low_confidence_fallback() -> None:
    result = select_strategy(
        "impulsive_firearms",
        0.1,
        min_confidence=0.35,
    )
    assert result.fallback_reason == "low_confidence"
    assert result.strategy.strategy_id == FALLBACK_STRATEGY.strategy_id


def test_unchanged_baseline_strategy_identity() -> None:
    # DF3 baseline must remain full attenuation (None), not a category limit.
    assert STRATEGY_DF3_FULL.atten_lim_db is None
    assert STRATEGY_NOISY.method == "passthrough"
    assert FALLBACK_STRATEGY.strategy_id == STRATEGY_DF3_FULL.strategy_id


def test_spectral_subtraction_deterministic_and_finite() -> None:
    rng = np.random.default_rng(0)
    noisy = rng.standard_normal(16_000).astype(np.float32) * 0.05
    first = spectral_subtraction(noisy, 16_000)
    second = spectral_subtraction(noisy, 16_000)
    assert np.array_equal(first, second)
    assert first.shape == noisy.shape
    assert np.isfinite(first).all()


def test_spectral_subtraction_empty_and_short() -> None:
    empty = spectral_subtraction(np.zeros(0, dtype=np.float32), 16_000)
    assert empty.size == 0
    short = spectral_subtraction(np.ones(100, dtype=np.float32) * 0.01, 16_000)
    assert short.shape == (100,)
    assert np.isfinite(short).all()


def main() -> None:
    print("=" * 70)
    print("DRDO-ANC | Noise-Aware Enhancement Experiment Tests")
    print("=" * 70)

    tests = [
        test_strategy_mapping_known_classes,
        test_classifier_to_strategy_mapping_case_insensitive,
        test_deterministic_strategy_selection,
        test_fallback_unknown_prediction,
        test_fallback_malformed_classifier_output,
        test_low_confidence_fallback,
        test_unchanged_baseline_strategy_identity,
        test_spectral_subtraction_deterministic_and_finite,
        test_spectral_subtraction_empty_and_short,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
