"""Category → enhancement strategy selection for the offline experiment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class EnhancementStrategy:
    """
    One enhancement configuration that can be applied offline.

    DF3 controls are limited to documented ``atten_lim_db`` on
    ``df.enhance`` (noise attenuation limit in dB). No undocumented
    DF3 internals are modified.
    """

    strategy_id: str
    method: str  # "passthrough" | "classical_spectral_subtraction" | "df3"
    atten_lim_db: float | None = None
    description: str = ""


@dataclass(frozen=True)
class StrategySelection:
    predicted_class: str
    confidence: float
    strategy: EnhancementStrategy
    fallback_reason: str | None = None


STRATEGY_NOISY = EnhancementStrategy(
    strategy_id="noisy",
    method="passthrough",
    description="No enhancement (noisy input passthrough).",
)

STRATEGY_CLASSICAL = EnhancementStrategy(
    strategy_id="classical_spectral_subtraction",
    method="classical_spectral_subtraction",
    description="Deterministic magnitude spectral subtraction.",
)

STRATEGY_DF3_FULL = EnhancementStrategy(
    strategy_id="df3_full",
    method="df3",
    atten_lim_db=None,
    description="DF3 offline enhance with default unlimited attenuation.",
)

STRATEGY_DF3_ATTEN_20 = EnhancementStrategy(
    strategy_id="df3_atten_20",
    method="df3",
    atten_lim_db=20.0,
    description="DF3 with atten_lim_db=20 (moderate attenuation limit).",
)

STRATEGY_DF3_ATTEN_12 = EnhancementStrategy(
    strategy_id="df3_atten_12",
    method="df3",
    atten_lim_db=12.0,
    description=(
        "DF3 with atten_lim_db=12 (documented limit; keeps residual noise)."
    ),
)

DEFAULT_STRATEGY_MAP: dict[str, EnhancementStrategy] = {
    "uav_drone": STRATEGY_DF3_FULL,
    "vehicle_engine": STRATEGY_DF3_ATTEN_20,
    "impulsive_firearms": STRATEGY_DF3_ATTEN_12,
}

FALLBACK_STRATEGY = STRATEGY_DF3_FULL
UNKNOWN_LABELS = frozenset({"unknown", "none", "null", ""})


def select_strategy(
    predicted_class: Any,
    confidence: Any = None,
    *,
    strategy_map: Mapping[str, EnhancementStrategy] | None = None,
    fallback: EnhancementStrategy = FALLBACK_STRATEGY,
    min_confidence: float | None = None,
) -> StrategySelection:
    """
    Map a classifier prediction to an enhancement strategy.

    Malformed / unknown / low-confidence predictions fall back to DF3 full
    (same control surface as the DF3 baseline).
    """

    mapping = dict(strategy_map or DEFAULT_STRATEGY_MAP)

    try:
        confidence_value = float(confidence) if confidence is not None else 0.0
    except (TypeError, ValueError):
        return StrategySelection(
            predicted_class="unknown",
            confidence=0.0,
            strategy=fallback,
            fallback_reason="malformed_confidence",
        )

    if not math.isfinite(confidence_value):
        return StrategySelection(
            predicted_class="unknown",
            confidence=0.0,
            strategy=fallback,
            fallback_reason="non_finite_confidence",
        )

    if predicted_class is None:
        return StrategySelection(
            predicted_class="unknown",
            confidence=confidence_value,
            strategy=fallback,
            fallback_reason="missing_prediction",
        )

    label = str(predicted_class).strip().lower()
    if label in UNKNOWN_LABELS:
        return StrategySelection(
            predicted_class="unknown",
            confidence=confidence_value,
            strategy=fallback,
            fallback_reason="unknown_prediction",
        )

    if min_confidence is not None and confidence_value < min_confidence:
        return StrategySelection(
            predicted_class=label,
            confidence=confidence_value,
            strategy=fallback,
            fallback_reason="low_confidence",
        )

    strategy = mapping.get(label)
    if strategy is None:
        return StrategySelection(
            predicted_class=label,
            confidence=confidence_value,
            strategy=fallback,
            fallback_reason="unmapped_prediction",
        )

    return StrategySelection(
        predicted_class=label,
        confidence=confidence_value,
        strategy=strategy,
        fallback_reason=None,
    )
