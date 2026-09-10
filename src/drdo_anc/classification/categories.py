"""Noise-classification category constants."""

from __future__ import annotations

NOISE_CLASSES: tuple[str, ...] = (
    "uav_drone",
    "vehicle_engine",
    "impulsive_firearms",
    "unknown",
)

DEFENCE_NOISE_CATEGORIES: tuple[str, ...] = (
    "uav_drone",
    "vehicle_engine",
    "impulsive_firearms",
)

UNKNOWN_CLASS = "unknown"
