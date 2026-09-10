"""Deterministic spectral and temporal feature extraction for noise analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .categories import NOISE_CLASSES

# Align with live session analysis: 50 ms window / 25 ms hop at 16 kHz.
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_WINDOW_MS = 50.0
DEFAULT_HOP_MS = 25.0
DEFAULT_N_FFT = 512


@dataclass(frozen=True)
class FrameFeatures:
    """Features extracted from one analysis frame."""

    rms_db: float
    zero_crossing_rate: float
    spectral_centroid_hz: float
    spectral_bandwidth_hz: float
    spectral_flatness: float
    spectral_rolloff_hz: float
    low_band_ratio: float
    mid_band_ratio: float
    high_band_ratio: float
    crest_factor: float
    peak_envelope: float


@dataclass(frozen=True)
class AggregatedFeatures:
    """Summary statistics aggregated across analysis frames."""

    num_frames: int
    rms_db: float
    zero_crossing_rate: float
    spectral_centroid_hz: float
    spectral_bandwidth_hz: float
    spectral_flatness: float
    spectral_rolloff_hz: float
    low_band_ratio: float
    mid_band_ratio: float
    high_band_ratio: float
    crest_factor: float
    peak_envelope: float
    rms_variability: float
    impulsive_index: float
    tonal_index: float
    modulation_index: float
    silence_ratio: float


def window_samples(
    sample_rate: int,
    window_ms: float = DEFAULT_WINDOW_MS,
) -> int:
    return max(1, int(round(sample_rate * window_ms / 1000.0)))


def hop_samples(
    sample_rate: int,
    hop_ms: float = DEFAULT_HOP_MS,
) -> int:
    return max(1, int(round(sample_rate * hop_ms / 1000.0)))


def sanitize_audio(audio: np.ndarray) -> np.ndarray:
    """Replace non-finite samples with zero."""

    array = np.asarray(audio, dtype=np.float32)

    if array.ndim != 1:
        raise ValueError(f"Expected mono audio [T], got shape {array.shape}")

    if array.size == 0:
        return array

    if not np.isfinite(array).all():
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

    return array


def _rms_db(audio: np.ndarray) -> float:
    if audio.size == 0:
        return float("-inf")

    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))

    if rms <= 0.0:
        return float("-inf")

    return float(20.0 * np.log10(rms))


def _zero_crossing_rate(audio: np.ndarray) -> float:
    if audio.size < 2:
        return 0.0

    signs = np.signbit(audio)
    crossings = np.count_nonzero(signs[1:] != signs[:-1])

    return float(crossings / (audio.size - 1))


def _crest_factor(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0

    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))

    if rms <= 1e-12:
        return 0.0

    peak = float(np.max(np.abs(audio)))

    return peak / rms


def _hann_window(length: int) -> np.ndarray:
    return np.hanning(length).astype(np.float32)


def _frame_spectrum(
    frame: np.ndarray,
    sample_rate: int,
    n_fft: int,
) -> tuple[np.ndarray, np.ndarray]:
    window = _hann_window(len(frame))
    windowed = frame * window
    spectrum = np.fft.rfft(windowed, n=n_fft)
    magnitude = np.abs(spectrum).astype(np.float64)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

    return freqs, magnitude


def _band_energy_ratio(
    freqs: np.ndarray,
    magnitude: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> float:
    power = np.square(magnitude)
    total = float(np.sum(power))

    if total <= 1e-20:
        return 0.0

    mask = (freqs >= low_hz) & (freqs < high_hz)
    band = float(np.sum(power[mask]))

    return band / total


def _spectral_centroid(
    freqs: np.ndarray,
    magnitude: np.ndarray,
) -> float:
    total = float(np.sum(magnitude))

    if total <= 1e-20:
        return 0.0

    return float(np.sum(freqs * magnitude) / total)


def _spectral_bandwidth(
    freqs: np.ndarray,
    magnitude: np.ndarray,
    centroid_hz: float,
) -> float:
    total = float(np.sum(magnitude))

    if total <= 1e-20:
        return 0.0

    deviation = np.square(freqs - centroid_hz)

    return float(np.sqrt(np.sum(deviation * magnitude) / total))


def _spectral_flatness(magnitude: np.ndarray) -> float:
    positive = magnitude[magnitude > 1e-20]

    if positive.size == 0:
        return 0.0

    geometric = float(np.exp(np.mean(np.log(positive))))
    arithmetic = float(np.mean(positive))

    if arithmetic <= 1e-20:
        return 0.0

    return geometric / arithmetic


def _spectral_rolloff(
    freqs: np.ndarray,
    magnitude: np.ndarray,
    percentile: float = 0.85,
) -> float:
    power = np.square(magnitude)
    cumulative = np.cumsum(power)
    total = float(cumulative[-1]) if cumulative.size else 0.0

    if total <= 1e-20:
        return 0.0

    threshold = percentile * total
    index = int(np.searchsorted(cumulative, threshold, side="left"))
    index = min(index, len(freqs) - 1)

    return float(freqs[index])


def _tonal_index(magnitude: np.ndarray) -> float:
    """Share of spectral energy concentrated in local magnitude peaks."""

    if magnitude.size < 3:
        return 0.0

    total = float(np.sum(np.square(magnitude)))

    if total <= 1e-20:
        return 0.0

    local_max = magnitude[1:-1] > magnitude[:-2]
    local_max &= magnitude[1:-1] >= magnitude[2:]
    peak_power = float(np.sum(np.square(magnitude[1:-1][local_max])))

    return peak_power / total


def extract_frame_features(
    frame: np.ndarray,
    sample_rate: int,
    *,
    n_fft: int = DEFAULT_N_FFT,
    silence_threshold_db: float = -55.0,
) -> FrameFeatures:
    frame = sanitize_audio(frame)

    rms_db = _rms_db(frame)
    zcr = _zero_crossing_rate(frame)
    crest = _crest_factor(frame)
    peak_env = float(np.max(np.abs(frame))) if frame.size else 0.0

    if rms_db <= silence_threshold_db or frame.size == 0:
        return FrameFeatures(
            rms_db=rms_db,
            zero_crossing_rate=zcr,
            spectral_centroid_hz=0.0,
            spectral_bandwidth_hz=0.0,
            spectral_flatness=0.0,
            spectral_rolloff_hz=0.0,
            low_band_ratio=0.0,
            mid_band_ratio=0.0,
            high_band_ratio=0.0,
            crest_factor=crest,
            peak_envelope=peak_env,
        )

    freqs, magnitude = _frame_spectrum(frame, sample_rate, n_fft)
    centroid = _spectral_centroid(freqs, magnitude)

    return FrameFeatures(
        rms_db=rms_db,
        zero_crossing_rate=zcr,
        spectral_centroid_hz=centroid,
        spectral_bandwidth_hz=_spectral_bandwidth(
            freqs,
            magnitude,
            centroid,
        ),
        spectral_flatness=_spectral_flatness(magnitude),
        spectral_rolloff_hz=_spectral_rolloff(freqs, magnitude),
        low_band_ratio=_band_energy_ratio(freqs, magnitude, 0.0, 400.0),
        mid_band_ratio=_band_energy_ratio(freqs, magnitude, 400.0, 2000.0),
        high_band_ratio=_band_energy_ratio(freqs, magnitude, 2000.0, 8000.0),
        crest_factor=crest,
        peak_envelope=peak_env,
    )


def iter_analysis_frames(
    audio: np.ndarray,
    sample_rate: int,
    *,
    window_ms: float = DEFAULT_WINDOW_MS,
    hop_ms: float = DEFAULT_HOP_MS,
) -> list[np.ndarray]:
    audio = sanitize_audio(audio)
    window = window_samples(sample_rate, window_ms)
    hop = hop_samples(sample_rate, hop_ms)

    if audio.size < window:
        return [audio] if audio.size > 0 else []

    frames: list[np.ndarray] = []

    for start in range(0, audio.size - window + 1, hop):
        frames.append(audio[start : start + window])

    return frames


def extract_features(
    audio: np.ndarray,
    sample_rate: int,
    *,
    window_ms: float = DEFAULT_WINDOW_MS,
    hop_ms: float = DEFAULT_HOP_MS,
    n_fft: int = DEFAULT_N_FFT,
    silence_threshold_db: float = -55.0,
) -> AggregatedFeatures:
    """Extract deterministic aggregated features from mono audio."""

    frames = iter_analysis_frames(
        audio,
        sample_rate,
        window_ms=window_ms,
        hop_ms=hop_ms,
    )

    if not frames:
        return AggregatedFeatures(
            num_frames=0,
            rms_db=float("-inf"),
            zero_crossing_rate=0.0,
            spectral_centroid_hz=0.0,
            spectral_bandwidth_hz=0.0,
            spectral_flatness=0.0,
            spectral_rolloff_hz=0.0,
            low_band_ratio=0.0,
            mid_band_ratio=0.0,
            high_band_ratio=0.0,
            crest_factor=0.0,
            peak_envelope=0.0,
            rms_variability=0.0,
            impulsive_index=0.0,
            tonal_index=0.0,
            modulation_index=0.0,
            silence_ratio=1.0,
        )

    frame_features = [
        extract_frame_features(
            frame,
            sample_rate,
            n_fft=n_fft,
            silence_threshold_db=silence_threshold_db,
        )
        for frame in frames
    ]

    active = [
        feature
        for feature in frame_features
        if feature.rms_db > silence_threshold_db
    ]

    silence_ratio = 1.0 - (len(active) / len(frame_features))

    if not active:
        return AggregatedFeatures(
            num_frames=len(frame_features),
            rms_db=float("-inf"),
            zero_crossing_rate=0.0,
            spectral_centroid_hz=0.0,
            spectral_bandwidth_hz=0.0,
            spectral_flatness=0.0,
            spectral_rolloff_hz=0.0,
            low_band_ratio=0.0,
            mid_band_ratio=0.0,
            high_band_ratio=0.0,
            crest_factor=0.0,
            peak_envelope=0.0,
            rms_variability=0.0,
            impulsive_index=0.0,
            tonal_index=0.0,
            modulation_index=0.0,
            silence_ratio=silence_ratio,
        )

    rms_values = np.array(
        [feature.rms_db for feature in active],
        dtype=np.float64,
    )
    crest_values = np.array(
        [feature.crest_factor for feature in active],
        dtype=np.float64,
    )

    tonal_values = []

    for frame in frames:
        if frame.size == 0:
            continue

        freqs, magnitude = _frame_spectrum(frame, sample_rate, n_fft)
        tonal_values.append(_tonal_index(magnitude))

    rms_linear = np.power(10.0, rms_values / 20.0)
    rms_mean = float(np.mean(rms_linear))
    rms_std = float(np.std(rms_linear))
    modulation_index = rms_std / (rms_mean + 1e-12)

    impulsive_threshold = 4.0
    impulsive_index = float(
        np.mean(crest_values > impulsive_threshold)
    )

    return AggregatedFeatures(
        num_frames=len(frame_features),
        rms_db=float(np.mean(rms_values)),
        zero_crossing_rate=float(
            np.mean([feature.zero_crossing_rate for feature in active])
        ),
        spectral_centroid_hz=float(
            np.mean([feature.spectral_centroid_hz for feature in active])
        ),
        spectral_bandwidth_hz=float(
            np.mean([feature.spectral_bandwidth_hz for feature in active])
        ),
        spectral_flatness=float(
            np.mean([feature.spectral_flatness for feature in active])
        ),
        spectral_rolloff_hz=float(
            np.mean([feature.spectral_rolloff_hz for feature in active])
        ),
        low_band_ratio=float(
            np.mean([feature.low_band_ratio for feature in active])
        ),
        mid_band_ratio=float(
            np.mean([feature.mid_band_ratio for feature in active])
        ),
        high_band_ratio=float(
            np.mean([feature.high_band_ratio for feature in active])
        ),
        crest_factor=float(np.mean(crest_values)),
        peak_envelope=float(
            np.max([feature.peak_envelope for feature in active])
        ),
        rms_variability=float(
            rms_std / (rms_mean + 1e-12)
        ),
        impulsive_index=impulsive_index,
        tonal_index=float(np.mean(tonal_values)) if tonal_values else 0.0,
        modulation_index=float(modulation_index),
        silence_ratio=float(silence_ratio),
    )


FEATURE_NAMES: tuple[str, ...] = (
    "rms_db",
    "zero_crossing_rate",
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
    "spectral_flatness",
    "spectral_rolloff_hz",
    "low_band_ratio",
    "mid_band_ratio",
    "high_band_ratio",
    "crest_factor",
    "peak_envelope",
    "rms_variability",
    "impulsive_index",
    "tonal_index",
    "modulation_index",
    "silence_ratio",
)

FEATURE_VERSION = "noise-features-v1"


def feature_vector(features: AggregatedFeatures) -> np.ndarray:
    """Return a fixed-order numeric vector for inspection/testing."""

    return np.array(
        [
            features.rms_db,
            features.zero_crossing_rate,
            features.spectral_centroid_hz,
            features.spectral_bandwidth_hz,
            features.spectral_flatness,
            features.spectral_rolloff_hz,
            features.low_band_ratio,
            features.mid_band_ratio,
            features.high_band_ratio,
            features.crest_factor,
            features.peak_envelope,
            features.rms_variability,
            features.impulsive_index,
            features.tonal_index,
            features.modulation_index,
            features.silence_ratio,
        ],
        dtype=np.float64,
    )


def validate_class_names(class_names: tuple[str, ...]) -> None:
    if tuple(class_names) != NOISE_CLASSES:
        raise ValueError(
            f"Unexpected class names: {class_names}. "
            f"Expected {NOISE_CLASSES}."
        )
