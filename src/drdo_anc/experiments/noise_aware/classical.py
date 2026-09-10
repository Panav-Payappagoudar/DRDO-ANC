"""Deterministic classical spectral-subtraction baseline (experiment-local)."""

from __future__ import annotations

import numpy as np


def spectral_subtraction(
    noisy: np.ndarray,
    sample_rate: int,
    *,
    frame_ms: float = 32.0,
    hop_ms: float = 16.0,
    noise_frames: int = 8,
    oversubtraction: float = 1.0,
    spectral_floor: float = 0.02,
) -> np.ndarray:
    """
    Simple magnitude spectral subtraction.

    Estimates noise from the first ``noise_frames`` analysis frames.
    Deterministic for a fixed input; no randomness.
    """

    audio = np.asarray(noisy, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f"Expected mono audio [T], got shape {audio.shape}")
    if audio.size == 0:
        return audio.copy()
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")

    frame_length = max(8, int(round(sample_rate * frame_ms / 1000.0)))
    hop_length = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    n_fft = int(2 ** np.ceil(np.log2(frame_length)))
    window = np.hanning(frame_length).astype(np.float32)

    if audio.size < frame_length:
        padded = np.zeros(frame_length, dtype=np.float32)
        padded[: audio.size] = audio
        frames_src = padded
    else:
        frames_src = audio

    spectra: list[np.ndarray] = []
    for start in range(0, max(1, len(frames_src) - frame_length + 1), hop_length):
        frame = frames_src[start : start + frame_length]
        if frame.size < frame_length:
            tmp = np.zeros(frame_length, dtype=np.float32)
            tmp[: frame.size] = frame
            frame = tmp
        spectrum = np.fft.rfft(frame * window, n=n_fft)
        spectra.append(spectrum)

    if not spectra:
        return audio.copy()

    magnitudes = np.stack([np.abs(s) for s in spectra], axis=0)
    phases = np.stack([np.angle(s) for s in spectra], axis=0)

    noise_count = max(1, min(noise_frames, magnitudes.shape[0]))
    noise_mag = np.mean(magnitudes[:noise_count], axis=0)

    cleaned = magnitudes - oversubtraction * noise_mag
    cleaned = np.maximum(cleaned, spectral_floor * noise_mag)

    reconstructed = cleaned * np.exp(1j * phases)

    output = np.zeros(len(frames_src) + n_fft, dtype=np.float32)
    window_sum = np.zeros_like(output)
    window_sq = window * window

    for index, spectrum in enumerate(reconstructed):
        start = index * hop_length
        frame = np.fft.irfft(spectrum, n=n_fft).astype(np.float32)[:frame_length]
        output[start : start + frame_length] += frame * window
        window_sum[start : start + frame_length] += window_sq

    nonzero = window_sum > 1e-8
    output[nonzero] /= window_sum[nonzero]
    output = output[: len(audio)]

    if not np.isfinite(output).all():
        output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)

    return output.astype(np.float32, copy=False)
