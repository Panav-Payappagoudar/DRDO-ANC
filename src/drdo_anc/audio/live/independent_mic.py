"""
Experimental capture from two independent input devices.

This module is for acoustic/reference investigation only. The two microphones
are **not** hardware-synchronized. Sample indices must not be assumed to align
across devices without inspecting the reported delay and drift metrics.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from drdo_anc.audio.resampling import resample_mono

from .multimic import (
    compute_correlation,
    compute_peak,
    compute_rms,
    estimate_relative_delay_samples,
)
from .sounddevice_backend import (
    _device_channel_count,
    _import_sounddevice,
    downmix_to_mono,
)


CAPTURE_CONDITIONS = (
    "speech_only",
    "stationary_noise",
    "impulsive_noise",
)


@dataclass(frozen=True)
class IndependentMicConfig:
    """Configuration for two-device independent microphone capture."""

    primary_device: int | str
    reference_device: int | str
    sample_rate: int = 48_000
    reference_sample_rate: int | None = None
    duration_s: float = 5.0
    blocksize: int = 1024
    condition: str | None = None

    @property
    def primary_sample_rate(self) -> int:
        return self.sample_rate

    @property
    def effective_reference_sample_rate(self) -> int:
        return (
            self.sample_rate
            if self.reference_sample_rate is None
            else self.reference_sample_rate
        )

    def validate(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        if self.reference_sample_rate is not None and self.reference_sample_rate <= 0:
            raise ValueError("reference_sample_rate must be positive when set.")
        if self.duration_s <= 0.0:
            raise ValueError("duration_s must be positive.")
        if self.blocksize <= 0:
            raise ValueError("blocksize must be positive.")
        if self.condition is not None and self.condition not in CAPTURE_CONDITIONS:
            raise ValueError(
                f"condition must be one of {CAPTURE_CONDITIONS}, "
                f"got {self.condition!r}."
            )

        validate_input_device(self.primary_device, label="primary_device")
        validate_input_device(self.reference_device, label="reference_device")


@dataclass(frozen=True)
class DeviceStreamCapture:
    """Raw capture result from one independent input device."""

    device: int | str
    audio: np.ndarray
    requested_sample_rate: int
    host_sample_rate: float
    samples_captured: int
    capture_elapsed_s: float
    start_time_s: float
    end_time_s: float
    input_overflows: int

    @property
    def estimated_sample_rate(self) -> float:
        if self.capture_elapsed_s <= 0.0:
            return float(self.requested_sample_rate)

        return float(self.samples_captured) / self.capture_elapsed_s

    @property
    def duration_s(self) -> float:
        if self.estimated_sample_rate <= 0.0:
            return 0.0

        return float(self.samples_captured) / self.estimated_sample_rate


@dataclass(frozen=True)
class IndependentPairAnalysis:
    """Analysis for two independent mono captures."""

    requested_sample_rate: int
    alignment_note: str
    primary_rms: float
    reference_rms: float
    primary_peak: float
    reference_peak: float
    correlation: float
    relative_delay_samples: int
    relative_delay_ms: float
    primary_samples: int
    reference_samples: int
    primary_duration_s: float
    reference_duration_s: float
    sample_count_difference: int
    duration_difference_s: float
    primary_estimated_sample_rate: float
    reference_estimated_sample_rate: float
    sample_rate_ratio: float | None
    estimated_drift_samples_per_second: float | None
    analysis_length_samples: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested_sample_rate": self.requested_sample_rate,
            "alignment_note": self.alignment_note,
            "primary_rms": self.primary_rms,
            "reference_rms": self.reference_rms,
            "primary_peak": self.primary_peak,
            "reference_peak": self.reference_peak,
            "correlation": self.correlation,
            "relative_delay_samples": self.relative_delay_samples,
            "relative_delay_ms": self.relative_delay_ms,
            "primary_samples": self.primary_samples,
            "reference_samples": self.reference_samples,
            "primary_duration_s": self.primary_duration_s,
            "reference_duration_s": self.reference_duration_s,
            "sample_count_difference": self.sample_count_difference,
            "duration_difference_s": self.duration_difference_s,
            "primary_estimated_sample_rate": self.primary_estimated_sample_rate,
            "reference_estimated_sample_rate": self.reference_estimated_sample_rate,
            "sample_rate_ratio": self.sample_rate_ratio,
            "estimated_drift_samples_per_second": (
                self.estimated_drift_samples_per_second
            ),
            "analysis_length_samples": self.analysis_length_samples,
        }


@dataclass(frozen=True)
class IndependentCaptureResult:
    """Combined result from independent primary/reference capture."""

    config: IndependentMicConfig
    primary: DeviceStreamCapture
    reference: DeviceStreamCapture
    analysis: IndependentPairAnalysis | None
    capture_elapsed_s: float


def validate_input_device(device: int | str, *, label: str) -> None:
    """Raise ``ValueError`` when a device cannot capture input."""

    channels = _device_channel_count(device, "input")

    if channels < 1:
        raise ValueError(f"{label}={device!r} does not support audio input.")


def query_host_sample_rate(device: int | str) -> float:
    sd = _import_sounddevice()
    info = sd.query_devices(device, kind="input")
    return float(info["default_samplerate"])


def _capture_device_stream(
    *,
    device: int | str,
    requested_sample_rate: int,
    duration_s: float,
    blocksize: int,
    stop_event: threading.Event,
) -> DeviceStreamCapture:
    sd = _import_sounddevice()
    host_channels = _device_channel_count(device, "input")
    host_sample_rate = query_host_sample_rate(device)
    capture_channels = 1 if host_channels >= 1 else 1

    stream = sd.InputStream(
        samplerate=requested_sample_rate,
        device=device,
        channels=capture_channels,
        dtype="float32",
        blocksize=blocksize,
        latency="high",
    )

    chunks: list[np.ndarray] = []
    overflows = 0
    start_time_s = time.perf_counter()
    deadline = start_time_s + duration_s

    stream.start()

    try:
        while time.perf_counter() < deadline and not stop_event.is_set():
            available = stream.read_available
            if available <= 0:
                time.sleep(0.001)
                continue

            frames = min(blocksize, available)
            data, overflowed = stream.read(frames)

            if overflowed:
                overflows += 1

            mono = downmix_to_mono(np.asarray(data, dtype=np.float32))

            if mono.size == 0:
                continue

            chunks.append(mono.copy())
    finally:
        end_time_s = time.perf_counter()
        stream.stop()
        stream.close()

    if chunks:
        audio = np.concatenate(chunks).astype(np.float32, copy=False)
    else:
        audio = np.empty(0, dtype=np.float32)

    elapsed_s = end_time_s - start_time_s

    return DeviceStreamCapture(
        device=device,
        audio=audio,
        requested_sample_rate=requested_sample_rate,
        host_sample_rate=host_sample_rate,
        samples_captured=int(audio.shape[0]),
        capture_elapsed_s=elapsed_s,
        start_time_s=start_time_s,
        end_time_s=end_time_s,
        input_overflows=overflows,
    )


def _capture_worker(
    *,
    device: int | str,
    requested_sample_rate: int,
    duration_s: float,
    blocksize: int,
    stop_event: threading.Event,
    results: dict[str, DeviceStreamCapture],
    errors: list[BaseException],
    result_key: str,
) -> None:
    try:
        results[result_key] = _capture_device_stream(
            device=device,
            requested_sample_rate=requested_sample_rate,
            duration_s=duration_s,
            blocksize=blocksize,
            stop_event=stop_event,
        )
    except BaseException as exc:
        errors.append(exc)
        stop_event.set()


def record_independent_microphones(
    config: IndependentMicConfig,
    *,
    on_progress: Callable[[float, float], None] | None = None,
    capture_worker: Callable[..., None] | None = None,
    defer_analysis: bool = False,
) -> IndependentCaptureResult:
    """
    Capture primary and reference microphones from separate input devices.

    Both streams are opened in parallel threads and run for approximately
    ``config.duration_s`` wall-clock seconds. The devices are **not**
    sample-clock locked.
    """

    config.validate()

    stop_event = threading.Event()
    results: dict[str, DeviceStreamCapture] = {}
    errors: list[BaseException] = []

    worker = capture_worker or _capture_worker
    threads = [
        threading.Thread(
            target=worker,
            kwargs={
                "device": config.primary_device,
                "requested_sample_rate": config.sample_rate,
                "duration_s": config.duration_s,
                "blocksize": config.blocksize,
                "stop_event": stop_event,
                "results": results,
                "errors": errors,
                "result_key": "primary",
            },
            daemon=True,
        ),
        threading.Thread(
            target=worker,
            kwargs={
                "device": config.reference_device,
                "requested_sample_rate": config.effective_reference_sample_rate,
                "duration_s": config.duration_s,
                "blocksize": config.blocksize,
                "stop_event": stop_event,
                "results": results,
                "errors": errors,
                "result_key": "reference",
            },
            daemon=True,
        ),
    ]

    capture_start = time.perf_counter()

    for thread in threads:
        thread.start()

    while any(thread.is_alive() for thread in threads):
        elapsed_s = time.perf_counter() - capture_start

        if on_progress is not None:
            on_progress(elapsed_s, config.duration_s)

        for thread in threads:
            thread.join(timeout=0.05)

    capture_elapsed_s = time.perf_counter() - capture_start

    if errors:
        raise RuntimeError(
            "Independent microphone capture failed: "
            + "; ".join(str(error) for error in errors)
        ) from errors[0]

    if "primary" not in results or "reference" not in results:
        raise RuntimeError("Independent microphone capture did not return both streams.")

    primary = results["primary"]
    reference = results["reference"]
    analysis = (
        None
        if defer_analysis
        else analyze_independent_pair(
            primary,
            reference,
            config.primary_sample_rate,
            reference_sample_rate=config.effective_reference_sample_rate,
        )
    )

    return IndependentCaptureResult(
        config=config,
        primary=primary,
        reference=reference,
        analysis=analysis,
        capture_elapsed_s=capture_elapsed_s,
    )


def prepare_independent_pair_for_analysis(
    primary: DeviceStreamCapture | np.ndarray,
    reference: DeviceStreamCapture | np.ndarray,
    analysis_sample_rate: int,
    *,
    reference_sample_rate: int | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Return mono primary/reference arrays at ``analysis_sample_rate`` for comparison.

    When the reference was captured at a different native rate, it is resampled
    to ``analysis_sample_rate`` for delay/correlation estimation only. Raw
    captures are not time-shifted.
    """

    if isinstance(primary, DeviceStreamCapture):
        primary_audio = primary.audio
        primary_native_rate = int(primary.requested_sample_rate)
    else:
        primary_audio = np.asarray(primary, dtype=np.float32).reshape(-1)
        primary_native_rate = analysis_sample_rate

    if isinstance(reference, DeviceStreamCapture):
        reference_audio = reference.audio
        reference_native_rate = int(reference.requested_sample_rate)
    else:
        reference_audio = np.asarray(reference, dtype=np.float32).reshape(-1)
        reference_native_rate = (
            reference_sample_rate
            if reference_sample_rate is not None
            else analysis_sample_rate
        )

    if reference_native_rate != analysis_sample_rate:
        reference_audio = resample_mono(
            reference_audio,
            reference_native_rate,
            analysis_sample_rate,
        )
        resample_note = (
            f"Reference resampled from {reference_native_rate} Hz to "
            f"{analysis_sample_rate} Hz for analysis only."
        )
    else:
        resample_note = ""

    if primary_native_rate != analysis_sample_rate:
        primary_audio = resample_mono(
            primary_audio,
            primary_native_rate,
            analysis_sample_rate,
        )

    analysis_length = min(primary_audio.shape[0], reference_audio.shape[0])
    return (
        primary_audio[:analysis_length],
        reference_audio[:analysis_length],
        resample_note,
    )


def analyze_independent_pair(
    primary: DeviceStreamCapture | np.ndarray,
    reference: DeviceStreamCapture | np.ndarray,
    requested_sample_rate: int,
    *,
    reference_sample_rate: int | None = None,
    max_delay_samples: int | None = None,
) -> IndependentPairAnalysis:
    """
    Analyze two independent mono captures.

    Correlation and relative delay are estimated on the overlapping prefix of
    the two signals. When capture rates differ, the reference is resampled to
    ``requested_sample_rate`` for analysis only; no time-shift compensation is
    applied beyond integer lag estimation.
    """

    if isinstance(primary, DeviceStreamCapture):
        primary_audio = primary.audio
        primary_samples = primary.samples_captured
        primary_duration_s = primary.duration_s
        primary_rate = primary.estimated_sample_rate
    else:
        primary_audio = np.asarray(primary, dtype=np.float32).reshape(-1)
        primary_samples = int(primary_audio.shape[0])
        primary_duration_s = (
            primary_samples / requested_sample_rate
            if requested_sample_rate > 0
            else 0.0
        )
        primary_rate = float(requested_sample_rate)

    if isinstance(reference, DeviceStreamCapture):
        reference_audio_raw = reference.audio
        reference_samples = reference.samples_captured
        reference_duration_s = reference.duration_s
        reference_rate = reference.estimated_sample_rate
        effective_reference_rate = int(reference.requested_sample_rate)
    else:
        reference_audio_raw = np.asarray(reference, dtype=np.float32).reshape(-1)
        reference_samples = int(reference_audio_raw.shape[0])
        reference_duration_s = (
            reference_samples / requested_sample_rate
            if requested_sample_rate > 0
            else 0.0
        )
        reference_rate = float(
            reference_sample_rate
            if reference_sample_rate is not None
            else requested_sample_rate
        )
        effective_reference_rate = (
            reference_sample_rate
            if reference_sample_rate is not None
            else requested_sample_rate
        )

    primary_prefix, reference_prefix, resample_note = (
        prepare_independent_pair_for_analysis(
            primary,
            reference,
            requested_sample_rate,
            reference_sample_rate=effective_reference_rate,
        )
    )
    analysis_length = int(primary_prefix.shape[0])

    if analysis_length == 0:
        correlation = 0.0
        delay_samples = 0
    else:
        correlation = compute_correlation(primary_prefix, reference_prefix)
        delay_samples = estimate_relative_delay_samples(
            primary_prefix,
            reference_prefix,
            max_delay_samples=max_delay_samples,
        )

    sample_count_difference = primary_samples - reference_samples
    duration_difference_s = primary_duration_s - reference_duration_s

    shared_elapsed = max(
        primary.capture_elapsed_s
        if isinstance(primary, DeviceStreamCapture)
        else 0.0,
        reference.capture_elapsed_s
        if isinstance(reference, DeviceStreamCapture)
        else 0.0,
    )

    drift_samples_per_second: float | None = None
    if shared_elapsed > 0.0:
        drift_samples_per_second = sample_count_difference / shared_elapsed

    sample_rate_ratio: float | None = None
    if reference_rate > 0.0:
        sample_rate_ratio = primary_rate / reference_rate

    delay_ms = (
        1000.0 * delay_samples / requested_sample_rate
        if requested_sample_rate > 0
        else 0.0
    )

    alignment_note = (
        "Independent input devices are not hardware-synchronized. "
        "Correlation and relative delay are estimated on the first "
        f"{analysis_length} samples at {requested_sample_rate} Hz. "
        "Sample-index alignment across devices is approximate only."
    )
    if resample_note:
        alignment_note = f"{alignment_note} {resample_note}"

    return IndependentPairAnalysis(
        requested_sample_rate=requested_sample_rate,
        alignment_note=alignment_note,
        primary_rms=compute_rms(primary_prefix if analysis_length else primary_audio),
        reference_rms=compute_rms(
            reference_prefix if analysis_length else reference_audio_raw
        ),
        primary_peak=compute_peak(primary_prefix if analysis_length else primary_audio),
        reference_peak=compute_peak(
            reference_prefix if analysis_length else reference_audio_raw
        ),
        correlation=correlation,
        relative_delay_samples=delay_samples,
        relative_delay_ms=delay_ms,
        primary_samples=primary_samples,
        reference_samples=reference_samples,
        primary_duration_s=primary_duration_s,
        reference_duration_s=reference_duration_s,
        sample_count_difference=sample_count_difference,
        duration_difference_s=duration_difference_s,
        primary_estimated_sample_rate=primary_rate,
        reference_estimated_sample_rate=reference_rate,
        sample_rate_ratio=sample_rate_ratio,
        estimated_drift_samples_per_second=drift_samples_per_second,
        analysis_length_samples=analysis_length,
    )
