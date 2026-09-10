"""USB-C + Bluetooth independent dual-microphone experiment (Task 6).

Reuses ``independent_mic`` capture/analysis utilities. Does not modify the
production single-microphone demo or integrate NLMS.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from drdo_anc.audio import save_mono_wav
from drdo_anc.audio.live import format_device_listing, list_audio_devices
from drdo_anc.audio.live.independent_mic import (
    IndependentMicConfig,
    analyze_independent_pair,
    prepare_independent_pair_for_analysis,
    query_host_sample_rate,
    record_independent_microphones,
    validate_input_device,
)
from drdo_anc.audio.live.multimic import (
    compute_correlation,
    compute_peak,
    compute_rms,
    estimate_relative_delay_samples,
)
from drdo_anc.audio.resampling import resample_mono
from drdo_anc.audio.live.sounddevice_backend import _import_sounddevice

DEFAULT_PRIMARY = 21  # WASAPI Headset (EarPods) — USB-C microphone
DEFAULT_REFERENCE = 19  # WASAPI Headset (Boult Audio Airbass) — Bluetooth
DEFAULT_SAMPLE_RATE = 48_000
DEFAULT_REFERENCE_SAMPLE_RATE = 16_000  # Bluetooth WASAPI native rate
DEFAULT_OUTPUT_DIR = Path("data") / "usb_bluetooth_dual_mic_experiment"
MAX_DELAY_SAMPLES = 48_000  # 1 s search window at 48 kHz analysis rate
ANALYSIS_SAMPLE_RATE = 4_000  # downsampled rate for delay/correlation search
ANALYSIS_MAX_DELAY_SAMPLES = 4_000  # 1 s at analysis rate


@dataclass(frozen=True)
class DeviceInfo:
    index: int
    name: str
    host_api: str
    default_sample_rate: float
    input_channels: int


@dataclass(frozen=True)
class CompatibilityResult:
    primary_opens: bool
    reference_opens: bool
    simultaneous_opens: bool
    requested_sample_rate: int
    primary_actual_rate: float | None
    reference_actual_rate: float | None
    primary_error: str | None
    reference_error: str | None
    simultaneous_error: str | None
    notes: list[str]


def _device_info(index: int) -> DeviceInfo:
    sd = _import_sounddevice()
    info = sd.query_devices(index)
    host_api = sd.query_hostapis(info["hostapi"])["name"]
    return DeviceInfo(
        index=index,
        name=str(info["name"]),
        host_api=str(host_api),
        default_sample_rate=float(info["default_samplerate"]),
        input_channels=int(info["max_input_channels"]),
    )


def _find_wasapi_input(name_fragment: str) -> DeviceInfo | None:
    fragment = name_fragment.lower()
    for device in list_audio_devices():
        if device["max_input_channels"] < 1:
            continue
        if "wasapi" not in str(device.get("hostapi_name", "")).lower():
            continue
        if fragment in str(device["name"]).lower():
            return _device_info(int(device["index"]))
    return None


def identify_devices(
    primary_index: int | None,
    reference_index: int | None,
) -> tuple[DeviceInfo, DeviceInfo]:
    if primary_index is not None:
        primary = _device_info(primary_index)
    else:
        primary = _find_wasapi_input("earpods")
        if primary is None:
            raise RuntimeError("Could not find WASAPI USB-C microphone (EarPods).")

    if reference_index is not None:
        reference = _device_info(reference_index)
    else:
        reference = _find_wasapi_input("boult audio airbass")
        if reference is None:
            raise RuntimeError(
                "Could not find WASAPI Bluetooth microphone (Boult Audio Airbass)."
            )

    return primary, reference


def test_device_compatibility(
    primary: DeviceInfo,
    reference: DeviceInfo,
    sample_rate: int,
    *,
    reference_sample_rate: int | None = None,
    test_duration_s: float = 0.5,
) -> CompatibilityResult:
    sd = _import_sounddevice()
    notes: list[str] = []
    reference_rate = (
        reference_sample_rate
        if reference_sample_rate is not None
        else sample_rate
    )
    if reference_rate != sample_rate:
        notes.append(
            f"Per-device capture rates: primary={sample_rate} Hz, "
            f"reference={reference_rate} Hz."
        )

    primary_error: str | None = None
    reference_error: str | None = None
    simultaneous_error: str | None = None
    primary_opens = False
    reference_opens = False
    simultaneous_opens = False
    primary_actual_rate: float | None = None
    reference_actual_rate: float | None = None

    def _probe(
        device: int,
        probe_rate: int,
        label: str,
    ) -> tuple[bool, float | None, str | None]:
        try:
            stream = sd.InputStream(
                samplerate=probe_rate,
                device=device,
                channels=1,
                dtype="float32",
                blocksize=1024,
                latency="high",
            )
            stream.start()
            _data, overflowed = stream.read(1024)
            stream.stop()
            stream.close()
            actual = float(stream.samplerate)
            if overflowed:
                notes.append(
                    f"{label}: overflow on first read at {probe_rate} Hz."
                )
            return True, actual, None
        except Exception as exc:
            return False, None, str(exc)

    primary_opens, primary_actual_rate, primary_error = _probe(
        primary.index, sample_rate, "primary"
    )
    reference_opens, reference_actual_rate, reference_error = _probe(
        reference.index, reference_rate, "reference"
    )

    if primary_opens and reference_opens:
        try:
            primary_stream = sd.InputStream(
                samplerate=sample_rate,
                device=primary.index,
                channels=1,
                dtype="float32",
                blocksize=1024,
                latency="high",
            )
            reference_stream = sd.InputStream(
                samplerate=reference_rate,
                device=reference.index,
                channels=1,
                dtype="float32",
                blocksize=1024,
                latency="high",
            )
            primary_stream.start()
            reference_stream.start()
            deadline = time.perf_counter() + test_duration_s
            overflows = 0
            while time.perf_counter() < deadline:
                _, p_overflow = primary_stream.read(1024)
                _, r_overflow = reference_stream.read(1024)
                if p_overflow or r_overflow:
                    overflows += 1
            primary_stream.stop()
            reference_stream.stop()
            primary_stream.close()
            reference_stream.close()
            simultaneous_opens = True
            if overflows:
                notes.append(
                    f"Simultaneous capture: {overflows} read cycles reported overflow."
                )
        except Exception as exc:
            simultaneous_error = str(exc)
    else:
        simultaneous_error = "Skipped because one or both devices failed solo open."

    if reference.default_sample_rate < sample_rate:
        notes.append(
            "Reference host default sample rate is lower than primary rate; "
            "reference captured at native rate when per-device rates are used."
        )

    return CompatibilityResult(
        primary_opens=primary_opens,
        reference_opens=reference_opens,
        simultaneous_opens=simultaneous_opens,
        requested_sample_rate=sample_rate,
        primary_actual_rate=primary_actual_rate,
        reference_actual_rate=reference_actual_rate,
        primary_error=primary_error,
        reference_error=reference_error,
        simultaneous_error=simultaneous_error,
        notes=notes,
    )


def _segment_delay_ms(
    primary: np.ndarray,
    reference: np.ndarray,
    sample_rate: int,
    start_sample: int,
    end_sample: int,
    *,
    max_delay_samples: int | None = None,
) -> tuple[int, float, float]:
    length = min(end_sample, primary.shape[0], reference.shape[0]) - start_sample
    if length <= 0:
        return 0, 0.0, 0.0

    p = primary[start_sample : start_sample + length]
    r = reference[start_sample : start_sample + length]

    # Limit analysis window length to keep cross-correlation bounded.
    max_analysis_samples = min(length, sample_rate * 5)
    p = p[:max_analysis_samples]
    r = r[:max_analysis_samples]

    delay = estimate_relative_delay_samples(
        p,
        r,
        max_delay_samples=max_delay_samples,
    )
    delay_ms = 1000.0 * delay / sample_rate if sample_rate > 0 else 0.0
    corr = compute_correlation(p, r)
    return delay, delay_ms, corr


def _detect_impulsive_windows(
    signal: np.ndarray,
    sample_rate: int,
    *,
    frame_ms: float = 20.0,
    top_k: int = 8,
) -> list[tuple[int, int]]:
    frame = max(1, int(sample_rate * frame_ms / 1000.0))
    if signal.size < frame * 3:
        return []

    energies = []
    for start in range(0, signal.size - frame, frame):
        chunk = signal[start : start + frame]
        energies.append((float(np.mean(np.square(chunk))), start))

    energies.sort(reverse=True)
    windows: list[tuple[int, int]] = []
    half_window = int(sample_rate * 0.25)

    for _energy, start in energies[: top_k * 3]:
        center = start + frame // 2
        win_start = max(0, center - half_window)
        win_end = min(signal.size, center + half_window)
        candidate = (win_start, win_end)
        if any(abs(candidate[0] - existing[0]) < half_window for existing in windows):
            continue
        windows.append(candidate)
        if len(windows) >= top_k:
            break

    windows.sort(key=lambda item: item[0])
    return windows


def _prepare_analysis_signals(
    primary: np.ndarray,
    reference: np.ndarray,
    sample_rate: int,
    *,
    reference_sample_rate: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    effective_reference_rate = (
        reference_sample_rate
        if reference_sample_rate is not None
        else sample_rate
    )
    primary_aligned = resample_mono(primary, sample_rate, ANALYSIS_SAMPLE_RATE)
    reference_aligned = resample_mono(
        reference,
        effective_reference_rate,
        ANALYSIS_SAMPLE_RATE,
    )
    analysis_length = min(primary_aligned.shape[0], reference_aligned.shape[0])
    return (
        primary_aligned[:analysis_length],
        reference_aligned[:analysis_length],
        ANALYSIS_SAMPLE_RATE,
    )


def analyze_capture_segments(
    primary: np.ndarray,
    reference: np.ndarray,
    sample_rate: int,
    *,
    reference_sample_rate: int | None = None,
    max_delay_samples: int = ANALYSIS_MAX_DELAY_SAMPLES,
) -> dict[str, Any]:
    analysis_primary, analysis_reference, analysis_rate = _prepare_analysis_signals(
        primary,
        reference,
        sample_rate,
        reference_sample_rate=reference_sample_rate,
    )

    preview_samples = min(
        analysis_primary.shape[0],
        int(analysis_rate * 10.0),
    )
    analysis = analyze_independent_pair(
        analysis_primary[:preview_samples],
        analysis_reference[:preview_samples],
        analysis_rate,
        max_delay_samples=max_delay_samples,
    )

    total = int(analysis_primary.shape[0])
    segment_s = 30.0
    segment_samples = int(analysis_rate * segment_s)
    drift_points: list[dict[str, float | int]] = []

    for index, start in enumerate(range(0, total, segment_samples)):
        end = min(total, start + segment_samples)
        delay, delay_ms, corr = _segment_delay_ms(
            analysis_primary,
            analysis_reference,
            analysis_rate,
            start,
            end,
            max_delay_samples=max_delay_samples,
        )
        drift_points.append(
            {
                "segment_index": index,
                "start_s": start / analysis_rate,
                "end_s": end / analysis_rate,
                "relative_delay_samples": delay,
                "relative_delay_ms": delay_ms,
                "correlation": corr,
            }
        )

    event_windows = _detect_impulsive_windows(analysis_primary, analysis_rate)
    event_metrics: list[dict[str, float | int]] = []
    for index, (start, end) in enumerate(event_windows):
        delay, delay_ms, corr = _segment_delay_ms(
            analysis_primary,
            analysis_reference,
            analysis_rate,
            start,
            end,
            max_delay_samples=max_delay_samples,
        )
        event_metrics.append(
            {
                "event_index": index,
                "start_s": start / analysis_rate,
                "end_s": end / analysis_rate,
                "relative_delay_samples": delay,
                "relative_delay_ms": delay_ms,
                "correlation": corr,
            }
        )

    quiet_start = int(total * 0.1)
    quiet_end = int(total * 0.2)
    bg_delay, bg_delay_ms, bg_corr = _segment_delay_ms(
        analysis_primary,
        analysis_reference,
        analysis_rate,
        quiet_start,
        quiet_end,
        max_delay_samples=max_delay_samples,
    )

    initial_delay_ms = drift_points[0]["relative_delay_ms"] if drift_points else 0.0
    final_delay_ms = drift_points[-1]["relative_delay_ms"] if drift_points else 0.0
    delay_change_ms = float(final_delay_ms) - float(initial_delay_ms)
    duration_s = total / analysis_rate if analysis_rate > 0 else 0.0
    drift_rate_ms_per_min = (
        delay_change_ms / (duration_s / 60.0) if duration_s > 0 else 0.0
    )

    event_corrs = [float(item["correlation"]) for item in event_metrics]
    event_corr_mean = float(np.mean(event_corrs)) if event_corrs else 0.0

    return {
        "full_recording": analysis.as_dict(),
        "drift_segments": drift_points,
        "event_windows": event_metrics,
        "event_correlation_mean": event_corr_mean,
        "background_window": {
            "start_s": quiet_start / analysis_rate,
            "end_s": quiet_end / analysis_rate,
            "relative_delay_ms": bg_delay_ms,
            "correlation": bg_corr,
        },
        "initial_relative_delay_ms": float(initial_delay_ms),
        "final_relative_delay_ms": float(final_delay_ms),
        "delay_change_ms": delay_change_ms,
        "drift_rate_ms_per_minute": drift_rate_ms_per_min,
    }


def _interpret_correlation(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 0.7:
        return "strong"
    if abs_value >= 0.4:
        return "moderate"
    if abs_value >= 0.15:
        return "weak"
    return "essentially absent"


def classify_conclusion(
    *,
    compatibility: CompatibilityResult,
    segment_analysis: dict[str, Any],
    stability_runs_ms: list[float],
) -> str:
    if not compatibility.simultaneous_opens:
        return "Category C — Poor reference"

    delay_change = abs(float(segment_analysis["delay_change_ms"]))
    drift_rate = abs(float(segment_analysis["drift_rate_ms_per_minute"]))
    event_corr = abs(float(segment_analysis["event_correlation_mean"]))
    bg_corr = abs(float(segment_analysis["background_window"]["correlation"]))

    if stability_runs_ms:
        spread = max(stability_runs_ms) - min(stability_runs_ms)
    else:
        spread = delay_change

    if event_corr < 0.15 and bg_corr < 0.15:
        return "Category C — Poor reference"

    if spread > 50.0 or drift_rate > 10.0:
        if event_corr >= 0.4:
            return "Category B — Usable with compensation"
        return "Category C — Poor reference"

    if delay_change > 20.0 or abs(float(segment_analysis["initial_relative_delay_ms"])) > 200.0:
        if event_corr >= 0.4:
            return "Category B — Usable with compensation"
        return "Category C — Poor reference"

    if event_corr >= 0.4:
        return "Category A — Promising"

    if event_corr >= 0.15:
        return "Category B — Usable with compensation"

    return "Category C — Poor reference"


def run_capture(
    *,
    primary: DeviceInfo,
    reference: DeviceInfo,
    sample_rate: int,
    reference_sample_rate: int | None,
    duration_s: float,
    output_dir: Path,
    run_label: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = IndependentMicConfig(
        primary_device=primary.index,
        reference_device=reference.index,
        sample_rate=sample_rate,
        reference_sample_rate=reference_sample_rate,
        duration_s=duration_s,
        blocksize=1024,
        condition="impulsive_noise",
    )
    validate_input_device(primary.index, label="primary_device")
    validate_input_device(reference.index, label="reference_device")

    print(f"\n=== Capture: {run_label} ({duration_s:.0f} s) ===")
    print("Perform claps/taps/speech several times during recording.")
    result = record_independent_microphones(
        config,
        defer_analysis=True,
    )

    primary_path = output_dir / f"{run_label}_primary.wav"
    reference_path = output_dir / f"{run_label}_reference.wav"
    save_mono_wav(primary_path, result.primary.audio, sample_rate)
    save_mono_wav(
        reference_path,
        result.reference.audio,
        config.effective_reference_sample_rate,
    )

    segment_analysis = analyze_capture_segments(
        result.primary.audio,
        result.reference.audio,
        sample_rate,
        reference_sample_rate=config.effective_reference_sample_rate,
    )

    metadata = {
        "run_label": run_label,
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "synchronization": "independent_devices",
        "clock_locked": False,
        "primary_device": asdict(primary),
        "reference_device": asdict(reference),
        "requested_sample_rate": sample_rate,
        "reference_sample_rate": config.effective_reference_sample_rate,
        "analysis_sample_rate": ANALYSIS_SAMPLE_RATE,
        "analysis_note": (
            "Delay/correlation/drift metrics are computed on signals downsampled "
            f"to {ANALYSIS_SAMPLE_RATE} Hz for computational efficiency. "
            "Raw WAV captures retain native device sample rates."
        ),
        "requested_duration_s": duration_s,
        "capture_elapsed_s": result.capture_elapsed_s,
        "primary": {
            "samples_captured": result.primary.samples_captured,
            "duration_s": result.primary.duration_s,
            "estimated_sample_rate": result.primary.estimated_sample_rate,
            "host_sample_rate": result.primary.host_sample_rate,
            "rms": compute_rms(result.primary.audio),
            "peak": compute_peak(result.primary.audio),
            "clipping": bool(np.max(np.abs(result.primary.audio)) >= 0.999),
            "input_overflows": result.primary.input_overflows,
            "start_time_s": result.primary.start_time_s,
            "end_time_s": result.primary.end_time_s,
        },
        "reference": {
            "samples_captured": result.reference.samples_captured,
            "duration_s": result.reference.duration_s,
            "estimated_sample_rate": result.reference.estimated_sample_rate,
            "host_sample_rate": result.reference.host_sample_rate,
            "rms": compute_rms(result.reference.audio),
            "peak": compute_peak(result.reference.audio),
            "clipping": bool(np.max(np.abs(result.reference.audio)) >= 0.999),
            "input_overflows": result.reference.input_overflows,
            "start_time_s": result.reference.start_time_s,
            "end_time_s": result.reference.end_time_s,
        },
        "analysis": segment_analysis,
        "files": {
            "primary_wav": primary_path.name,
            "reference_wav": reference_path.name,
        },
    }

    metadata_path = output_dir / f"{run_label}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def render_report(
    *,
    primary: DeviceInfo,
    reference: DeviceInfo,
    compatibility: CompatibilityResult,
    main_capture: dict[str, Any],
    drift_capture: dict[str, Any] | None,
    stability_runs: list[dict[str, Any]],
    conclusion: str,
) -> str:
    main_analysis = main_capture["analysis"]["full_recording"]
    drift_analysis = (
        drift_capture["analysis"] if drift_capture is not None else main_capture["analysis"]
    )
    drift_samples_per_second = main_analysis.get("estimated_drift_samples_per_second")
    drift_text = (
        f"{drift_samples_per_second:.4f} samples/s"
        if drift_samples_per_second is not None
        else "UNKNOWN"
    )

    lines = [
        "Dual Microphone Experiment",
        "==========================",
        "",
        "Setup: independent-device experimental capture (not synchronized hardware).",
        "",
        "Primary (USB-C):",
        f"    device index: {primary.index}",
        f"    name: {primary.name}",
        f"    API: {primary.host_api}",
        f"    default sample rate: {primary.default_sample_rate:.0f} Hz",
        f"    input channels: {primary.input_channels}",
        "",
        "Reference (Bluetooth):",
        f"    device index: {reference.index}",
        f"    name: {reference.name}",
        f"    API: {reference.host_api}",
        f"    default sample rate: {reference.default_sample_rate:.0f} Hz",
        f"    input channels: {reference.input_channels}",
        "",
        "Compatibility:",
        f"    primary opens: {compatibility.primary_opens}",
        f"    reference opens: {compatibility.reference_opens}",
        f"    simultaneous opens: {compatibility.simultaneous_opens}",
        f"    requested sample rate: {compatibility.requested_sample_rate} Hz",
    ]

    for note in compatibility.notes:
        lines.append(f"    note: {note}")

    lines.extend(
        [
            "",
            f"Capture duration (main): {main_capture['requested_duration_s']:.1f} s",
            "",
            "Main recording — per-microphone:",
            f"    primary RMS: {main_capture['primary']['rms']:.6f}",
            f"    primary peak: {main_capture['primary']['peak']:.6f}",
            f"    primary overflows: {main_capture['primary']['input_overflows']}",
            f"    reference RMS: {main_capture['reference']['rms']:.6f}",
            f"    reference peak: {main_capture['reference']['peak']:.6f}",
            f"    reference overflows: {main_capture['reference']['input_overflows']}",
            "",
            "Relative delay (full recording):",
            f"    {main_analysis['relative_delay_samples']} samples",
            f"    {main_analysis['relative_delay_ms']:.2f} ms",
            "",
            "Correlation:",
            f"    full recording: {main_analysis['correlation']:.4f} "
            f"({_interpret_correlation(main_analysis['correlation'])})",
            f"    acoustic events (mean): "
            f"{main_capture['analysis']['event_correlation_mean']:.4f} "
            f"({_interpret_correlation(main_capture['analysis']['event_correlation_mean'])})",
            f"    background window: "
            f"{main_capture['analysis']['background_window']['correlation']:.4f} "
            f"({_interpret_correlation(main_capture['analysis']['background_window']['correlation'])})",
            "",
            "Clock drift (segmented delay over time):",
            f"    initial relative delay: {drift_analysis['initial_relative_delay_ms']:.2f} ms",
            f"    final relative delay: {drift_analysis['final_relative_delay_ms']:.2f} ms",
            f"    delay change: {drift_analysis['delay_change_ms']:.2f} ms",
            f"    drift rate: {drift_analysis['drift_rate_ms_per_minute']:.2f} ms/min",
            f"    sample-count drift: {drift_text}",
            "",
        ]
    )

    if stability_runs:
        lines.append("Synchronization stability (short-run initial delays):")
        for run in stability_runs:
            lines.append(
                f"    {run['run_label']}: "
                f"{run['analysis']['initial_relative_delay_ms']:.2f} ms"
            )
        delays = [
            float(run["analysis"]["initial_relative_delay_ms"]) for run in stability_runs
        ]
        lines.append(f"    spread: {max(delays) - min(delays):.2f} ms")
        lines.append("")

    lines.extend(
        [
            "Conclusion:",
            f"    {conclusion}",
            "",
            "NLMS integration: not tested in this task.",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="USB-C + Bluetooth independent dual-microphone experiment.",
    )
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--primary-device", type=int, default=DEFAULT_PRIMARY)
    parser.add_argument("--reference-device", type=int, default=DEFAULT_REFERENCE)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument(
        "--reference-sample-rate",
        type=int,
        default=DEFAULT_REFERENCE_SAMPLE_RATE,
        help="Reference capture sample rate (Bluetooth WASAPI native rate).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--main-duration",
        type=float,
        default=60.0,
        help="Main capture duration in seconds.",
    )
    parser.add_argument(
        "--drift-duration",
        type=float,
        default=300.0,
        help="Drift capture duration in seconds.",
    )
    parser.add_argument(
        "--skip-drift",
        action="store_true",
        help="Skip the drift capture.",
    )
    parser.add_argument(
        "--skip-stability",
        action="store_true",
        help="Skip two additional short stability captures.",
    )
    args = parser.parse_args()

    if args.list_devices:
        print(format_device_listing())
        return

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = args.output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    primary, reference = identify_devices(args.primary_device, args.reference_device)
    print("Identified devices:")
    print(f"  Primary:   [{primary.index}] {primary.name} ({primary.host_api})")
    print(f"  Reference: [{reference.index}] {reference.name} ({reference.host_api})")

    reference_rate = args.reference_sample_rate
    shared_rate_compat = test_device_compatibility(
        primary,
        reference,
        args.sample_rate,
    )
    per_device_compat = test_device_compatibility(
        primary,
        reference,
        args.sample_rate,
        reference_sample_rate=reference_rate,
    )
    compatibility = CompatibilityResult(
        primary_opens=per_device_compat.primary_opens,
        reference_opens=per_device_compat.reference_opens,
        simultaneous_opens=per_device_compat.simultaneous_opens,
        requested_sample_rate=args.sample_rate,
        primary_actual_rate=per_device_compat.primary_actual_rate,
        reference_actual_rate=per_device_compat.reference_actual_rate,
        primary_error=per_device_compat.primary_error,
        reference_error=per_device_compat.reference_error,
        simultaneous_error=per_device_compat.simultaneous_error,
        notes=[
            *per_device_compat.notes,
            f"Shared-rate ({args.sample_rate} Hz) simultaneous open: "
            f"{shared_rate_compat.simultaneous_opens}",
        ],
    )
    compatibility_path = output_dir / "compatibility.json"
    compatibility_path.write_text(
        json.dumps(asdict(compatibility), indent=2),
        encoding="utf-8",
    )
    print(json.dumps(asdict(compatibility), indent=2))

    if not compatibility.simultaneous_opens:
        report = render_report(
            primary=primary,
            reference=reference,
            compatibility=compatibility,
            main_capture={
                "requested_duration_s": 0.0,
                "primary": {"rms": 0.0, "peak": 0.0, "input_overflows": 0},
                "reference": {"rms": 0.0, "peak": 0.0, "input_overflows": 0},
                "analysis": {
                    "full_recording": {
                        "relative_delay_samples": 0,
                        "relative_delay_ms": 0.0,
                        "correlation": 0.0,
                        "estimated_drift_samples_per_second": 0.0,
                    },
                    "event_correlation_mean": 0.0,
                    "background_window": {"correlation": 0.0},
                    "initial_relative_delay_ms": 0.0,
                    "final_relative_delay_ms": 0.0,
                    "delay_change_ms": 0.0,
                    "drift_rate_ms_per_minute": 0.0,
                },
            },
            drift_capture=None,
            stability_runs=[],
            conclusion="Category C — Poor reference (devices could not capture simultaneously).",
        )
        report_path = output_dir / "report.txt"
        report_path.write_text(report, encoding="utf-8")
        print(report)
        raise SystemExit(1)

    main_capture = run_capture(
        primary=primary,
        reference=reference,
        sample_rate=args.sample_rate,
        reference_sample_rate=reference_rate,
        duration_s=args.main_duration,
        output_dir=output_dir,
        run_label="main_60s",
    )

    drift_capture = None
    if not args.skip_drift:
        drift_capture = run_capture(
            primary=primary,
            reference=reference,
            sample_rate=args.sample_rate,
            reference_sample_rate=reference_rate,
            duration_s=args.drift_duration,
            output_dir=output_dir,
            run_label="drift_5min",
        )

    stability_runs: list[dict[str, Any]] = []
    if not args.skip_stability:
        for index in (1, 2):
            stability_runs.append(
                run_capture(
                    primary=primary,
                    reference=reference,
                    sample_rate=args.sample_rate,
                    reference_sample_rate=reference_rate,
                    duration_s=15.0,
                    output_dir=output_dir,
                    run_label=f"stability_{index}_15s",
                )
            )

    stability_delays = [
        float(run["analysis"]["initial_relative_delay_ms"]) for run in stability_runs
    ]
    drift_source = drift_capture or main_capture
    conclusion = classify_conclusion(
        compatibility=compatibility,
        segment_analysis=drift_source["analysis"],
        stability_runs_ms=stability_delays,
    )

    report = render_report(
        primary=primary,
        reference=reference,
        compatibility=compatibility,
        main_capture=main_capture,
        drift_capture=drift_capture,
        stability_runs=stability_runs,
        conclusion=conclusion,
    )
    report_path = output_dir / "report.txt"
    report_path.write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\nWrote report to {report_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
