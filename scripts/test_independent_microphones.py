"""Experimental independent-device microphone capture and tests."""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TextIO

import numpy as np

from drdo_anc.audio import save_mono_wav
from drdo_anc.audio.live import format_device_listing
from drdo_anc.audio.live.capture_ux import (
    COUNTDOWN_SECONDS,
    emit_analysis_complete,
    emit_analysis_start,
    emit_capture_complete,
    emit_capture_failed,
    finish_progress_line,
    run_countdown,
    update_recording_progress,
)
from drdo_anc.audio.live.independent_mic import (
    CAPTURE_CONDITIONS,
    DeviceStreamCapture,
    IndependentCaptureResult,
    IndependentMicConfig,
    analyze_independent_pair,
    record_independent_microphones,
    validate_input_device,
)


DEFAULT_SAMPLE_RATE = 48_000
DEFAULT_OUTPUT_DIR = Path("data") / "independent_mic_recordings"


def _parse_device(value: str | None) -> int | str | None:
    if value is None:
        return None

    try:
        return int(value)
    except ValueError:
        return value


def _synthetic_mono(
    length: int,
    *,
    frequency_hz: float,
    amplitude: float = 0.6,
    phase: float = 0.0,
) -> np.ndarray:
    time_axis = np.arange(length, dtype=np.float32)
    return (
        amplitude
        * np.sin(
            np.float32(2.0 * np.pi * frequency_hz / max(length, 1)) * time_axis
            + np.float32(phase)
        )
    ).astype(np.float32)


def _make_device_capture(
    *,
    device: int | str,
    audio: np.ndarray,
    requested_sample_rate: int,
    duration_s: float,
    elapsed_scale: float = 1.0,
) -> DeviceStreamCapture:
    elapsed_s = duration_s * elapsed_scale

    return DeviceStreamCapture(
        device=device,
        audio=audio.astype(np.float32, copy=False),
        requested_sample_rate=requested_sample_rate,
        host_sample_rate=float(requested_sample_rate),
        samples_captured=int(audio.shape[0]),
        capture_elapsed_s=elapsed_s,
        start_time_s=0.0,
        end_time_s=elapsed_s,
        input_overflows=0,
    )


def test_two_independent_streams_can_be_captured() -> None:
    sample_rate = 1_000
    duration_s = 1.0
    primary_samples = int(sample_rate * duration_s)
    reference_samples = primary_samples - 5

    config = IndependentMicConfig(
        primary_device=1,
        reference_device=2,
        sample_rate=sample_rate,
        duration_s=duration_s,
    )

    def fake_worker(
        *,
        device: int | str,
        requested_sample_rate: int,
        duration_s: float,
        blocksize: int,
        stop_event: object,
        results: dict[str, DeviceStreamCapture],
        errors: list[BaseException],
        result_key: str,
    ) -> None:
        length = (
            primary_samples
            if result_key == "primary"
            else reference_samples
        )
        audio = _synthetic_mono(length, frequency_hz=17.0)
        results[result_key] = _make_device_capture(
            device=device,
            audio=audio,
            requested_sample_rate=requested_sample_rate,
            duration_s=duration_s,
        )

    result = record_independent_microphones(
        config,
        capture_worker=fake_worker,
    )

    assert result.primary.samples_captured == primary_samples
    assert result.reference.samples_captured == reference_samples


def test_unequal_sample_counts_are_detected() -> None:
    primary = _make_device_capture(
        device=1,
        audio=_synthetic_mono(1_000, frequency_hz=5.0),
        requested_sample_rate=1_000,
        duration_s=1.0,
    )
    reference = _make_device_capture(
        device=2,
        audio=_synthetic_mono(992, frequency_hz=5.0),
        requested_sample_rate=1_000,
        duration_s=1.0,
        elapsed_scale=1.01,
    )

    analysis = analyze_independent_pair(primary, reference, 1_000)

    assert analysis.sample_count_difference == 8
    assert analysis.duration_difference_s != 0.0


def test_known_relative_delay_can_be_estimated() -> None:
    sample_rate = 48_000
    length = 4_096
    delay = 20
    time_axis = np.arange(length, dtype=np.float32) / np.float32(sample_rate)
    noise = np.sin(np.float32(2.0 * np.pi * 73.0) * time_axis, dtype=np.float32)
    primary = noise.copy()
    reference = np.zeros(length, dtype=np.float32)
    reference[delay:] = 0.85 * noise[:-delay]

    analysis = analyze_independent_pair(primary, reference, sample_rate)

    assert abs(analysis.relative_delay_samples + delay) <= 2


def test_correlation_is_calculated() -> None:
    signal = _synthetic_mono(2_048, frequency_hz=9.0)
    analysis = analyze_independent_pair(signal, signal.copy(), 48_000)

    assert analysis.correlation > 0.99


def test_drift_is_reported() -> None:
    config = IndependentMicConfig(
        primary_device=1,
        reference_device=2,
        sample_rate=1_000,
        duration_s=2.0,
    )

    def fake_worker(
        *,
        device: int | str,
        requested_sample_rate: int,
        duration_s: float,
        blocksize: int,
        stop_event: object,
        results: dict[str, DeviceStreamCapture],
        errors: list[BaseException],
        result_key: str,
    ) -> None:
        if result_key == "primary":
            audio = _synthetic_mono(2_000, frequency_hz=4.0)
            elapsed_scale = 1.0
        else:
            audio = _synthetic_mono(1_992, frequency_hz=4.0)
            elapsed_scale = 1.0

        results[result_key] = _make_device_capture(
            device=device,
            audio=audio,
            requested_sample_rate=requested_sample_rate,
            duration_s=duration_s,
            elapsed_scale=elapsed_scale,
        )

    result = record_independent_microphones(
        config,
        capture_worker=fake_worker,
    )

    assert result.analysis.sample_count_difference == 8
    assert result.analysis.estimated_drift_samples_per_second is not None
    assert abs(result.analysis.estimated_drift_samples_per_second - 4.0) < 1.0


def test_metadata_identifies_independent_devices() -> None:
    metadata = _build_metadata(
        config=IndependentMicConfig(
            primary_device=3,
            reference_device=7,
            sample_rate=48_000,
            duration_s=5.0,
            condition="speech_only",
        ),
        result=_make_fake_capture_result(),
    )

    assert metadata["synchronization"] == "independent_devices"
    assert metadata["clock_locked"] is False
    assert metadata["condition"] == "speech_only"


def test_requested_duration_is_preserved() -> None:
    duration_s = 2.5
    sample_rate = 1_000
    config = IndependentMicConfig(
        primary_device=1,
        reference_device=2,
        sample_rate=sample_rate,
        duration_s=duration_s,
    )

    def fake_worker(
        *,
        device: int | str,
        requested_sample_rate: int,
        duration_s: float,
        blocksize: int,
        stop_event: object,
        results: dict[str, DeviceStreamCapture],
        errors: list[BaseException],
        result_key: str,
    ) -> None:
        samples = int(requested_sample_rate * duration_s)
        results[result_key] = _make_device_capture(
            device=device,
            audio=_synthetic_mono(samples, frequency_hz=6.0),
            requested_sample_rate=requested_sample_rate,
            duration_s=duration_s,
        )

    result = record_independent_microphones(
        config,
        capture_worker=fake_worker,
    )

    assert result.config.duration_s == duration_s
    assert result.primary.samples_captured == int(sample_rate * duration_s)


def test_no_nan_or_inf() -> None:
    primary = _synthetic_mono(1_024, frequency_hz=11.0)
    reference = _synthetic_mono(1_020, frequency_hz=11.0, amplitude=0.2)

    analysis = analyze_independent_pair(primary, reference, 48_000)

    assert np.isfinite(analysis.correlation)
    assert np.isfinite(analysis.primary_rms)
    assert np.isfinite(analysis.reference_rms)


def test_different_reference_sample_rate_analysis() -> None:
    sample_rate = 48_000
    reference_rate = 16_000
    length_primary = sample_rate
    length_reference = reference_rate
    primary = _synthetic_mono(length_primary, frequency_hz=440.0)
    reference = _synthetic_mono(length_reference, frequency_hz=440.0, amplitude=0.5)

    analysis = analyze_independent_pair(
        primary,
        reference,
        sample_rate,
        reference_sample_rate=reference_rate,
        max_delay_samples=512,
    )

    assert analysis.correlation > 0.9
    assert "Reference resampled" in analysis.alignment_note


def test_capture_failure_is_reported() -> None:
    config = IndependentMicConfig(
        primary_device=1,
        reference_device=2,
        sample_rate=48_000,
        duration_s=1.0,
    )
    output = io.StringIO()

    def failing_record(*_args: object, **_kwargs: object) -> IndependentCaptureResult:
        raise RuntimeError("device open failed")

    try:
        run_hardware_capture(
            config=config,
            output_dir=Path("data/test_independent_mic_fail"),
            sleep=lambda _: None,
            write=lambda message="", **_: output.write(f"{message}\n"),
            progress_stream=io.StringIO(),
            record_fn=failing_record,
        )
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("Expected capture failure to exit with code 1.")

    text = output.getvalue()
    assert "✗ CAPTURE FAILED" in text
    assert "device open failed" in text


def test_countdown_does_not_extend_capture_duration() -> None:
    config = IndependentMicConfig(
        primary_device=1,
        reference_device=2,
        sample_rate=1_000,
        duration_s=10.0,
    )
    sleep_calls: list[float] = []
    recorded_durations: list[float] = []
    output = io.StringIO()

    def fake_record(
        capture_config: IndependentMicConfig,
        **_: object,
    ) -> IndependentCaptureResult:
        recorded_durations.append(capture_config.duration_s)
        return _make_fake_capture_result(config=capture_config)

    run_hardware_capture(
        config=config,
        output_dir=Path("data/test_independent_mic_countdown"),
        sleep=lambda seconds: sleep_calls.append(seconds),
        write=lambda message="", **_: output.write(f"{message}\n"),
        progress_stream=io.StringIO(),
        record_fn=fake_record,
    )

    assert sleep_calls == [1.0, 1.0, 1.0]
    assert recorded_durations == [10.0]
    assert "GET READY" in output.getvalue()


def _make_fake_capture_result(
    *,
    config: IndependentMicConfig | None = None,
) -> IndependentCaptureResult:
    config = config or IndependentMicConfig(
        primary_device=1,
        reference_device=2,
        sample_rate=1_000,
        duration_s=1.0,
    )

    primary = _make_device_capture(
        device=config.primary_device,
        audio=_synthetic_mono(1_000, frequency_hz=5.0),
        requested_sample_rate=config.sample_rate,
        duration_s=config.duration_s,
    )
    reference = _make_device_capture(
        device=config.reference_device,
        audio=_synthetic_mono(992, frequency_hz=5.0),
        requested_sample_rate=config.sample_rate,
        duration_s=config.duration_s,
    )
    analysis = analyze_independent_pair(primary, reference, config.sample_rate)

    return IndependentCaptureResult(
        config=config,
        primary=primary,
        reference=reference,
        analysis=analysis,
        capture_elapsed_s=config.duration_s,
    )


def _build_metadata(
    *,
    config: IndependentMicConfig,
    result: IndependentCaptureResult,
) -> dict[str, object]:
    return {
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "synchronization": "independent_devices",
        "clock_locked": False,
        "primary_device": config.primary_device,
        "reference_device": config.reference_device,
        "requested_sample_rate": config.sample_rate,
        "requested_duration_s": config.duration_s,
        "capture_elapsed_s": result.capture_elapsed_s,
        "condition": config.condition,
        "primary": {
            "device": result.primary.device,
            "requested_sample_rate": result.primary.requested_sample_rate,
            "host_sample_rate": result.primary.host_sample_rate,
            "estimated_sample_rate": result.primary.estimated_sample_rate,
            "samples_captured": result.primary.samples_captured,
            "duration_s": result.primary.duration_s,
            "capture_elapsed_s": result.primary.capture_elapsed_s,
            "input_overflows": result.primary.input_overflows,
        },
        "reference": {
            "device": result.reference.device,
            "requested_sample_rate": result.reference.requested_sample_rate,
            "host_sample_rate": result.reference.host_sample_rate,
            "estimated_sample_rate": result.reference.estimated_sample_rate,
            "samples_captured": result.reference.samples_captured,
            "duration_s": result.reference.duration_s,
            "capture_elapsed_s": result.reference.capture_elapsed_s,
            "input_overflows": result.reference.input_overflows,
        },
        "analysis": result.analysis.as_dict(),
        "alignment_limitation": result.analysis.alignment_note,
        "files": {
            "primary_wav": "primary.wav",
            "reference_wav": "reference.wav",
        },
    }


def run_hardware_capture(
    *,
    config: IndependentMicConfig,
    output_dir: Path,
    sleep: Callable[[float], None] = time.sleep,
    write: Callable[..., None] = print,
    progress_stream: TextIO | None = None,
    record_fn: Callable[..., IndependentCaptureResult] = record_independent_microphones,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    write("=" * 60)
    write("DRDO-ANC | Independent Microphone Experiment")
    write("=" * 60)
    write(f"Primary device:    {config.primary_device}")
    write(f"Reference device:  {config.reference_device}")
    write(f"Sample rate:       {config.sample_rate} Hz")
    write(f"Duration:          {config.duration_s:.1f} s")
    if config.condition is not None:
        write(f"Condition:         {config.condition}")
    write("Synchronization:   independent devices")
    write("=" * 60)

    try:
        validate_input_device(config.primary_device, label="primary_device")
        validate_input_device(config.reference_device, label="reference_device")

        run_countdown(seconds=COUNTDOWN_SECONDS, sleep=sleep, write=write)

        def on_progress(elapsed_s: float, total_s: float) -> None:
            update_recording_progress(
                elapsed_s=elapsed_s,
                total_s=total_s,
                stream=progress_stream,
            )

        result = record_fn(
            config,
            on_progress=on_progress,
            defer_analysis=True,
        )
        finish_progress_line(progress_stream)
        emit_capture_complete(write=write)

        emit_analysis_start(write=write, message="Analyzing signals...")
        if result.analysis is None:
            analysis = analyze_independent_pair(
                result.primary,
                result.reference,
                config.sample_rate,
            )
            result = IndependentCaptureResult(
                config=result.config,
                primary=result.primary,
                reference=result.reference,
                analysis=analysis,
                capture_elapsed_s=result.capture_elapsed_s,
            )
        emit_analysis_complete(write=write)
    except Exception as exc:
        finish_progress_line(progress_stream)
        emit_capture_failed(exc, write=write)
        raise SystemExit(1) from exc

    primary_path = output_dir / "primary.wav"
    reference_path = output_dir / "reference.wav"
    metadata_path = output_dir / "metadata.json"

    save_mono_wav(primary_path, result.primary.audio, config.sample_rate)
    save_mono_wav(reference_path, result.reference.audio, config.sample_rate)

    metadata = _build_metadata(config=config, result=result)
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    write(json.dumps(metadata, indent=2))
    write(f"Wrote {primary_path}")
    write(f"Wrote {reference_path}")
    write(f"Wrote {metadata_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Experimental capture from two independent input devices for "
            "reference-microphone investigation."
        ),
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List host audio devices and exit.",
    )
    parser.add_argument(
        "--capture",
        action="store_true",
        help="Capture from two independent input devices.",
    )
    parser.add_argument(
        "--primary-device",
        required=False,
        help="Primary input device index or name.",
    )
    parser.add_argument(
        "--reference-device",
        required=False,
        help="Reference input device index or name.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Recording duration in seconds.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Requested capture sample rate in Hz.",
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        default=1024,
        help="PortAudio block size hint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for capture mode.",
    )
    parser.add_argument(
        "--condition",
        choices=CAPTURE_CONDITIONS,
        default=None,
        help="Optional experiment condition label stored in metadata.json.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.list_devices:
        print(format_device_listing())
        return

    if args.capture:
        if args.primary_device is None or args.reference_device is None:
            parser.error("--capture requires --primary-device and --reference-device")

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = (
            args.output_dir
            if args.output_dir is not None
            else DEFAULT_OUTPUT_DIR / timestamp
        )

        config = IndependentMicConfig(
            primary_device=_parse_device(args.primary_device),
            reference_device=_parse_device(args.reference_device),
            sample_rate=args.sample_rate,
            duration_s=args.duration,
            blocksize=args.blocksize,
            condition=args.condition,
        )

        run_hardware_capture(
            config=config,
            output_dir=output_dir,
        )
        return

    tests = [
        test_two_independent_streams_can_be_captured,
        test_unequal_sample_counts_are_detected,
        test_known_relative_delay_can_be_estimated,
        test_correlation_is_calculated,
        test_drift_is_reported,
        test_metadata_identifies_independent_devices,
        test_requested_duration_is_preserved,
        test_no_nan_or_inf,
        test_different_reference_sample_rate_analysis,
        test_capture_failure_is_reported,
        test_countdown_does_not_extend_capture_duration,
    ]

    print("=" * 70)
    print("DRDO-ANC | Independent Microphone Tests")
    print("=" * 70)

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
