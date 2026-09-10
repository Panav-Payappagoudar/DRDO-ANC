#!/usr/bin/env python3
"""Continuous live microphone soak test with measured diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from drdo_anc.audio.live import (
    StreamingPipeline,
    close_sounddevice_io,
    create_live_recorder,
    format_device_listing,
    open_sounddevice_io,
)
from drdo_anc.enhancement import create_enhancer, get_model_config


DEFAULT_MODEL_NAME = "DeepFilterNet3"
DEFAULT_READ_CHUNK_SIZE = 1024
DEFAULT_DURATION_S = 300.0
DEFAULT_REPORT_DIR = Path("data") / "live_soak_reports"


def _parse_device(value: str | None) -> int | str | None:
    if value is None:
        return None

    try:
        return int(value)
    except ValueError:
        return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run microphone → enhancer → headphones continuously and "
            "record RTF, overflows, and latency estimates."
        ),
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List host audio devices and exit.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Registered enhancer model name.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=DEFAULT_DURATION_S,
        help="Target soak duration in seconds (default: 300).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_READ_CHUNK_SIZE,
        help="Samples requested per AudioInput.read() call.",
    )
    parser.add_argument(
        "--input-device",
        default=None,
        help="Input device index or name.",
    )
    parser.add_argument(
        "--output-device",
        default=None,
        help="Output device index or name.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory for JSON soak report.",
    )
    parser.add_argument(
        "--diagnose-interval-s",
        type=float,
        default=5.0,
        help="Print interim diagnostics every N seconds.",
    )
    return parser


def _estimate_latency_s(
    *,
    chunk_size: int,
    sample_rate: int,
    host_blocksize: int,
) -> dict[str, float]:
    """Approximate live latency from buffering, not evaluation delay."""

    input_buffer_s = chunk_size / sample_rate
    output_buffer_s = (
        host_blocksize / sample_rate if host_blocksize > 0 else input_buffer_s
    )
    duplex_buffer_s = input_buffer_s + output_buffer_s

    return {
        "input_buffer_s": input_buffer_s,
        "output_buffer_s": output_buffer_s,
        "duplex_buffer_s": duplex_buffer_s,
        "note": (
            "Excludes model algorithmic delay and host API extra latency. "
            "Does not use evaluation streaming_delay_samples."
        ),
    }


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.list_devices:
        print(format_device_listing())
        return

    if args.duration_s <= 0:
        raise SystemExit("--duration-s must be positive.")

    model_config = get_model_config(args.model)
    enhancer = create_enhancer(args.model)
    sample_rate = enhancer.sample_rate()
    input_device = _parse_device(args.input_device)
    output_device = _parse_device(args.output_device)

    start_time_utc = datetime.now(timezone.utc).isoformat()
    started_perf = time.perf_counter()

    print("=" * 70)
    print("DRDO-ANC | Live Soak Test")
    print("=" * 70)
    print(f"Model:       {args.model}")
    print(f"Duration:    {args.duration_s:.1f} s")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Chunk size:  {args.chunk_size} samples/read")
    print(f"Input dev:   {input_device if input_device is not None else 'default'}")
    print(f"Output dev:  {output_device if output_device is not None else 'default'}")
    print("\nSpeak normally. Press Ctrl+C to stop early.")
    print("=" * 70)

    audio_input, audio_output = open_sounddevice_io(
        sample_rate,
        input_device=input_device,
        output_device=output_device,
        blocksize=args.chunk_size,
    )

    host_blocksize = getattr(audio_input, "stats", None)
    blocksize = host_blocksize.blocksize if host_blocksize is not None else args.chunk_size

    latency_estimate = _estimate_latency_s(
        chunk_size=args.chunk_size,
        sample_rate=sample_rate,
        host_blocksize=blocksize,
    )

    pipeline = StreamingPipeline(
        audio_input,
        audio_output,
        enhancer,
        read_chunk_size=args.chunk_size,
        instrumentation=True,
    )

    stop_event = threading.Event()

    def stop_after_duration() -> None:
        if stop_event.wait(timeout=args.duration_s):
            return

        print(
            f"\nDuration target reached ({args.duration_s:.1f}s). "
            "Stopping pipeline...",
            flush=True,
        )
        pipeline.request_stop()

    timer = threading.Thread(target=stop_after_duration, daemon=True)
    timer.start()

    error: str | None = None

    try:
        pipeline.run(
            diagnose=True,
            diagnose_interval_s=args.diagnose_interval_s,
        )
    except Exception as exc:
        error = str(exc)
        raise
    finally:
        stop_event.set()
        close_sounddevice_io(audio_input, audio_output)

    elapsed_s = time.perf_counter() - started_perf
    instrumentation = pipeline.instrumentation
    input_stats = getattr(audio_input, "stats", None)

    pipeline_stats = (
        instrumentation.as_dict(sample_rate=sample_rate)
        if instrumentation is not None
        else {}
    )

    input_dict = input_stats.as_dict() if input_stats is not None else {}

    report = {
        "start_time_utc": start_time_utc,
        "elapsed_s": elapsed_s,
        "target_duration_s": args.duration_s,
        "model": args.model,
        "sample_rate": sample_rate,
        "chunk_size": args.chunk_size,
        "input_device": input_device,
        "output_device": output_device,
        "host_input_channels": getattr(
            audio_input,
            "host_input_channels",
            None,
        ),
        "host_output_channels": getattr(
            audio_output,
            "host_output_channels",
            None,
        ),
        "streaming_delay_samples_eval_only": (
            model_config.streaming_delay_samples
        ),
        "latency_estimate_s": latency_estimate,
        "input_stats": input_dict,
        "pipeline_stats": pipeline_stats,
        "input_overflows": input_dict.get("input_overflows"),
        "realtime_ratio": pipeline_stats.get("realtime_ratio"),
        "processing_time_s": pipeline_stats.get("processing_time_s"),
        "error": error,
    }

    args.report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = args.report_dir / f"soak_{stamp}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("Soak test complete")
    print("=" * 70)
    print(json.dumps(report, indent=2))
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
