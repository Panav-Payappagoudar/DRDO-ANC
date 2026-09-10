#!/usr/bin/env python3
"""Measure demo playback write timing (hardware optional)."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

src_dir = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_dir))

from drdo_anc.gui.demo import DemoAudioController


class _SilentBridge:
    def set_stream_metadata(self, **kwargs) -> None:
        return None

    def set_demo_scenario(self, label: str) -> None:
        return None

    def clear_error(self) -> None:
        return None

    def set_error(self, message: str) -> None:
        print(f"ERROR: {message}", file=sys.stderr)

    def publish_data(self, in_chunk, out_chunk, proc_time, stats=None) -> None:
        return None

    def set_pipeline_stage(self, stage: str) -> None:
        return None

    def set_audio_status(self, status: str) -> None:
        return None

    def set_playback_state(self, state: str) -> None:
        return None

    def set_ab_mode(self, mode: str) -> None:
        return None

    def set_selected_scenario_index(self, index: int) -> None:
        return None


def _parse_device(value: str | None) -> int | str | None:
    if value is None:
        return None

    try:
        return int(value)
    except ValueError:
        return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo playback timing report")
    parser.add_argument("--scenario", type=int, default=0)
    parser.add_argument("--output-device", default=None)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--duration-s", type=float, default=12.0)
    args = parser.parse_args()

    bridge = _SilentBridge()
    controller = DemoAudioController(
        bridge,
        chunk_size=args.chunk_size,
        output_device=_parse_device(args.output_device),
        physical_output=True,
    )
    controller.set_scenario_index(args.scenario)
    controller.play()

    def stop_later() -> None:
        time.sleep(args.duration_s)
        controller.stop()

    threading.Thread(target=stop_later, daemon=True).start()

    while controller._thread is not None and controller._thread.is_alive():
        time.sleep(0.2)

    queue = controller._playback_queue
    report = {
        "scenario_index": args.scenario,
        "chunk_size": args.chunk_size,
        "output_device": args.output_device,
        "playback_queue": queue.timing_stats.as_dict() if queue else None,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
