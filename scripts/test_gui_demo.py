#!/usr/bin/env python3
"""Demo mode streaming tests (no Qt or microphone required)."""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

src_dir = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_dir))

from drdo_anc.enhancement.base import Enhancer
from drdo_anc.gui.demo import (
    DemoAudioController,
    DemoPipelineOutput,
    ReplayAudioInput,
    SelectableAudioOutput,
    apply_impulsive_overlay,
    load_demo_scenarios,
    load_scenario_audio,
)
from drdo_anc.gui.demo_manifest import (
    DemoManifestError,
    compute_demo_reference_metrics,
    get_scenario_by_index,
    load_validated_demo_catalog,
)
from drdo_anc.audio.live.fake import FakeAudioOutput
from drdo_anc.audio.live.playback_queue import ABQueuedPlaybackOutput, QueuedPlaybackOutput
from drdo_anc.audio.live.pipeline import StreamingPipeline
import torch


class _IdentityStreamingEnhancer(Enhancer):
    def __init__(self, sample_rate: int = 48_000, scale: float = 0.75) -> None:
        self._sample_rate = sample_rate
        self._scale = scale
        self.flush_calls = 0

    def load(self) -> None:
        return None

    def reset(self) -> None:
        return None

    def sample_rate(self) -> int:
        return self._sample_rate

    def name(self) -> str:
        return "IdentityStreamingEnhancer"

    def process(self, audio: torch.Tensor) -> torch.Tensor:
        mono = audio.squeeze(0) if audio.ndim == 2 else audio
        return mono.unsqueeze(0)

    def process_stream(self, audio_chunk: torch.Tensor) -> torch.Tensor:
        mono = audio_chunk.squeeze(0) if audio_chunk.ndim == 2 else audio_chunk
        return (mono.float() * self._scale).unsqueeze(0)

    def flush(self) -> torch.Tensor:
        self.flush_calls += 1
        return torch.zeros(1, 0)


class _RecordingBridge:
    def __init__(self) -> None:
        self.snapshots: list[tuple[np.ndarray, np.ndarray, float]] = []

    def set_stream_metadata(self, **kwargs) -> None:
        return None

    def set_demo_scenario(self, label: str) -> None:
        return None

    def clear_error(self) -> None:
        return None

    def set_error(self, message: str) -> None:
        raise RuntimeError(message)

    def publish_data(
        self,
        input_chunk: np.ndarray,
        output_chunk: np.ndarray,
        proc_time_s: float,
        stats: dict | None = None,
    ) -> None:
        self.snapshots.append(
            (
                input_chunk.copy(),
                output_chunk.copy(),
                proc_time_s,
            )
        )

    def set_pipeline_stage(self, stage: str) -> None:
        return None

    def set_audio_status(self, status: str) -> None:
        return None

    def set_playback_state(self, state: str) -> None:
        return None

    def set_ab_mode(self, mode: str) -> None:
        return None


def test_impulsive_overlay_is_deterministic() -> None:
    audio = np.zeros(200_000, dtype=np.float32)
    audio[5000:15000] = 0.05
    first = apply_impulsive_overlay(audio)
    second = apply_impulsive_overlay(audio)
    np.testing.assert_array_equal(first, second)
    assert first[12_050] > audio[12_050]


def test_replay_audio_input_pause_and_resume() -> None:
    audio = np.arange(4096, dtype=np.float32)
    replay = ReplayAudioInput(audio, 48_000, realtime=False)
    sink = FakeAudioOutput(48_000)
    enhancer = _IdentityStreamingEnhancer()
    pipeline = StreamingPipeline(replay, sink, enhancer, read_chunk_size=1024)

    def run() -> None:
        pipeline.run()

    thread = threading.Thread(target=run, daemon=False)
    thread.start()
    replay.play()
    time.sleep(0.05)
    replay.pause()
    paused_position = replay.position_samples
    assert paused_position > 0
    time.sleep(0.05)
    assert replay.position_samples == paused_position
    replay.play()
    time.sleep(0.05)
    replay.stop()
    pipeline.request_stop()
    thread.join(timeout=5.0)
    assert replay.position_samples == 0


def test_selectable_output_routes_raw_or_enhanced() -> None:
    sink = FakeAudioOutput(48_000)
    output = SelectableAudioOutput(sink)
    raw = np.ones(128, dtype=np.float32) * 0.5
    enhanced = np.ones(128, dtype=np.float32) * 0.1
    output.prepare_raw(raw)
    output.set_mode("raw")
    output.write(enhanced)
    np.testing.assert_allclose(sink.all_written(), raw)
    output.set_mode("enhanced")
    output.write(enhanced)
    written = sink.all_written()
    np.testing.assert_allclose(written[128:], enhanced)


def test_demo_scenarios_use_project_training_assets() -> None:
    _, scenarios = load_demo_scenarios()
    assert len(scenarios) == 1
    scenario = scenarios[0]
    assert scenario.wav_path.name == "train_noisy_snr5.wav"
    assert scenario.clean_reference_path is not None
    assert scenario.clean_reference_path.name == "train_clean_snr5.wav"
    assert scenario.enhanced_wav_path is not None
    assert scenario.enhanced_wav_path.name == "train_enh_snr5.wav"
    assert scenario.enhanced_playback == "live"
    assert scenario.target_snr_db == 5.0
    assert scenario.wav_path.is_file()
    assert scenario.clean_reference_path.is_file()
    assert scenario.enhanced_wav_path.is_file()


def test_demo_reference_metrics_use_train_triplet() -> None:
    catalog = load_validated_demo_catalog()
    scenario = catalog.scenarios[0]
    metrics = compute_demo_reference_metrics(scenario)
    assert metrics["noisy_snr"] > 4.0
    assert metrics["enhanced_snr"] > metrics["noisy_snr"]


def test_enhanced_mode_uses_live_output_not_offline_reference() -> None:
    sink = FakeAudioOutput(48_000)
    replay = ReplayAudioInput(np.zeros(1024, dtype=np.float32), 48_000, realtime=False)
    reference = np.ones(128, dtype=np.float32) * 0.9
    output = DemoPipelineOutput(
        sink,
        replay,
        reference,
        reference_for_playback=False,
    )
    raw = np.ones(128, dtype=np.float32) * 0.5
    live = np.ones(128, dtype=np.float32) * 0.1
    output.prepare_raw(raw)
    output.set_mode("enhanced")
    output.write(live)
    np.testing.assert_allclose(sink.all_written(), live)


def test_clean_reference_is_not_routed_to_ab_output() -> None:
    sink = FakeAudioOutput(48_000)
    output = SelectableAudioOutput(sink)
    clean = np.ones(128, dtype=np.float32) * 0.8
    noisy = np.ones(128, dtype=np.float32) * 0.5
    enhanced = np.ones(128, dtype=np.float32) * 0.1
    output.prepare_raw(noisy)
    output.prepare_reference(clean)
    output.set_mode("raw")
    output.write(enhanced)
    np.testing.assert_allclose(sink.all_written(), noisy)
    output.set_mode("enhanced")
    output.write(enhanced)
    written = sink.all_written()
    np.testing.assert_allclose(written[128:], enhanced)


def test_demo_pipeline_uses_process_stream_and_flush() -> None:
    _, scenarios = load_demo_scenarios()
    scenario = scenarios[0]
    audio, sample_rate = load_scenario_audio(scenario)

    replay = ReplayAudioInput(audio[:4096], sample_rate, realtime=False)
    sink = FakeAudioOutput(sample_rate)
    selectable = SelectableAudioOutput(sink)
    enhancer = _IdentityStreamingEnhancer(sample_rate=sample_rate)
    bridge = _RecordingBridge()

    def on_telemetry(in_chunk, out_chunk, proc_time):
        selectable.bind_chunk(in_chunk, out_chunk)
        bridge.publish_data(in_chunk, out_chunk, proc_time)

    pipeline = StreamingPipeline(
        replay,
        selectable,
        enhancer,
        read_chunk_size=1024,
        telemetry_callback=on_telemetry,
    )

    thread = threading.Thread(target=pipeline.run, daemon=False)
    thread.start()
    replay.play()
    thread.join(timeout=10.0)

    assert len(bridge.snapshots) >= 3
    assert sink.all_written().size > 0
    assert enhancer.flush_calls == 1


def test_demo_pipeline_is_deterministic() -> None:
    _, scenarios = load_demo_scenarios()
    scenario = scenarios[0]
    audio, sample_rate = load_scenario_audio(scenario)
    clip = audio[:8192]

    def run_once() -> np.ndarray:
        replay = ReplayAudioInput(clip, sample_rate, realtime=False)
        sink = FakeAudioOutput(sample_rate)
        enhancer = _IdentityStreamingEnhancer(sample_rate=sample_rate)
        pipeline = StreamingPipeline(
            replay,
            sink,
            enhancer,
            read_chunk_size=1024,
        )
        thread = threading.Thread(target=pipeline.run, daemon=False)
        thread.start()
        replay.play()
        thread.join(timeout=10.0)
        return sink.all_written()

    first = run_once()
    second = run_once()
    np.testing.assert_allclose(first, second, rtol=1e-5, atol=1e-5)


def test_demo_manifest_rejects_missing_asset() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        manifest = Path(tmp) / "demo_scenarios.json"
        manifest.write_text(
            json.dumps(
                {
                    "sample_rate": 48000,
                    "scenarios": [
                        {
                            "id": "missing",
                            "label": "Missing",
                            "source": "file",
                            "wav": "data/does_not_exist.wav",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        try:
            load_validated_demo_catalog(manifest)
        except DemoManifestError as exc:
            assert "not found" in str(exc).lower()
        else:
            raise AssertionError("Expected DemoManifestError for missing asset")


def test_demo_scenario_selection_is_deterministic() -> None:
    catalog = load_validated_demo_catalog()
    first = get_scenario_by_index(catalog, 0)
    second = get_scenario_by_index(catalog, 0)
    assert first == second
    assert first.wav_path == second.wav_path

    other = get_scenario_by_index(catalog, 0)
    assert other.wav_path.name == "train_noisy_snr5.wav"


def test_demo_scenario_index_out_of_range_fails() -> None:
    catalog = load_validated_demo_catalog()
    try:
        get_scenario_by_index(catalog, 1)
    except DemoManifestError as exc:
        assert "out of range" in str(exc).lower()
    else:
        raise AssertionError("Expected DemoManifestError for invalid index")


def test_ab_switching_does_not_change_recording() -> None:
    _, scenarios = load_demo_scenarios()
    scenario = scenarios[0]
    audio, sample_rate = load_scenario_audio(scenario)
    clip = audio[:4096]

    replay = ReplayAudioInput(clip, sample_rate, realtime=False)
    sink = FakeAudioOutput(sample_rate)
    output = SelectableAudioOutput(sink)
    enhancer = _IdentityStreamingEnhancer(sample_rate=sample_rate)

    def on_telemetry(in_chunk, out_chunk, proc_time):
        output.bind_chunk(in_chunk, out_chunk)

    pipeline = StreamingPipeline(
        replay,
        output,
        enhancer,
        read_chunk_size=1024,
        telemetry_callback=on_telemetry,
    )

    thread = threading.Thread(target=pipeline.run, daemon=False)
    thread.start()
    replay.play()
    time.sleep(0.02)
    output.set_mode("raw")
    time.sleep(0.02)
    output.set_mode("enhanced")
    time.sleep(0.02)
    replay.stop()
    pipeline.request_stop()
    thread.join(timeout=5.0)

    written = sink.all_written()
    assert written.size > 0


def test_ab_queue_selects_mode_at_playback_time() -> None:
    class _Selector:
        def __init__(self) -> None:
            self.mode = "enhanced"

        def select_playback_chunk(self, raw, enhanced, reference):
            if self.mode == "raw" and raw.size > 0:
                return raw
            return enhanced

    sink = FakeAudioOutput(48_000)
    selector = _Selector()
    queued = ABQueuedPlaybackOutput(
        sink,
        sample_rate=48_000,
        chunk_samples=1024,
        max_chunks=4,
    )
    queued.bind_selectable(selector)

    raw = np.ones(64, dtype=np.float32) * 0.8
    enhanced = np.ones(64, dtype=np.float32) * 0.1
    queued.enqueue_ab(raw, enhanced, np.empty(0, dtype=np.float32))
    time.sleep(0.05)
    selector.mode = "raw"
    queued.enqueue_ab(raw, enhanced, np.empty(0, dtype=np.float32))
    time.sleep(0.1)
    queued.close()

    written = sink.all_written()
    assert written.size == 128
    np.testing.assert_allclose(written[:64], enhanced)
    np.testing.assert_allclose(written[64:], raw)


def test_playback_queue_bounded_and_drains_on_close() -> None:
    sink = FakeAudioOutput(48_000)
    queued = QueuedPlaybackOutput(
        sink,
        sample_rate=48_000,
        chunk_samples=1024,
        max_chunks=3,
    )

    for _ in range(3):
        queued.write(np.ones(512, dtype=np.float32) * 0.25)

    queued.close()
    assert sink.all_written().size == 3 * 512
    assert queued.timing_stats.write_count == 3
    assert queued.timing_stats.queue_high_water <= 3


def test_playback_queue_respects_capacity() -> None:
    sink = FakeAudioOutput(48_000)
    queued = QueuedPlaybackOutput(
        sink,
        sample_rate=48_000,
        chunk_samples=1024,
        max_chunks=2,
    )

    for _ in range(6):
        queued.write(np.ones(128, dtype=np.float32))

    queued.close()
    assert queued.timing_stats.queue_high_water <= 2
    assert sink.all_written().size == 6 * 128


def test_demo_controller_uses_playback_queue_for_physical_output() -> None:
    sinks: list[FakeAudioOutput] = []

    def factory(sample_rate, *, output_device=None, blocksize=1024):
        sink = FakeAudioOutput(sample_rate)
        sinks.append(sink)
        return sink

    bridge = _RecordingBridge()
    controller = DemoAudioController(
        bridge,
        physical_output=True,
        open_output=factory,
    )
    controller._ensure_enhancer = lambda: _IdentityStreamingEnhancer()

    controller.play()
    time.sleep(0.15)
    assert controller._playback_queue is not None
    controller.stop()
    assert controller._playback_queue is None
    assert len(sinks) == 1


def test_demo_controller_opens_output_sink_on_play() -> None:
    created: list[FakeAudioOutput] = []

    def factory(sample_rate, *, output_device=None, blocksize=1024):
        sink = FakeAudioOutput(sample_rate)
        created.append(sink)
        return sink

    bridge = _RecordingBridge()
    controller = DemoAudioController(
        bridge,
        physical_output=True,
        open_output=factory,
    )
    controller._ensure_enhancer = lambda: _IdentityStreamingEnhancer()

    controller.play()
    time.sleep(0.15)
    assert len(created) == 1

    written = created[0].all_written()
    assert written.size > 0
    assert float(np.max(np.abs(written))) > 0.0

    controller.stop()
    assert controller._sink is None


def test_demo_controller_repeated_play_stop_reopens_output() -> None:
    open_count = 0

    def factory(sample_rate, *, output_device=None, blocksize=1024):
        nonlocal open_count
        open_count += 1
        return FakeAudioOutput(sample_rate)

    bridge = _RecordingBridge()
    controller = DemoAudioController(
        bridge,
        physical_output=True,
        open_output=factory,
    )
    controller._ensure_enhancer = lambda: _IdentityStreamingEnhancer()

    for _ in range(3):
        controller.play()
        time.sleep(0.1)
        controller.stop()

    assert open_count == 3


def test_demo_controller_honors_ab_mode_on_pipeline_build() -> None:
    sink = FakeAudioOutput(48_000)

    def factory(sample_rate, *, output_device=None, blocksize=1024):
        return sink

    bridge = _RecordingBridge()
    controller = DemoAudioController(
        bridge,
        physical_output=True,
        open_output=factory,
    )
    controller._ensure_enhancer = lambda: _IdentityStreamingEnhancer(scale=0.25)

    # UI can show Raw while the controller still tracks Enhanced until synced.
    bridge.set_ab_mode("raw")
    controller.set_ab_mode("raw")
    controller.play()
    time.sleep(0.12)
    controller.stop()

    written = sink.all_written()
    assert written.size > 0

    _, scenarios = load_demo_scenarios()
    noisy, _ = load_scenario_audio(scenarios[0])
    n = min(len(written), len(noisy))
    np.testing.assert_allclose(written[:n], noisy[:n], rtol=0.0, atol=1e-6)

    sink2 = FakeAudioOutput(48_000)
    controller2 = DemoAudioController(
        bridge,
        physical_output=True,
        open_output=lambda sample_rate, *, output_device=None, blocksize=1024: sink2,
    )
    controller2._ensure_enhancer = lambda: _IdentityStreamingEnhancer(scale=0.25)
    controller2.set_ab_mode("enhanced")
    controller2.play()
    time.sleep(0.12)
    controller2.stop()

    enhanced_written = sink2.all_written()
    assert enhanced_written.size > 0
    n2 = min(len(enhanced_written), len(noisy))
    np.testing.assert_allclose(
        enhanced_written[:n2],
        noisy[:n2] * 0.25,
        rtol=0.0,
        atol=1e-6,
    )


def test_demo_controller_ab_routes_physical_sink() -> None:
    sink = FakeAudioOutput(48_000)

    def factory(sample_rate, *, output_device=None, blocksize=1024):
        return sink

    bridge = _RecordingBridge()
    controller = DemoAudioController(
        bridge,
        physical_output=True,
        open_output=factory,
    )
    controller._ensure_enhancer = lambda: _IdentityStreamingEnhancer(scale=0.25)

    controller.play()
    time.sleep(0.08)
    controller.set_ab_mode("raw")
    time.sleep(0.05)
    controller.set_ab_mode("enhanced")
    time.sleep(0.05)
    controller.stop()

    written = sink.all_written()
    assert written.size > 0


def main() -> None:
    tests = [
        test_impulsive_overlay_is_deterministic,
        test_replay_audio_input_pause_and_resume,
        test_selectable_output_routes_raw_or_enhanced,
        test_demo_scenarios_use_project_training_assets,
        test_demo_reference_metrics_use_train_triplet,
        test_enhanced_mode_uses_live_output_not_offline_reference,
        test_clean_reference_is_not_routed_to_ab_output,
        test_demo_manifest_rejects_missing_asset,
        test_demo_scenario_selection_is_deterministic,
        test_demo_scenario_index_out_of_range_fails,
        test_ab_switching_does_not_change_recording,
        test_ab_queue_selects_mode_at_playback_time,
        test_playback_queue_bounded_and_drains_on_close,
        test_playback_queue_respects_capacity,
        test_demo_controller_uses_playback_queue_for_physical_output,
        test_demo_pipeline_uses_process_stream_and_flush,
        test_demo_pipeline_is_deterministic,
        test_demo_controller_opens_output_sink_on_play,
        test_demo_controller_repeated_play_stop_reopens_output,
        test_demo_controller_honors_ab_mode_on_pipeline_build,
        test_demo_controller_ab_routes_physical_sink,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print(f"\nAll {len(tests)} demo tests passed.")


if __name__ == "__main__":
    main()
