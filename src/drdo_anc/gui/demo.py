from __future__ import annotations

import json
import threading
import time
import traceback
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from drdo_anc.audio.io import load_mono_wav
from drdo_anc.audio.live.fake import FakeAudioOutput
from drdo_anc.audio.live.interfaces import AudioInput, AudioOutput
from drdo_anc.audio.live.playback_queue import (
    ABQueuedPlaybackOutput,
    DEFAULT_MAX_CHUNKS,
    QueuedPlaybackOutput,
)
from drdo_anc.audio.live.pipeline import StreamingPipeline
from drdo_anc.enhancement.base import Enhancer
from drdo_anc.gui.demo_manifest import (
    DemoManifestError,
    DemoScenario,
    ValidatedDemoCatalog,
    get_scenario_by_index,
    load_validated_demo_catalog,
    project_root,
)

DEFAULT_CHUNK_SIZE = 1024
DEFAULT_PLAYBACK_QUEUE_CHUNKS = DEFAULT_MAX_CHUNKS


class _AudioOutputFactory(Protocol):
    def __call__(
        self,
        sample_rate: int,
        *,
        output_device: int | str | None,
        blocksize: int,
    ) -> AudioOutput: ...


def _default_open_output(
    sample_rate: int,
    *,
    output_device: int | str | None,
    blocksize: int,
) -> AudioOutput:
    from drdo_anc.audio.live import open_sounddevice_output

    return open_sounddevice_output(
        sample_rate,
        output_device=output_device,
        blocksize=blocksize,
    )

# Deterministic impulse positions used only by unit tests.
_IMPULSE_OFFSETS = (
    12_000,
    48_000,
    96_000,
    144_000,
    210_000,
    288_000,
    360_000,
    420_000,
)


def load_demo_scenarios(
    manifest_path: Path | None = None,
) -> tuple[int, list[DemoScenario]]:
    catalog = load_validated_demo_catalog(manifest_path)
    return catalog.sample_rate, list(catalog.scenarios)


def load_scenario_audio(scenario: DemoScenario) -> tuple[np.ndarray, int]:
    audio, sample_rate = load_mono_wav(scenario.wav_path)
    return audio.astype(np.float32, copy=False), sample_rate


def apply_impulsive_overlay(audio: np.ndarray) -> np.ndarray:
    """Add deterministic sparse impulses to clean speech (test helper only)."""

    mixed = np.asarray(audio, dtype=np.float32).copy()
    window = np.hanning(160).astype(np.float32)

    for offset in _IMPULSE_OFFSETS:
        if offset >= len(mixed):
            break

        end = min(offset + len(window), len(mixed))
        width = end - offset
        mixed[offset:end] += 0.85 * window[:width]

    return np.clip(mixed, -1.0, 1.0)


class ReplayAudioInput(AudioInput):
    """Replay mono audio through the streaming pipeline with transport controls."""

    def __init__(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        realtime: bool = True,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")

        self._audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        self._sample_rate = sample_rate
        self._realtime = realtime
        self._position = 0
        self._last_chunk_start = 0
        self._closed = False
        self._paused = True
        self._stop_requested = False
        self._lock = threading.Lock()
        self._play_event = threading.Event()

    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def duration_samples(self) -> int:
        return int(len(self._audio))

    @property
    def position_samples(self) -> int:
        return self._position

    @property
    def last_chunk_start(self) -> int:
        return self._last_chunk_start

    def play(self) -> None:
        with self._lock:
            self._paused = False
            self._stop_requested = False
        self._play_event.set()

    def pause(self) -> None:
        with self._lock:
            self._paused = True
        self._play_event.clear()

    def stop(self) -> None:
        with self._lock:
            self._paused = True
            self._stop_requested = True
            self._position = 0
        self._play_event.clear()

    def seek_start(self) -> None:
        with self._lock:
            self._position = 0

    def close(self) -> None:
        self._closed = True
        self._play_event.set()

    def read(self, max_samples: int) -> np.ndarray:
        if self._closed:
            raise RuntimeError("AudioInput is closed.")

        if max_samples <= 0:
            return np.empty(0, dtype=np.float32)

        while True:
            with self._lock:
                if self._closed:
                    raise RuntimeError("AudioInput is closed.")

                if self._stop_requested and self._position == 0:
                    return np.empty(0, dtype=np.float32)

                if self._paused:
                    waiting = True
                else:
                    waiting = False
                    if self._position >= len(self._audio):
                        return np.empty(0, dtype=np.float32)

                    end = min(self._position + max_samples, len(self._audio))
                    self._last_chunk_start = self._position
                    chunk = self._audio[self._position : end].copy()
                    self._position = end

            if waiting:
                self._play_event.wait(timeout=0.05)
                continue

            if self._realtime and chunk.size > 0:
                time.sleep(chunk.size / self._sample_rate)

            return chunk


class SelectableAudioOutput(AudioOutput):
    """Route either raw input or enhanced output to a downstream sink."""

    def __init__(
        self,
        sink: AudioOutput,
        *,
        reference_for_playback: bool = False,
    ) -> None:
        self._sink = sink
        self._mode = "raw"
        self._reference_for_playback = reference_for_playback
        self._last_raw = np.empty(0, dtype=np.float32)
        self._last_enhanced = np.empty(0, dtype=np.float32)
        self._reference_enhanced = np.empty(0, dtype=np.float32)
        self._closed = False
        self._lock = threading.Lock()

    def sample_rate(self) -> int:
        return self._sink.sample_rate()

    def set_mode(self, mode: str) -> None:
        if mode not in {"raw", "enhanced"}:
            raise ValueError("mode must be 'raw' or 'enhanced'.")

        with self._lock:
            self._mode = mode

    @property
    def mode(self) -> str:
        return self._mode

    def select_playback_chunk(
        self,
        raw: np.ndarray,
        enhanced: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        with self._lock:
            if self._mode == "raw" and raw.size > 0:
                return raw

            if (
                self._mode == "enhanced"
                and self._reference_for_playback
                and reference.size > 0
            ):
                return reference

            return enhanced

    def prepare_raw(self, raw: np.ndarray) -> None:
        self._last_raw = np.asarray(raw, dtype=np.float32).reshape(-1)

    def prepare_reference(self, reference: np.ndarray) -> None:
        self._reference_enhanced = np.asarray(
            reference,
            dtype=np.float32,
        ).reshape(-1)

    def bind_chunk(self, raw: np.ndarray, enhanced: np.ndarray) -> None:
        with self._lock:
            self._last_raw = np.asarray(raw, dtype=np.float32).reshape(-1)
            self._last_enhanced = np.asarray(enhanced, dtype=np.float32).reshape(-1)

    def write(self, audio: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("AudioOutput is closed.")

        enhanced = np.asarray(audio, dtype=np.float32).reshape(-1)
        with self._lock:
            self._last_enhanced = enhanced
            raw = (
                self._last_raw.copy()
                if self._last_raw.size > 0
                else np.empty(0, dtype=np.float32)
            )
            reference = (
                self._reference_enhanced.copy()
                if self._reference_enhanced.size > 0
                else np.empty(0, dtype=np.float32)
            )

        enqueue_ab = getattr(self._sink, "enqueue_ab", None)
        if enqueue_ab is not None:
            enqueue_ab(raw, enhanced, reference)
            return

        payload = self.select_playback_chunk(raw, enhanced, reference)

        if payload.size > 0:
            self._sink.write(payload)

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        self._sink.close()


class DemoPipelineOutput(SelectableAudioOutput):
    """Selectable output; optional offline reference for precomputed B mode."""

    def __init__(
        self,
        sink: AudioOutput,
        replay_input: ReplayAudioInput,
        reference_audio: np.ndarray | None = None,
        *,
        reference_for_playback: bool = False,
    ) -> None:
        super().__init__(
            sink,
            reference_for_playback=reference_for_playback,
        )
        self._replay_input = replay_input
        self._reference_audio = (
            np.asarray(reference_audio, dtype=np.float32).reshape(-1)
            if reference_audio is not None
            else None
        )

    def prepare_raw(self, raw: np.ndarray) -> None:
        super().prepare_raw(raw)

        if (
            not self._reference_for_playback
            or self._reference_audio is None
            or len(raw) == 0
        ):
            self.prepare_reference(np.empty(0, dtype=np.float32))
            return

        start = self._replay_input.last_chunk_start
        end = start + len(raw)
        reference = self._reference_audio[start:end]

        if len(reference) == len(raw):
            self.prepare_reference(reference)
        else:
            self.prepare_reference(np.empty(0, dtype=np.float32))


class DemoAudioController:
    """Runs recorded audio through the live streaming pipeline for presentation."""

    def __init__(
        self,
        bridge,
        *,
        model_name: str = "DeepFilterNet3",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        output_device: int | str | None = None,
        physical_output: bool = True,
        open_output: _AudioOutputFactory | None = None,
        on_finished: Callable[[], None] | None = None,
    ) -> None:
        self._bridge = bridge
        self._model_name = model_name
        self._chunk_size = chunk_size
        self._output_device = output_device
        self._physical_output = physical_output
        self._open_output = open_output or _default_open_output
        self._on_finished = on_finished

        self._catalog = load_validated_demo_catalog()
        self._scenario_index = 0
        self._ab_mode = "raw"

        self._enhancer: Enhancer | None = None
        self._pipeline: StreamingPipeline | None = None
        self._replay_input: ReplayAudioInput | None = None
        self._selectable_output: SelectableAudioOutput | None = None
        self._sink: AudioOutput | None = None
        self._playback_queue: QueuedPlaybackOutput | ABQueuedPlaybackOutput | None = None
        self._hardware_sink: AudioOutput | None = None
        self._reference_enhanced_audio = None
        self._clean_reference_audio = None
        self._thread: threading.Thread | None = None
        self._session_lock = threading.Lock()
        self._running = False

    @property
    def scenarios(self) -> list[DemoScenario]:
        return list(self._catalog.scenarios)

    @property
    def catalog(self) -> ValidatedDemoCatalog:
        return self._catalog

    def _current_scenario(self) -> DemoScenario:
        return get_scenario_by_index(self._catalog, self._scenario_index)

    def _load_current_audio(self) -> tuple[np.ndarray, int]:
        return load_scenario_audio(self._current_scenario())

    def _ensure_enhancer(self) -> Enhancer:
        if self._enhancer is None:
            from drdo_anc.enhancement import create_enhancer

            self._enhancer = create_enhancer(self._model_name)

        return self._enhancer

    def _build_pipeline(self) -> None:
        enhancer = self._ensure_enhancer()
        scenario = self._current_scenario()
        audio, sample_rate = self._load_current_audio()

        if sample_rate != enhancer.sample_rate():
            raise ValueError(
                f"Demo audio sample rate ({sample_rate} Hz) does not match "
                f"enhancer ({enhancer.sample_rate()} Hz)."
            )

        self._replay_input = ReplayAudioInput(
            audio,
            sample_rate,
            realtime=not self._physical_output,
        )

        if self._physical_output:
            self._hardware_sink = self._open_output(
                sample_rate,
                output_device=self._output_device,
                blocksize=self._chunk_size,
            )
            self._playback_queue = ABQueuedPlaybackOutput(
                self._hardware_sink,
                sample_rate=sample_rate,
                chunk_samples=self._chunk_size,
                max_chunks=DEFAULT_PLAYBACK_QUEUE_CHUNKS,
            )
            self._sink = self._playback_queue
        else:
            self._hardware_sink = None
            self._playback_queue = None
            self._sink = FakeAudioOutput(sample_rate)

        reference_audio = None
        reference_for_playback = scenario.enhanced_playback == "reference"
        if scenario.enhanced_wav_path is not None and reference_for_playback:
            reference_audio, ref_rate = load_mono_wav(scenario.enhanced_wav_path)
            if ref_rate != sample_rate:
                raise DemoManifestError(
                    f"Reference enhanced WAV sample rate mismatch for "
                    f"scenario '{scenario.id}': {scenario.enhanced_wav_path}"
                )
            if len(reference_audio) != len(audio):
                raise DemoManifestError(
                    f"Reference enhanced WAV length mismatch for "
                    f"scenario '{scenario.id}': {scenario.enhanced_wav_path}"
                )

        if scenario.clean_reference_path is not None:
            clean_audio, clean_rate = load_mono_wav(scenario.clean_reference_path)
            if clean_rate != sample_rate:
                raise DemoManifestError(
                    f"Clean reference sample rate mismatch for "
                    f"scenario '{scenario.id}': {scenario.clean_reference_path}"
                )
            if len(clean_audio) != len(audio):
                raise DemoManifestError(
                    f"Clean reference length mismatch for "
                    f"scenario '{scenario.id}': {scenario.clean_reference_path}"
                )
            self._clean_reference_audio = clean_audio
        else:
            self._clean_reference_audio = None

        self._reference_enhanced_audio = reference_audio
        self._selectable_output = DemoPipelineOutput(
            self._sink,
            self._replay_input,
            reference_audio,
            reference_for_playback=reference_for_playback,
        )
        if self._playback_queue is not None:
            bind = getattr(self._playback_queue, "bind_selectable", None)
            if bind is not None:
                bind(self._selectable_output)

        self._selectable_output.set_mode(self._ab_mode)

        self._bridge.set_stream_metadata(
            model_name=self._model_name,
            sample_rate=sample_rate,
        )
        self._bridge.set_demo_scenario(scenario.label)
        self._bridge.clear_error()

        def on_telemetry(in_chunk, out_chunk, proc_time):
            if self._selectable_output is not None:
                self._selectable_output.bind_chunk(in_chunk, out_chunk)

            stats = {"input_overflows": 0, "output_underflows": 0}
            if self._playback_queue is not None:
                stats["playback_queue"] = (
                    self._playback_queue.timing_stats.as_dict()
                )

            self._bridge.publish_data(
                in_chunk,
                out_chunk,
                proc_time,
                stats=stats,
            )
            self._bridge.set_pipeline_stage("df3")

        self._pipeline = StreamingPipeline(
            self._replay_input,
            self._selectable_output,
            enhancer,
            read_chunk_size=self._chunk_size,
            telemetry_callback=on_telemetry,
            instrumentation=True,
        )

    def set_scenario_index(self, index: int) -> None:
        scenario = get_scenario_by_index(self._catalog, index)
        self.stop()
        self._scenario_index = index
        self._bridge.set_demo_scenario(scenario.label)
        self._bridge.set_selected_scenario_index(index)
        self._bridge.clear_error()

    def set_ab_mode(self, mode: str) -> None:
        if mode not in {"raw", "enhanced"}:
            raise ValueError("mode must be 'raw' or 'enhanced'.")

        self._ab_mode = mode

        if self._selectable_output is not None:
            self._selectable_output.set_mode(mode)

        self._bridge.set_ab_mode(mode)

    def play(self) -> None:
        with self._session_lock:
            if self._thread is not None and self._thread.is_alive():
                if self._replay_input is not None:
                    self._bridge.set_pipeline_stage("stream")
                    self._bridge.set_audio_status("Playing")
                    self._bridge.set_playback_state("playing")
                    self._replay_input.play()
                return

            try:
                self._build_pipeline()
            except DemoManifestError as exc:
                self._bridge.set_error(f"Demo asset invalid: {exc}")
                self._bridge.set_audio_status("Error")
                traceback.print_exc()
                return
            except Exception as exc:
                self._bridge.set_error(f"Demo startup failed: {exc}")
                self._bridge.set_audio_status("Error")
                traceback.print_exc()
                return

            assert self._pipeline is not None
            assert self._replay_input is not None

            pipeline = self._pipeline
            replay_input = self._replay_input

            def run() -> None:
                self._running = True
                self._bridge.set_pipeline_stage("capture")
                self._bridge.set_audio_status("Playing")
                self._bridge.set_playback_state("playing")
                replay_input.play()

                try:
                    pipeline.run()
                except Exception as exc:
                    self._bridge.set_error(f"Demo pipeline error: {exc}")
                    self._bridge.set_audio_status("Error")
                    traceback.print_exc()
                finally:
                    self._running = False
                    replay_input.pause()
                    self._bridge.set_playback_state("stopped")
                    self._bridge.set_audio_status("Stopped")
                    self._bridge.set_pipeline_stage("input")
                    if self._on_finished is not None:
                        self._on_finished()

            self._thread = threading.Thread(
                target=run,
                name="drdo-anc-demo-audio",
                daemon=False,
            )
            self._thread.start()

    def pause(self) -> None:
        if self._replay_input is not None:
            self._replay_input.pause()
            self._bridge.set_playback_state("paused")
            self._bridge.set_audio_status("Paused")
            self._bridge.set_pipeline_stage("stream")

    def stop(self) -> None:
        if self._pipeline is not None:
            self._pipeline.request_stop()

        if self._replay_input is not None:
            self._replay_input.stop()

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)

        if self._sink is not None:
            try:
                self._sink.close()
            except Exception:
                traceback.print_exc()

        if self._enhancer is not None:
            self._enhancer.reset()

        self._pipeline = None
        self._replay_input = None
        self._selectable_output = None
        self._playback_queue = None
        self._hardware_sink = None
        self._sink = None
        self._reference_enhanced_audio = None
        self._clean_reference_audio = None
        self._thread = None
        self._running = False
        self._bridge.set_playback_state("stopped")
        self._bridge.set_audio_status("Stopped")
        self._bridge.set_pipeline_stage("input")

    def shutdown(self) -> None:
        self.stop()
        self._enhancer = None


def load_benchmark_summary() -> tuple[int | None, int | None]:
    report_path = (
        project_root()
        / "data"
        / "benchmark_results"
        / "df3_manifest_benchmark_full.json"
    )

    if not report_path.is_file():
        return None, None

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    cases = payload.get("successful_cases")

    if cases is None:
        return None, None

    # 60 development cases x 2 modes = 120 evaluations when report is complete.
    development_cases = int(cases) // 2 if cases else None
    return development_cases, int(cases)
