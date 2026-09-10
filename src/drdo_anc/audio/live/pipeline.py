from __future__ import annotations
import time
from typing import Callable, Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from drdo_anc.enhancement.base import Enhancer

from .interfaces import AudioInput, AudioOutput
from .recorder import LiveInstrumentation, LiveStreamRecorder


class StreamingPipeline:
    """
    Synchronous microphone-to-speaker streaming through an ``Enhancer``.

    The pipeline repeatedly:

    1. Reads an arbitrary-sized chunk from ``AudioInput``
    2. Optionally records the input chunk
    3. Passes it to ``Enhancer.process_stream()`` (or pass-through)
    4. Optionally records the enhanced chunk
    5. Writes any produced output to ``AudioOutput``

    Hardware chunk sizes are unrelated to model frame sizes. Frame
    assembly remains inside the enhancer via ``StreamingBuffer``.

    Shutdown semantics
    ------------------
    * ``run()`` resets the enhancer once at stream start.
    * On normal end-of-input, ``KeyboardInterrupt``, or ``request_stop()``,
      ``run()`` calls ``enhancer.flush()`` **exactly once** before closing
      I/O devices.
    * Pass-through mode (``enhancer=None``) skips enhancement and flush.
    * When a ``LiveStreamRecorder`` is attached, recordings are finalized
      during shutdown after the enhancer flush tail is captured.
    """

    def __init__(
        self,
        audio_input: AudioInput,
        audio_output: AudioOutput,
        enhancer: Enhancer | None = None,
        *,
        read_chunk_size: int = 1024,
        recorder: LiveStreamRecorder | None = None,
        passthrough: bool = False,
        telemetry_callback: Callable[[np.ndarray, np.ndarray, float], None] | None = None,
        instrumentation: bool = False,
    ) -> None:
        if read_chunk_size <= 0:
            raise ValueError("read_chunk_size must be positive.")

        input_rate = audio_input.sample_rate()
        output_rate = audio_output.sample_rate()

        if input_rate != output_rate:
            raise ValueError(
                f"AudioInput sample rate ({input_rate} Hz) does not "
                f"match AudioOutput sample rate ({output_rate} Hz)."
            )

        if enhancer is not None:
            enhancer_rate = enhancer.sample_rate()

            if input_rate != enhancer_rate:
                raise ValueError(
                    f"AudioInput sample rate ({input_rate} Hz) does not "
                    f"match enhancer sample rate ({enhancer_rate} Hz)."
                )

        self._audio_input = audio_input
        self._audio_output = audio_output
        self._enhancer = enhancer
        self._read_chunk_size = read_chunk_size
        self._recorder = recorder
        self._passthrough = passthrough or enhancer is None
        self._instrumentation = (
            LiveInstrumentation()
            if recorder is not None or instrumentation
            else None
        )
        self._telemetry_callback = telemetry_callback
        self._stop_requested = False
        self._flushed = False
        self._shutdown_complete = False

    @property
    def sample_rate(self) -> int:
        return self._audio_input.sample_rate()

    @property
    def read_chunk_size(self) -> int:
        return self._read_chunk_size

    @property
    def instrumentation(self) -> LiveInstrumentation | None:
        return self._instrumentation

    def request_stop(self) -> None:
        """Request graceful shutdown after the current read cycle."""

        self._stop_requested = True

    def run(
        self,
        *,
        diagnose: bool = False,
        diagnose_interval_s: float = 1.0,
        max_chunks: int | None = None,
    ) -> None:
        """
        Process audio until input is exhausted or shutdown is requested.

        ``KeyboardInterrupt`` triggers the same graceful shutdown path.

        When ``diagnose=True``, print ``SoundDeviceStreamStats`` (if
        available on the input device) every ``diagnose_interval_s``.

        When ``max_chunks`` is set, stop after that many successful reads.
        """

        if self._instrumentation is not None:
            self._instrumentation.mark_start()

        if self._enhancer is not None:
            self._enhancer.reset()

        last_report = time.perf_counter()
        chunks_processed = 0

        try:
            while not self._stop_requested:
                chunk = self._audio_input.read(
                    self._read_chunk_size,
                )

                if len(chunk) == 0:
                    break

                self._process_chunk(chunk)
                chunks_processed += 1

                if (
                    max_chunks is not None
                    and chunks_processed >= max_chunks
                ):
                    break

                if diagnose:
                    now = time.perf_counter()

                    if now - last_report >= diagnose_interval_s:
                        self._print_diagnostics()
                        last_report = now
        except KeyboardInterrupt:
            pass
        finally:
            if diagnose:
                self._print_diagnostics()
            self._shutdown()

    def _print_diagnostics(self) -> None:
        import json

        payload: dict = {}

        stats = getattr(self._audio_input, "stats", None)

        if stats is not None:
            payload.update(stats.as_dict())

        host_input = getattr(
            self._audio_input,
            "host_input_channels",
            None,
        )
        host_output = getattr(
            self._audio_output,
            "host_output_channels",
            None,
        )

        if host_input is not None:
            payload["host_input_channels"] = host_input

        if host_output is not None:
            payload["host_output_channels"] = host_output

        if self._instrumentation is not None:
            payload["pipeline"] = self._instrumentation.as_dict(
                sample_rate=self.sample_rate,
            )

        if self._recorder is not None:
            payload["dropped_recording_chunks"] = (
                self._recorder.dropped_chunks
            )

        if payload:
            print(json.dumps(payload, indent=2), flush=True)

    def _process_chunk(self, chunk: np.ndarray) -> None:
        if self._recorder is not None:
            self._recorder.write_input(chunk)

        if self._instrumentation is not None:
            self._instrumentation.add_input_chunk(len(chunk))

        if self._enhancer is None:
            prepare_raw = getattr(self._audio_output, "prepare_raw", None)
            if prepare_raw is not None:
                prepare_raw(chunk)
            self._write_output(chunk, processing_time_s=0.0)
            if self._telemetry_callback is not None:
                self._telemetry_callback(chunk, chunk, 0.0)
            return

        start = time.perf_counter()

        import torch
        output_tensor = self._enhancer.process_stream(
            torch.from_numpy(chunk).float(),
        )

        processing_time_s = time.perf_counter() - start
        
        output_array = _tensor_to_mono_numpy(output_tensor)
        prepare_raw = getattr(self._audio_output, "prepare_raw", None)
        if prepare_raw is not None:
            prepare_raw(chunk)
        self._write_output(
            output_array,
            processing_time_s,
            from_flush=False,
        )
        
        if self._telemetry_callback is not None:
            self._telemetry_callback(chunk, output_array, processing_time_s)

    def _write_tensor(
        self,
        audio: Any,
        processing_time_s: float,
        *,
        from_flush: bool = False,
    ) -> None:
        array = _tensor_to_mono_numpy(audio)
        self._write_output(
            array,
            processing_time_s,
            from_flush=from_flush,
        )

    def _write_output(
        self,
        array: np.ndarray,
        processing_time_s: float,
        *,
        from_flush: bool = False,
    ) -> None:
        if len(array) == 0:
            return

        if self._recorder is not None:
            if from_flush:
                self._recorder.note_flush_enhanced(array)
            else:
                self._recorder.write_enhanced(array)

        if self._instrumentation is not None:
            self._instrumentation.add_enhanced_chunk(
                len(array),
                processing_time_s=processing_time_s,
            )

        self._audio_output.write(array)

    def _shutdown(self) -> None:
        if self._shutdown_complete:
            return

        try:
            if self._enhancer is not None and not self._flushed:
                flush_tensor = self._enhancer.flush()
                self._write_tensor(
                    flush_tensor,
                    processing_time_s=0.0,
                    from_flush=True,
                )
                self._flushed = True
            elif self._enhancer is None and self._recorder is not None:
                pass
        finally:
            if self._instrumentation is not None:
                self._instrumentation.mark_stop()

                stats = getattr(self._audio_input, "stats", None)

                if stats is not None:
                    self._instrumentation.input_overflows = (
                        stats.input_overflows
                    )

            if self._recorder is not None:
                self._recorder.finalize(
                    self._instrumentation,
                    passthrough=self._passthrough,
                )

            self._audio_input.close()
            self._audio_output.close()
            self._shutdown_complete = True


def _tensor_to_mono_numpy(audio: Any) -> np.ndarray:
    array = (
        audio.detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )

    if array.ndim == 2:
        if array.shape[0] != 1:
            raise ValueError("Expected mono audio tensor.")

        array = array.squeeze(0)

    return array
