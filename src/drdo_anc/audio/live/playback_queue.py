from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .interfaces import AudioOutput


class _ABSelectable(Protocol):
    def select_playback_chunk(
        self,
        raw: np.ndarray,
        enhanced: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class _ABPlaybackChunk:
    raw: np.ndarray
    enhanced: np.ndarray
    reference: np.ndarray

DEFAULT_MAX_CHUNKS = 6


@dataclass
class PlaybackTimingStats:
    """Write-side timing for queued physical playback."""

    queue_capacity: int = 0
    expected_interval_s: float = 0.0
    write_count: int = 0
    samples_written: int = 0
    interval_sum_s: float = 0.0
    min_interval_s: float | None = None
    max_interval_s: float | None = None
    late_writes: int = 0
    queue_high_water: int = 0
    _last_enqueue_time: float | None = field(default=None, repr=False)

    def record_enqueue(self, depth: int) -> None:
        self.queue_high_water = max(self.queue_high_water, depth)

    def record_write(self, num_samples: int, interval_s: float) -> None:
        self.write_count += 1
        self.samples_written += num_samples
        self.interval_sum_s += interval_s

        if self.min_interval_s is None or interval_s < self.min_interval_s:
            self.min_interval_s = interval_s

        if self.max_interval_s is None or interval_s > self.max_interval_s:
            self.max_interval_s = interval_s

        if (
            self.expected_interval_s > 0
            and interval_s > self.expected_interval_s * 1.5
        ):
            self.late_writes += 1

    @property
    def average_interval_s(self) -> float | None:
        if self.write_count == 0:
            return None

        return self.interval_sum_s / self.write_count

    def as_dict(self) -> dict:
        return {
            "queue_capacity": self.queue_capacity,
            "queue_high_water": self.queue_high_water,
            "expected_interval_s": self.expected_interval_s,
            "write_count": self.write_count,
            "samples_written": self.samples_written,
            "min_interval_s": self.min_interval_s,
            "max_interval_s": self.max_interval_s,
            "average_interval_s": self.average_interval_s,
            "late_writes": self.late_writes,
        }


class QueuedPlaybackOutput(AudioOutput):
    """
    Decouple enhancer processing from PortAudio playback.

    The producer thread enqueues mono chunks. A dedicated consumer thread
    calls ``sink.write()`` so hardware blocking does not stall DF3 processing.
    """

    def __init__(
        self,
        sink: AudioOutput,
        *,
        sample_rate: int,
        chunk_samples: int = 1024,
        max_chunks: int = DEFAULT_MAX_CHUNKS,
    ) -> None:
        if max_chunks <= 0:
            raise ValueError("max_chunks must be positive.")

        self._sink = sink
        self._sample_rate = sample_rate
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(
            maxsize=max_chunks,
        )
        self._stop = threading.Event()
        self._closed = False
        self._consumer_error: BaseException | None = None
        self._error_lock = threading.Lock()
        self._stats = PlaybackTimingStats(
            queue_capacity=max_chunks,
            expected_interval_s=(
                chunk_samples / sample_rate if sample_rate > 0 else 0.0
            ),
        )
        self._last_write_time: float | None = None
        self._thread = threading.Thread(
            target=self._consume,
            name="drdo-anc-playback-queue",
            daemon=False,
        )
        self._thread.start()

    @property
    def timing_stats(self) -> PlaybackTimingStats:
        return self._stats

    def sample_rate(self) -> int:
        return self._sink.sample_rate()

    def write(self, audio: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("QueuedPlaybackOutput is closed.")

        with self._error_lock:
            if self._consumer_error is not None:
                raise RuntimeError(
                    f"Playback consumer failed: {self._consumer_error}"
                )

        chunk = np.asarray(audio, dtype=np.float32).reshape(-1)

        if chunk.size == 0:
            return

        while not self._closed:
            try:
                self._queue.put(chunk, timeout=0.05)
                self._stats.record_enqueue(self._queue.qsize())
                return
            except queue.Full:
                if self._stop.is_set():
                    return

    def _consume(self) -> None:
        try:
            while True:
                try:
                    item = self._queue.get(timeout=0.05)
                except queue.Empty:
                    if self._stop.is_set():
                        break
                    continue

                if item is None:
                    break

                now = time.perf_counter()
                interval = (
                    now - self._last_write_time
                    if self._last_write_time is not None
                    else 0.0
                )
                self._last_write_time = now
                self._stats.record_write(len(item), interval)
                self._sink.write(item)
        except BaseException as exc:
            with self._error_lock:
                self._consumer_error = exc
            raise

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        self._stop.set()

        while True:
            try:
                self._queue.put(None, timeout=0.05)
                break
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass

        self._thread.join(timeout=5.0)
        self._sink.close()


class ABQueuedPlaybackOutput(QueuedPlaybackOutput):
    """
    Queue paired raw/enhanced chunks and apply A/B at playback time.

    Selecting Raw vs Enhanced after enqueue must affect the next samples
    heard, not only chunks already committed to the queue.
    """

    def __init__(
        self,
        sink: AudioOutput,
        *,
        sample_rate: int,
        chunk_samples: int = 1024,
        max_chunks: int = DEFAULT_MAX_CHUNKS,
    ) -> None:
        self._selectable = None
        self._ab_queue: queue.Queue[_ABPlaybackChunk | None] = queue.Queue(
            maxsize=max_chunks,
        )
        super().__init__(
            sink,
            sample_rate=sample_rate,
            chunk_samples=chunk_samples,
            max_chunks=max_chunks,
        )

    def bind_selectable(self, selectable: _ABSelectable) -> None:
        self._selectable = selectable

    def enqueue_ab(
        self,
        raw: np.ndarray,
        enhanced: np.ndarray,
        reference: np.ndarray,
    ) -> None:
        if self._closed:
            raise RuntimeError("ABQueuedPlaybackOutput is closed.")

        with self._error_lock:
            if self._consumer_error is not None:
                raise RuntimeError(
                    f"Playback consumer failed: {self._consumer_error}"
                )

        if enhanced.size == 0 and raw.size == 0:
            return

        chunk = _ABPlaybackChunk(
            raw=np.asarray(raw, dtype=np.float32).reshape(-1),
            enhanced=np.asarray(enhanced, dtype=np.float32).reshape(-1),
            reference=np.asarray(reference, dtype=np.float32).reshape(-1),
        )

        while not self._closed:
            try:
                self._ab_queue.put(chunk, timeout=0.05)
                self._stats.record_enqueue(self._ab_queue.qsize())
                return
            except queue.Full:
                if self._stop.is_set():
                    return

    def _consume(self) -> None:
        try:
            while True:
                try:
                    item = self._ab_queue.get(timeout=0.05)
                except queue.Empty:
                    if self._stop.is_set():
                        break
                    continue

                if item is None:
                    break

                if self._selectable is None:
                    raise RuntimeError(
                        "ABQueuedPlaybackOutput requires bind_selectable()"
                    )

                payload = self._selectable.select_playback_chunk(
                    item.raw,
                    item.enhanced,
                    item.reference,
                )

                if payload.size == 0:
                    continue

                now = time.perf_counter()
                interval = (
                    now - self._last_write_time
                    if self._last_write_time is not None
                    else 0.0
                )
                self._last_write_time = now
                self._stats.record_write(len(payload), interval)
                self._sink.write(payload)
        except BaseException as exc:
            with self._error_lock:
                self._consumer_error = exc
            raise

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        self._stop.set()

        while True:
            try:
                self._ab_queue.put(None, timeout=0.05)
                break
            except queue.Full:
                try:
                    self._ab_queue.get_nowait()
                except queue.Empty:
                    pass

        self._thread.join(timeout=5.0)
        self._sink.close()
