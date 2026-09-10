from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .interfaces import AudioInput, AudioOutput


def _import_sounddevice():
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise ImportError(
            "sounddevice is required for desktop live audio. "
            "Install it with: pip install sounddevice"
        ) from exc

    return sd


def list_audio_devices() -> list[dict[str, Any]]:
    """
    Return host audio devices reported by PortAudio/sounddevice.

    Each entry contains ``index``, ``name``, ``max_input_channels``,
    ``max_output_channels``, and ``default_sample_rate``.
    """

    sd = _import_sounddevice()
    devices: list[dict[str, Any]] = []

    for index, info in enumerate(sd.query_devices()):
        devices.append(
            {
                "index": index,
                "name": info["name"],
                "max_input_channels": info["max_input_channels"],
                "max_output_channels": info["max_output_channels"],
                "default_sample_rate": info["default_samplerate"],
                "hostapi": info["hostapi"],
            }
        )

    return devices


def format_device_listing() -> str:
    """Return a human-readable device listing for CLI output."""

    sd = _import_sounddevice()
    lines = ["Available audio devices:"]

    for device in list_audio_devices():
        hostapi_name = sd.query_hostapis(device["hostapi"])["name"]
        lines.append(
            f"  [{device['index']}] {device['name']} "
            f"(in={device['max_input_channels']}, "
            f"out={device['max_output_channels']}, "
            f"default_sr={device['default_sample_rate']:.0f} Hz, "
            f"api={hostapi_name})"
        )

    return "\n".join(lines)


def _device_channel_count(
    device: int | str | None,
    kind: str,
) -> int:
    sd = _import_sounddevice()
    info = sd.query_devices(device, kind=kind)

    if kind == "input":
        channels = info["max_input_channels"]
    else:
        channels = info["max_output_channels"]

    if channels < 1:
        raise ValueError(
            f"Device {device!r} does not support {kind}."
        )

    return int(channels)


def downmix_to_mono(audio: np.ndarray) -> np.ndarray:
    """
    Convert captured audio to mono float32 ``[T]``.

    Accepts ``[T]``, ``[T, 1]``, or ``[T, C]`` and averages channels.
    """

    array = np.asarray(audio, dtype=np.float32)

    if array.ndim == 1:
        return array.reshape(-1)

    if array.ndim == 2:
        if array.shape[1] == 1:
            return array[:, 0]

        return array.mean(axis=1, dtype=np.float32)

    raise ValueError(
        f"Expected 1D or 2D audio, got shape {array.shape}."
    )


def upmix_mono_to_channels(
    audio: np.ndarray,
    channels: int,
) -> np.ndarray:
    """
    Convert mono ``[T]`` to host layout ``[T, C]`` for playback.
    """

    if channels < 1:
        raise ValueError("channels must be positive.")

    mono = np.asarray(audio, dtype=np.float32).reshape(-1)

    if channels == 1:
        return mono.reshape(-1, 1)

    return np.column_stack([mono] * channels)


@dataclass
class SoundDeviceStreamStats:
    """Runtime counters for live-audio diagnostics."""

    sample_rate: int
    input_channels: int
    output_channels: int
    blocksize: int
    chunks_processed: int = 0
    samples_read: int = 0
    samples_written: int = 0
    input_overflows: int = 0
    peak_input: float = 0.0
    peak_output: float = 0.0
    elapsed_s: float = 0.0
    _start_time: float | None = field(default=None, repr=False)

    def mark_start(self) -> None:
        self._start_time = time.perf_counter()

    def mark_stop(self) -> None:
        if self._start_time is not None:
            self.elapsed_s = time.perf_counter() - self._start_time

    def as_dict(self) -> dict[str, Any]:
        expected_duration = (
            self.samples_read / self.sample_rate
            if self.sample_rate > 0
            else 0.0
        )

        return {
            "sample_rate": self.sample_rate,
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "blocksize": self.blocksize,
            "dtype": "float32",
            "range": "[-1.0, 1.0]",
            "chunks_processed": self.chunks_processed,
            "samples_read": self.samples_read,
            "samples_written": self.samples_written,
            "input_overflows": self.input_overflows,
            "peak_input": self.peak_input,
            "peak_output": self.peak_output,
            "elapsed_s": self.elapsed_s,
            "expected_audio_duration_s": expected_duration,
            "realtime_ratio": (
                expected_duration / self.elapsed_s
                if self.elapsed_s > 0
                else None
            ),
        }


class SoundDeviceDuplexSession:
    """
    Shared full-duplex PortAudio stream for synchronized capture/playback.

    A single ``sd.Stream`` keeps input and output on the same clock. The
    stream is opened in the constructor but not started until the first
    ``read_mono()`` or ``write_mono()`` call so the output device does not
    underflow before audio is available.
    """

    def __init__(
        self,
        sample_rate: int,
        *,
        input_device: int | str | None = None,
        output_device: int | str | None = None,
        blocksize: int = 0,
        latency: str | float = "high",
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")

        sd = _import_sounddevice()

        self._sample_rate = sample_rate
        self._input_device = input_device
        self._output_device = output_device
        self._input_channels = _device_channel_count(
            input_device,
            "input",
        )
        self._output_channels = _device_channel_count(
            output_device,
            "output",
        )
        self._blocksize = blocksize
        self._started = False
        self._closed = False
        self.stats = SoundDeviceStreamStats(
            sample_rate=sample_rate,
            input_channels=self._input_channels,
            output_channels=self._output_channels,
            blocksize=blocksize,
        )

        self._stream = sd.Stream(
            samplerate=sample_rate,
            device=(input_device, output_device),
            channels=(self._input_channels, self._output_channels),
            dtype="float32",
            blocksize=blocksize,
            latency=latency,
        )

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def input_channels(self) -> int:
        return self._input_channels

    @property
    def output_channels(self) -> int:
        return self._output_channels

    def _ensure_started(self) -> None:
        if self._closed:
            raise RuntimeError("Duplex session is closed.")

        if not self._started:
            self._stream.start()
            self._started = True
            self.stats.mark_start()

    def read_mono(self, frames: int) -> tuple[np.ndarray, bool]:
        if frames <= 0:
            return np.empty(0, dtype=np.float32), False

        self._ensure_started()

        data, overflowed = self._stream.read(frames)
        mono = downmix_to_mono(data)

        if overflowed:
            self.stats.input_overflows += 1

        if mono.size > 0:
            self.stats.peak_input = max(
                self.stats.peak_input,
                float(np.max(np.abs(mono))),
            )
            self.stats.samples_read += int(mono.size)

        self.stats.chunks_processed += 1

        return mono, bool(overflowed)

    def write_mono(self, audio: np.ndarray) -> None:
        mono = np.asarray(audio, dtype=np.float32).reshape(-1)

        if mono.size == 0:
            return

        self._ensure_started()

        host_audio = upmix_mono_to_channels(
            mono,
            self._output_channels,
        )

        self._stream.write(host_audio)

        self.stats.peak_output = max(
            self.stats.peak_output,
            float(np.max(np.abs(mono))),
        )
        self.stats.samples_written += int(mono.size)

    def close(self) -> None:
        if self._closed:
            return

        self.stats.mark_stop()

        if self._started and self._stream is not None:
            self._stream.stop()

        if self._stream is not None:
            self._stream.close()
            self._stream = None

        self._closed = True


class SoundDevicePlaybackSession:
    """
    Playback-only PortAudio stream for WAV/demo replay without microphone capture.

    Exposes the same ``write_mono`` / ``close`` surface as ``SoundDeviceDuplexSession``
    so ``SoundDeviceAudioOutput`` can be reused unchanged.
    """

    def __init__(
        self,
        sample_rate: int,
        *,
        output_device: int | str | None = None,
        blocksize: int = 0,
        latency: str | float = "high",
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")

        sd = _import_sounddevice()

        self._sample_rate = sample_rate
        self._output_device = output_device
        self._output_channels = _device_channel_count(
            output_device,
            "output",
        )
        self._blocksize = blocksize
        self._started = False
        self._closed = False
        self.stats = SoundDeviceStreamStats(
            sample_rate=sample_rate,
            input_channels=0,
            output_channels=self._output_channels,
            blocksize=blocksize,
        )

        self._stream = sd.OutputStream(
            samplerate=sample_rate,
            device=output_device,
            channels=self._output_channels,
            dtype="float32",
            blocksize=blocksize,
            latency=latency,
        )

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def output_channels(self) -> int:
        return self._output_channels

    def _ensure_started(self) -> None:
        if self._closed:
            raise RuntimeError("Playback session is closed.")

        if not self._started:
            self._stream.start()
            self._started = True
            self.stats.mark_start()

    def write_mono(self, audio: np.ndarray) -> None:
        mono = np.asarray(audio, dtype=np.float32).reshape(-1)

        if mono.size == 0:
            return

        self._ensure_started()

        host_audio = upmix_mono_to_channels(
            mono,
            self._output_channels,
        )

        self._stream.write(host_audio)

        self.stats.peak_output = max(
            self.stats.peak_output,
            float(np.max(np.abs(mono))),
        )
        self.stats.samples_written += int(mono.size)
        self.stats.chunks_processed += 1

    def close(self) -> None:
        if self._closed:
            return

        self.stats.mark_stop()

        if self._started and self._stream is not None:
            self._stream.stop()

        if self._stream is not None:
            self._stream.close()
            self._stream = None

        self._closed = True


def open_sounddevice_io(
    sample_rate: int,
    *,
    input_device: int | str | None = None,
    output_device: int | str | None = None,
    blocksize: int = 0,
    latency: str | float = "high",
) -> tuple[SoundDeviceAudioInput, SoundDeviceAudioOutput]:
    """
    Open a synchronized input/output pair backed by one duplex stream.

    Prefer this factory over constructing ``SoundDeviceAudioInput`` and
    ``SoundDeviceAudioOutput`` separately.
    """

    session = SoundDeviceDuplexSession(
        sample_rate,
        input_device=input_device,
        output_device=output_device,
        blocksize=blocksize,
        latency=latency,
    )

    return (
        SoundDeviceAudioInput(session),
        SoundDeviceAudioOutput(session),
    )


def open_sounddevice_output(
    sample_rate: int,
    *,
    output_device: int | str | None = None,
    blocksize: int = 0,
    latency: str | float = "high",
) -> SoundDeviceAudioOutput:
    """Open a playback-only output device for demo/offline replay."""

    session = SoundDevicePlaybackSession(
        sample_rate,
        output_device=output_device,
        blocksize=blocksize,
        latency=latency,
    )

    return SoundDeviceAudioOutput(session)


def close_sounddevice_output(audio_output: SoundDeviceAudioOutput) -> None:
    """Close a playback session opened via ``open_sounddevice_output()``."""

    audio_output.close()


class SoundDeviceAudioInput(AudioInput):
    """
    Desktop microphone capture via a shared duplex PortAudio stream.

    Host capture uses the device's native channel count (typically stereo
    for laptop microphone arrays). ``read()`` always returns mono float32
    ``[T]`` in the range ``[-1, 1]`` by averaging channels.
    """

    def __init__(
        self,
        session: SoundDeviceDuplexSession,
    ) -> None:
        self._session = session
        self._closed = False

    def sample_rate(self) -> int:
        return self._session.sample_rate

    @property
    def host_input_channels(self) -> int:
        return self._session.input_channels

    @property
    def stats(self) -> SoundDeviceStreamStats:
        return self._session.stats

    def read(self, max_samples: int) -> np.ndarray:
        if self._closed:
            raise RuntimeError("AudioInput is closed.")

        mono, _overflowed = self._session.read_mono(max_samples)
        return mono

    def close(self) -> None:
        self._closed = True


class SoundDeviceAudioOutput(AudioOutput):
    """
    Desktop speaker playback via a shared duplex PortAudio stream.

    Accepts mono float32 ``[T]`` and duplicates it to the device's native
    output channel count (typically stereo) before playback.
    """

    def __init__(
        self,
        session: SoundDeviceDuplexSession,
    ) -> None:
        self._session = session
        self._closed = False

    def sample_rate(self) -> int:
        return self._session.sample_rate

    @property
    def host_output_channels(self) -> int:
        return self._session.output_channels

    @property
    def stats(self) -> SoundDeviceStreamStats:
        return self._session.stats

    def write(self, audio: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("AudioOutput is closed.")

        self._session.write_mono(audio)

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        self._session.close()


def close_sounddevice_io(
    audio_input: SoundDeviceAudioInput,
    audio_output: SoundDeviceAudioOutput,
) -> None:
    """Close a duplex session opened via ``open_sounddevice_io()``."""

    audio_input.close()
    audio_output.close()
