from pathlib import Path

import numpy as np
import torch

from df import enhance, init_df

from .base import Enhancer
from .native import NativeDF3Backend
from .streaming import StreamingBuffer


class DeepFilterNetEnhancer(Enhancer):
    """DeepFilterNet3 implementation of the Enhancer interface."""

    def __init__(self):
        self.model = None
        self.df_state = None
        self.device = None

        self._sample_rate = None
        self._name = "DeepFilterNet3"

        self._native_backend = None
        self._stream_buffer = None

    def load(self) -> None:
        """Load DeepFilterNet3 and initialize offline and streaming backends."""

        # ---------------------------------------------------------
        # Offline PyTorch backend
        # ---------------------------------------------------------

        loaded = init_df()
        if len(loaded) == 4:
            self.model, self.df_state, suffix, epoch = loaded
        elif len(loaded) == 3:
            self.model, self.df_state, suffix = loaded
            epoch = "unknown"
        else:
            raise RuntimeError(
                f"Unexpected init_df() return length: {len(loaded)}"
            )

        self._name = suffix
        self._sample_rate = self.df_state.sr()
        self.device = next(self.model.parameters()).device

        print(f"Model:       {self._name}")
        print(f"Checkpoint:  epoch {epoch}")
        print(f"Device:      {self.device}")
        print(f"DF rate:     {self._sample_rate} Hz")

        # ---------------------------------------------------------
        # Native streaming backend
        # ---------------------------------------------------------

        project_root = Path(__file__).resolve().parents[3]

        dll_path = (
            project_root
            / "external"
            / "DeepFilterNet"
            / "target"
            / "release"
            / "df.dll"
        )

        model_path = (
            project_root
            / "external"
            / "DeepFilterNet"
            / "models"
            / "DeepFilterNet3_onnx.tar.gz"
        )

        self._native_backend = NativeDF3Backend(
            dll_path=dll_path,
            model_path=model_path,
        )

        self._native_backend.load()

        self._stream_buffer = StreamingBuffer(
            self._native_backend.frame_length()
        )

        # Sanity check: both backends must use the same sample rate.
        if self._sample_rate != 48_000:
            raise ValueError(
                f"Expected DeepFilterNet sample rate to be 48000 Hz, "
                f"got {self._sample_rate}"
            )

    def process(self, audio: torch.Tensor) -> torch.Tensor:
        """Enhance a complete audio signal using the offline PyTorch backend."""

        if self.model is None or self.df_state is None:
            raise RuntimeError(
                "Enhancer is not loaded. Call load() before process()."
            )

        if not isinstance(audio, torch.Tensor):
            raise TypeError(
                f"Expected torch.Tensor, got {type(audio).__name__}"
            )

        if audio.ndim not in (1, 2):
            raise ValueError(
                f"Expected audio with 1 or 2 dimensions, got {audio.ndim}"
            )

        if not audio.is_floating_point():
            audio = audio.float()

        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        return enhance(
            self.model,
            self.df_state,
            audio,
        )

    def process_stream(
        self,
        audio_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """
        Enhance an arbitrary-sized mono audio chunk using native DF3 streaming.

        Input:
            [T] or [1, T]

        Output:
            [T_enhanced]

        The output may contain fewer samples than the input because
        incomplete frames remain buffered internally.
        """

        if self._native_backend is None:
            raise RuntimeError(
                "Streaming backend is not loaded. "
                "Call load() before process_stream()."
            )

        if self._stream_buffer is None:
            raise RuntimeError(
                "Streaming buffer is not initialized."
            )

        if not isinstance(audio_chunk, torch.Tensor):
            raise TypeError(
                f"Expected torch.Tensor, "
                f"got {type(audio_chunk).__name__}"
            )

        if audio_chunk.ndim not in (1, 2):
            raise ValueError(
                f"Expected audio with 1 or 2 dimensions, "
                f"got {audio_chunk.ndim}"
            )

        if not audio_chunk.is_floating_point():
            audio_chunk = audio_chunk.float()

        # Streaming backend currently supports mono only.
        if audio_chunk.ndim == 2:
            if audio_chunk.shape[0] != 1:
                raise ValueError(
                    "Streaming currently supports mono audio only."
                )

            audio_chunk = audio_chunk.squeeze(0)

        # Torch → NumPy
        audio_np = (
            audio_chunk
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )

        # Split arbitrary chunk into complete DF frames.
        frames = self._stream_buffer.append(audio_np)

        if not frames:
            return torch.empty(
                0,
                dtype=torch.float32,
            )

        # Process every complete frame.
        enhanced_frames = [
            self._native_backend.process_frame(frame)
            for frame in frames
        ]

        enhanced = np.concatenate(
            enhanced_frames
        )

        return torch.from_numpy(
            enhanced.astype(
                np.float32,
                copy=False,
            )
        )

    def flush(self) -> torch.Tensor:
        """
        Finish the current streaming sequence.

        Any remaining partial frame is zero-padded to one complete
        DeepFilterNet frame. Only output corresponding to the original
        buffered samples is returned.
        """

        if self._native_backend is None:
            raise RuntimeError(
                "Streaming backend is not loaded. "
                "Call load() before flush()."
            )

        if self._stream_buffer is None:
            raise RuntimeError(
                "Streaming buffer is not initialized."
            )

        pending = self._stream_buffer.pending_samples()

        if pending == 0:
            return torch.empty(
                0,
                dtype=torch.float32,
            )

        frame_length = self._native_backend.frame_length()

        if pending >= frame_length:
            raise RuntimeError(
                "Streaming buffer contains a complete frame. "
                "This indicates a buffer management error."
            )

        # Retrieve and clear the remaining real samples.
        buffered = self._stream_buffer.flush()

        # Complete the frame with zeros.
        padded = np.zeros(
            frame_length,
            dtype=np.float32,
        )

        padded[:pending] = buffered

        # Process the final padded frame.
        enhanced = self._native_backend.process_frame(
            padded
        )

        # Discard output corresponding to artificial zero padding.
        enhanced = enhanced[:pending]

        return torch.from_numpy(
            enhanced.astype(
                np.float32,
                copy=False,
            )
        )
        """
        Finish the current streaming sequence.

        Any remaining partial frame is zero-padded to one complete
        DeepFilterNet frame. Only output corresponding to the original
        buffered samples is returned.
        """

        if self._native_backend is None:
            raise RuntimeError(
                "Streaming backend is not loaded. "
                "Call load() before flush()."
            )

        if self._stream_buffer is None:
            raise RuntimeError(
                "Streaming buffer is not initialized."
            )

        pending = self._stream_buffer.pending()

        if pending == 0:
            return torch.empty(
                0,
                dtype=torch.float32,
            )

        frame_length = self._native_backend.frame_length()

        # We need enough zeros to complete one DF frame.
        padding = frame_length - pending

        padded = np.zeros(
            frame_length,
            dtype=np.float32,
        )

        # Copy the real buffered samples into the beginning.
        buffered = self._stream_buffer.flush()

        padded[:pending] = buffered

        # Process the complete padded frame.
        enhanced = self._native_backend.process_frame(
            padded
        )

        # Only return samples corresponding to real input.
        enhanced = enhanced[:pending]

        return torch.from_numpy(
            enhanced.astype(
                np.float32,
                copy=False,
            )
        )

    def reset(self) -> None:
        """Reset native DF3 state and streaming buffer."""

        if self._native_backend is None:
            raise RuntimeError(
                "Streaming backend is not loaded. "
                "Call load() before reset()."
            )

        self._native_backend.reset()

        if self._stream_buffer is not None:
            self._stream_buffer.clear()

    def sample_rate(self) -> int:
        """Return the sample rate expected by DeepFilterNet."""

        if self._sample_rate is None:
            raise RuntimeError(
                "Enhancer is not loaded. Call load() first."
            )

        return self._sample_rate

    def name(self) -> str:
        """Return the model name."""

        return self._name