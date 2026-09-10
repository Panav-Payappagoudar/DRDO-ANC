"""Fine-tuned DeepFilterNet3 enhancer.

Loads the teammate artifact under ``models/dfn3_finetuned/`` without
modifying those files. Native streaming uses a derived ONNX tar.gz
written under ``data/cache/`` so the original export tree stays intact.
"""

from __future__ import annotations

import tarfile
from pathlib import Path

import numpy as np
import torch
from df import enhance, init_df

from .base import Enhancer
from .native import NativeDF3Backend
from .streaming import StreamingBuffer

FINETUNED_MODEL_NAME = "DeepFilterNet3-Finetuned"
FINETUNED_CHECKPOINT_EPOCH = 130
FINETUNED_STREAMING_DELAY_SAMPLES = 1440
REQUIRED_ONNX_MEMBERS = (
    "enc.onnx",
    "erb_dec.onnx",
    "df_dec.onnx",
    "config.ini",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_finetuned_artifact_root(
    root: Path | None = None,
) -> Path:
    """Return the extracted fine-tuned artifact root."""

    base = project_root() / "models" / "dfn3_finetuned"
    if root is not None:
        base = Path(root)

    nested = base / "live-finetuned"
    if (nested / "models" / "dfn3-epoch-130-onnx").exists():
        return nested

    if (base / "models" / "dfn3-epoch-130-onnx").exists():
        return base

    raise FileNotFoundError(
        "Fine-tuned DeepFilterNet3 artifact not found. "
        "Expected models/dfn3_finetuned/models/dfn3-epoch-130-onnx/ "
        "or models/dfn3_finetuned/live-finetuned/models/dfn3-epoch-130-onnx/."
    )


def finetuned_export_dir(artifact_root: Path | None = None) -> Path:
    root = resolve_finetuned_artifact_root(artifact_root)
    path = root / "models" / "dfn3-epoch-130-onnx" / "_export_model"
    if not (path / "config.ini").exists():
        raise FileNotFoundError(
            f"Fine-tuned export config.ini not found: {path / 'config.ini'}"
        )
    checkpoint = path / "checkpoints" / "model_130.ckpt"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Fine-tuned checkpoint not found: {checkpoint}"
        )
    return path


def finetuned_onnx_dir(artifact_root: Path | None = None) -> Path:
    root = resolve_finetuned_artifact_root(artifact_root)
    path = root / "models" / "dfn3-epoch-130-onnx" / "onnx"
    missing = [name for name in REQUIRED_ONNX_MEMBERS if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Fine-tuned ONNX export is incomplete in {path}: {missing}"
        )
    return path


def default_native_dll_path() -> Path:
    return (
        project_root()
        / "external"
        / "DeepFilterNet"
        / "target"
        / "release"
        / "df.dll"
    )


def finetuned_onnx_bundle_path() -> Path:
    return (
        project_root()
        / "data"
        / "cache"
        / "dfn3_finetuned_onnx.tar.gz"
    )


def ensure_finetuned_onnx_bundle(
    onnx_dir: Path | None = None,
    dest: Path | None = None,
) -> Path:
    """Pack the unzipped ONNX export into the layout expected by df.dll."""

    source = finetuned_onnx_dir() if onnx_dir is None else Path(onnx_dir)
    bundle = finetuned_onnx_bundle_path() if dest is None else Path(dest)
    bundle.parent.mkdir(parents=True, exist_ok=True)

    newest_source = max(
        (source / name).stat().st_mtime for name in REQUIRED_ONNX_MEMBERS
    )
    if bundle.exists() and bundle.stat().st_mtime >= newest_source:
        return bundle

    with tarfile.open(bundle, "w:gz") as archive:
        for name in REQUIRED_ONNX_MEMBERS:
            archive.add(source / name, arcname=f"tmp/export/{name}")

    return bundle


class FineTunedDeepFilterNetEnhancer(Enhancer):
    """DeepFilterNet3 enhancer backed by the epoch-130 fine-tuned artifact."""

    def __init__(self) -> None:
        self.model = None
        self.df_state = None
        self.device = None

        self._sample_rate = None
        self._name = FINETUNED_MODEL_NAME
        self._checkpoint_epoch = None

        self._native_backend = None
        self._stream_buffer = None

    def load(self) -> None:
        export_dir = finetuned_export_dir()
        loaded = init_df(
            model_base_dir=str(export_dir),
            epoch=FINETUNED_CHECKPOINT_EPOCH,
            log_file=None,
        )
        if len(loaded) == 4:
            self.model, self.df_state, _suffix, epoch = loaded
        elif len(loaded) == 3:
            self.model, self.df_state, _suffix = loaded
            epoch = FINETUNED_CHECKPOINT_EPOCH
        else:
            raise RuntimeError(
                f"Unexpected init_df() return length: {len(loaded)}"
            )

        self._checkpoint_epoch = epoch
        self._name = FINETUNED_MODEL_NAME
        self._sample_rate = self.df_state.sr()
        self.device = next(self.model.parameters()).device

        print(f"Model:       {self._name}")
        print(f"Checkpoint:  epoch {epoch}")
        print(f"Export dir:  {export_dir}")
        print(f"Device:      {self.device}")
        print(f"DF rate:     {self._sample_rate} Hz")

        dll_path = default_native_dll_path()
        model_path = ensure_finetuned_onnx_bundle()

        self._native_backend = NativeDF3Backend(
            dll_path=dll_path,
            model_path=model_path,
        )
        self._native_backend.load()
        self._stream_buffer = StreamingBuffer(
            self._native_backend.frame_length()
        )

        if self._sample_rate != 48_000:
            raise ValueError(
                "Expected fine-tuned DeepFilterNet sample rate to be 48000 Hz, "
                f"got {self._sample_rate}"
            )

    def process(self, audio: torch.Tensor) -> torch.Tensor:
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
        if self._native_backend is None:
            raise RuntimeError(
                "Streaming backend is not loaded. "
                "Call load() before process_stream()."
            )

        if self._stream_buffer is None:
            raise RuntimeError("Streaming buffer is not initialized.")

        if not isinstance(audio_chunk, torch.Tensor):
            raise TypeError(
                f"Expected torch.Tensor, got {type(audio_chunk).__name__}"
            )

        if audio_chunk.ndim not in (1, 2):
            raise ValueError(
                f"Expected audio with 1 or 2 dimensions, got {audio_chunk.ndim}"
            )

        if not audio_chunk.is_floating_point():
            audio_chunk = audio_chunk.float()

        if audio_chunk.ndim == 2:
            if audio_chunk.shape[0] != 1:
                raise ValueError(
                    "Streaming currently supports mono audio only."
                )
            audio_chunk = audio_chunk.squeeze(0)

        audio_np = (
            audio_chunk.detach().cpu().numpy().astype(np.float32, copy=False)
        )
        frames = self._stream_buffer.append(audio_np)
        if not frames:
            return torch.empty(0, dtype=torch.float32)

        enhanced_frames = [
            self._native_backend.process_frame(frame) for frame in frames
        ]
        enhanced = np.concatenate(enhanced_frames)
        return torch.from_numpy(enhanced.astype(np.float32, copy=False))

    def flush(self) -> torch.Tensor:
        if self._native_backend is None:
            raise RuntimeError(
                "Streaming backend is not loaded. Call load() before flush()."
            )

        if self._stream_buffer is None:
            raise RuntimeError("Streaming buffer is not initialized.")

        pending = self._stream_buffer.pending_samples()
        if pending == 0:
            return torch.empty(0, dtype=torch.float32)

        frame_length = self._native_backend.frame_length()
        if pending >= frame_length:
            raise RuntimeError(
                "Streaming buffer contains a complete frame. "
                "This indicates a buffer management error."
            )

        buffered = self._stream_buffer.flush()
        padded = np.zeros(frame_length, dtype=np.float32)
        padded[:pending] = buffered
        enhanced = self._native_backend.process_frame(padded)[:pending]
        return torch.from_numpy(enhanced.astype(np.float32, copy=False))

    def reset(self) -> None:
        if self._native_backend is None:
            raise RuntimeError(
                "Streaming backend is not loaded. Call load() before reset()."
            )

        self._native_backend.reset()
        if self._stream_buffer is not None:
            self._stream_buffer.clear()

    def sample_rate(self) -> int:
        if self._sample_rate is None:
            raise RuntimeError("Enhancer is not loaded. Call load() first.")
        return self._sample_rate

    def name(self) -> str:
        return self._name
