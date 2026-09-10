"""Live-path smoke tests for pretrained and fine-tuned DeepFilterNet3.

Uses the same registry + StreamingPipeline path as the live CLI.
Does not open physical devices (see run_live_soak.py for hardware).
"""

from __future__ import annotations

import numpy as np
import torch

from drdo_anc.audio.live import FakeAudioInput, FakeAudioOutput, StreamingPipeline
from drdo_anc.enhancement import (
    DeepFilterNetEnhancer,
    FineTunedDeepFilterNetEnhancer,
    create_enhancer,
    get_model_config,
    list_models,
)
from drdo_anc.enhancement.finetuned import FINETUNED_MODEL_NAME


PRETRAINED_MODEL_NAME = "DeepFilterNet3"
SAMPLE_RATE = 48_000
CHUNK_SIZE = 1024


def _assert_mono_finite(audio: torch.Tensor | np.ndarray, *, name: str) -> np.ndarray:
    if isinstance(audio, torch.Tensor):
        array = audio.detach().cpu().numpy()
    else:
        array = np.asarray(audio)

    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        if array.shape[0] != 1 and array.shape[1] != 1:
            raise AssertionError(f"{name} is not mono: shape {array.shape}")
        array = array.reshape(-1)
    elif array.ndim != 1:
        raise AssertionError(f"{name} has ndim {array.ndim}")

    if not np.isfinite(array).all():
        raise AssertionError(f"{name} contains NaN/Inf")
    return array


def test_registry_live_entry() -> None:
    models = list_models()
    assert PRETRAINED_MODEL_NAME in models
    assert FINETUNED_MODEL_NAME in models

    pretrained = get_model_config(PRETRAINED_MODEL_NAME)
    finetuned = get_model_config(FINETUNED_MODEL_NAME)
    assert pretrained.streaming_delay_samples == 1440
    assert finetuned.streaming_delay_samples == 1440

    unloaded_pre = create_enhancer(PRETRAINED_MODEL_NAME, load=False)
    unloaded_ft = create_enhancer(FINETUNED_MODEL_NAME, load=False)
    assert isinstance(unloaded_pre, DeepFilterNetEnhancer)
    assert isinstance(unloaded_ft, FineTunedDeepFilterNetEnhancer)
    assert unloaded_pre.name() == PRETRAINED_MODEL_NAME
    assert unloaded_ft.name() == FINETUNED_MODEL_NAME


def _check_enhancer(model_name: str) -> None:
    config = get_model_config(model_name)
    enhancer = create_enhancer(model_name)
    assert enhancer.sample_rate() == SAMPLE_RATE
    assert config.streaming_delay_samples == 1440

    audio = torch.randn(SAMPLE_RATE, dtype=torch.float32) * 0.05
    offline = _assert_mono_finite(enhancer.process(audio), name=f"{model_name} offline")
    assert offline.size > 0

    enhancer.reset()
    awkward = [300, 700, 250, 1024, 137, 911]
    position = 0
    streamed_parts: list[np.ndarray] = []
    for size in awkward:
        chunk = audio[position : position + size]
        position += size
        part = enhancer.process_stream(chunk)
        streamed_parts.append(_assert_mono_finite(part, name=f"{model_name} stream"))
    flush = _assert_mono_finite(enhancer.flush(), name=f"{model_name} flush")
    streamed = np.concatenate([*streamed_parts, flush]) if flush.size else np.concatenate(streamed_parts)
    consumed = position
    if abs(int(streamed.size) - consumed) > config.streaming_delay_samples:
        raise AssertionError(
            f"{model_name} streaming length {streamed.size} vs input {consumed}"
        )

    enhancer.reset()
    first = _collect_stream(enhancer, audio[: 8 * CHUNK_SIZE], CHUNK_SIZE)
    enhancer.reset()
    second = _collect_stream(enhancer, audio[: 8 * CHUNK_SIZE], CHUNK_SIZE)
    if first.size != second.size:
        raise AssertionError(
            f"{model_name} reset did not restore stream length "
            f"{first.size} vs {second.size}"
        )
    if not np.allclose(first, second, atol=1e-5, rtol=1e-4):
        raise AssertionError(
            f"{model_name} repeated stream after reset diverged"
        )


def _collect_stream(
    enhancer,
    audio: torch.Tensor,
    chunk_size: int,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for start in range(0, audio.numel(), chunk_size):
        part = enhancer.process_stream(audio[start : start + chunk_size])
        parts.append(_assert_mono_finite(part, name="collect"))
    parts.append(_assert_mono_finite(enhancer.flush(), name="collect flush"))
    nonempty = [part for part in parts if part.size]
    return np.concatenate(nonempty) if nonempty else np.empty(0, dtype=np.float32)


def _check_pipeline(model_name: str) -> None:
    enhancer = create_enhancer(model_name)
    rng = np.random.default_rng(0)
    chunks = [
        rng.standard_normal(CHUNK_SIZE).astype(np.float32) * 0.05
        for _ in range(12)
    ]
    audio_input = FakeAudioInput(chunks, sample_rate=SAMPLE_RATE)
    audio_output = FakeAudioOutput(sample_rate=SAMPLE_RATE)
    pipeline = StreamingPipeline(
        audio_input,
        audio_output,
        enhancer,
        read_chunk_size=CHUNK_SIZE,
        instrumentation=True,
    )
    pipeline.run()
    written = _assert_mono_finite(audio_output.all_written(), name=f"{model_name} pipeline")
    input_samples = CHUNK_SIZE * len(chunks)
    if abs(written.size - input_samples) > get_model_config(model_name).streaming_delay_samples:
        raise AssertionError(
            f"{model_name} pipeline wrote {written.size} for {input_samples} input"
        )
    assert pipeline.instrumentation is not None
    assert pipeline.instrumentation.chunk_count == len(chunks)
    assert pipeline.instrumentation.input_samples == input_samples


def test_pretrained_load_and_stream() -> None:
    _check_enhancer(PRETRAINED_MODEL_NAME)


def test_finetuned_load_and_stream() -> None:
    _check_enhancer(FINETUNED_MODEL_NAME)


def test_pretrained_streaming_pipeline() -> None:
    _check_pipeline(PRETRAINED_MODEL_NAME)


def test_finetuned_streaming_pipeline() -> None:
    _check_pipeline(FINETUNED_MODEL_NAME)


def main() -> None:
    print("=" * 70)
    print("DRDO-ANC | Fine-tuned DF3 live-path smoke tests")
    print("=" * 70)

    tests = [
        test_registry_live_entry,
        test_pretrained_load_and_stream,
        test_finetuned_load_and_stream,
        test_pretrained_streaming_pipeline,
        test_finetuned_streaming_pipeline,
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
