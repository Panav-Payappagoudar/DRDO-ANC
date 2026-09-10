#!/usr/bin/env python3
"""Unit tests for GUI waveform downsampling (no microphone or Qt required)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

src_dir = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_dir))

from drdo_anc.gui.waveform import WaveformProcessor


def test_empty_chunks_keep_previous_waveform() -> None:
  processor = WaveformProcessor(target_points=100)
  first_in = np.linspace(-1.0, 1.0, 256, dtype=np.float32)
  first_out = np.linspace(1.0, -1.0, 256, dtype=np.float32)

  in_reduced, out_reduced = processor.process(first_in, first_out)
  assert in_reduced.shape == (100,)
  assert out_reduced.shape == (100,)

  in_again, out_again = processor.process(np.array([], dtype=np.float32), np.array([], dtype=np.float32))
  np.testing.assert_array_equal(in_again, in_reduced)
  np.testing.assert_array_equal(out_again, out_reduced)


def test_small_chunk_is_padded() -> None:
  processor = WaveformProcessor(target_points=50)
  tiny = np.array([0.25, -0.5, 0.75], dtype=np.float32)

  reduced_in, _ = processor.process(tiny, tiny)
  assert reduced_in.shape == (50,)
  np.testing.assert_allclose(reduced_in[:3], tiny)


def test_large_chunk_preserves_signed_peaks() -> None:
  processor = WaveformProcessor(target_points=20)
  data = np.zeros(400, dtype=np.float32)
  data[37] = -0.9
  data[201] = 0.8

  reduced, _ = processor.process(data, data)
  assert reduced.shape == (20,)
  assert np.min(reduced) <= -0.8
  assert np.max(reduced) >= 0.7


def test_arbitrary_chunk_sizes() -> None:
  processor = WaveformProcessor(target_points=30)

  for length in (1, 7, 29, 30, 31, 1024, 4096):
    rng = np.random.default_rng(length)
    chunk = rng.uniform(-1.0, 1.0, size=length).astype(np.float32)
    reduced_in, reduced_out = processor.process(chunk, chunk)
    assert reduced_in.shape == (30,)
    assert reduced_out.shape == (30,)


def main() -> None:
  tests = [
    test_empty_chunks_keep_previous_waveform,
    test_small_chunk_is_padded,
    test_large_chunk_preserves_signed_peaks,
    test_arbitrary_chunk_sizes,
  ]

  for test in tests:
    test()
    print(f"PASS: {test.__name__}")

  print(f"\nAll {len(tests)} GUI waveform tests passed.")


if __name__ == "__main__":
  main()
