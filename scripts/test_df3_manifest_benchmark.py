import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from drdo_anc.benchmark import (
    BenchmarkMode,
    MixtureGenerator,
    build_development_manifest,
)
from drdo_anc.benchmark.evaluation_manifest import (
    EvaluationManifest,
)
from drdo_anc.benchmark.manifest_benchmark import (
    ManifestBenchmarkRunner,
    delay_samples_for_mode,
    resample_mixture_for_enhancer,
    select_smoke_cases,
    validate_development_manifest,
)
from drdo_anc.dataset import ZipManifestDataset
from drdo_anc.enhancement import (
    create_enhancer,
    get_model_config,
    list_models,
)
from drdo_anc.enhancement.base import Enhancer

from build_evaluation_fixtures import (
    FIXTURE_DIR,
    METADATA_PATH,
    build_fixtures,
)


class MockEnhancer(Enhancer):
    """Identity enhancer for manifest benchmark integration tests."""

    def __init__(
        self,
        sample_rate: int = 48_000,
    ) -> None:
        self._sample_rate = sample_rate
        self.reset_calls = 0

    def load(self) -> None:
        return None

    def reset(self) -> None:
        self.reset_calls += 1

    def sample_rate(self) -> int:
        return self._sample_rate

    def name(self) -> str:
        return "MockEnhancer"

    def _to_mono(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim == 2:
            audio = audio.squeeze(0)

        return audio.float()

    def process(self, audio: torch.Tensor) -> torch.Tensor:
        mono = self._to_mono(audio)
        return mono.unsqueeze(0)

    def process_stream(
        self,
        audio_chunk: torch.Tensor,
    ) -> torch.Tensor:
        return self._to_mono(audio_chunk)

    def flush(self) -> torch.Tensor:
        return torch.empty(0, dtype=torch.float32)


def setup_module() -> None:
    build_fixtures()


def test_model_registry_lists_deepfilternet3() -> None:
    assert "DeepFilterNet3" in list_models()
    assert "DeepFilterNet3-Finetuned" in list_models()

    model_config = get_model_config("DeepFilterNet3")
    assert model_config.streaming_delay_samples == 1440

    enhancer = create_enhancer("DeepFilterNet3", load=False)
    assert enhancer.name() == "DeepFilterNet3"

    finetuned = create_enhancer("DeepFilterNet3-Finetuned", load=False)
    assert finetuned.name() == "DeepFilterNet3-Finetuned"


def test_manifest_validation() -> None:
    manifest = build_development_manifest(METADATA_PATH)
    validate_development_manifest(manifest)


def test_smoke_case_selection() -> None:
    manifest = build_development_manifest(METADATA_PATH)
    smoke_cases = select_smoke_cases(manifest)

    assert len(smoke_cases) == 2
    assert smoke_cases[0].noise_category == smoke_cases[1].noise_category
    assert smoke_cases[0].clean_source.sample_id == (
        smoke_cases[1].clean_source.sample_id
    )
    assert {case.snr_db for case in smoke_cases} == {0.0, 5.0}


def test_resample_mixture_for_enhancer() -> None:
    clean = np.ones(16_000, dtype=np.float32)
    noisy = np.full(16_000, 0.5, dtype=np.float32)

    clean_48k, noisy_48k, sample_rate = resample_mixture_for_enhancer(
        clean,
        noisy,
        16_000,
        48_000,
    )

    assert sample_rate == 48_000
    assert len(clean_48k) == 48_000
    assert len(noisy_48k) == 48_000


def test_delay_compensation_configuration() -> None:
    model_config = get_model_config("DeepFilterNet3")

    assert delay_samples_for_mode(
        BenchmarkMode.OFFLINE,
        streaming_delay_samples=model_config.streaming_delay_samples,
    ) == 0
    assert delay_samples_for_mode(
        BenchmarkMode.STREAMING,
        streaming_delay_samples=model_config.streaming_delay_samples,
    ) == 1440


def test_offline_and_streaming_receive_identical_noisy_input() -> None:
    manifest = build_development_manifest(METADATA_PATH)
    dataset = ZipManifestDataset(
        metadata_path=METADATA_PATH,
        repo_id=None,
        archive_dir=FIXTURE_DIR,
    )
    generator = MixtureGenerator(dataset)
    smoke_cases = select_smoke_cases(manifest)

    mixture = generator.generate(smoke_cases[0])

    clean_a, noisy_a, _ = resample_mixture_for_enhancer(
        mixture.clean,
        mixture.noisy,
        mixture.sample_rate,
        48_000,
    )
    clean_b, noisy_b, _ = resample_mixture_for_enhancer(
        mixture.clean,
        mixture.noisy,
        mixture.sample_rate,
        48_000,
    )

    assert np.array_equal(clean_a, clean_b)
    assert np.array_equal(noisy_a, noisy_b)


def test_smoke_benchmark_with_mock_enhancer() -> None:
    manifest = build_development_manifest(METADATA_PATH)
    dataset = ZipManifestDataset(
        metadata_path=METADATA_PATH,
        repo_id=None,
        archive_dir=FIXTURE_DIR,
    )
    generator = MixtureGenerator(dataset)
    enhancer = MockEnhancer()

    runner = ManifestBenchmarkRunner(
        enhancer,
        generator,
        streaming_delay_samples=1440,
        modes=(
            BenchmarkMode.OFFLINE,
            BenchmarkMode.STREAMING,
        ),
    )

    report = runner.run(
        manifest,
        select_smoke_cases(manifest),
    )

    assert len(report.successful_results()) == 4
    assert all(result.num_samples > 0 for result in report.successful_results())


def test_streaming_flush_length_with_mock_enhancer() -> None:
    manifest = build_development_manifest(METADATA_PATH)
    dataset = ZipManifestDataset(
        metadata_path=METADATA_PATH,
        repo_id=None,
        archive_dir=FIXTURE_DIR,
    )
    generator = MixtureGenerator(dataset)
    enhancer = MockEnhancer()

    runner = ManifestBenchmarkRunner(
        enhancer,
        generator,
        streaming_delay_samples=1440,
        modes=(BenchmarkMode.STREAMING,),
    )

    report = runner.run(
        manifest,
        select_smoke_cases(manifest)[:1],
    )

    result = report.successful_results()[0]
    assert result.mode == "streaming"
    assert result.num_samples > 0


def test_result_serialization() -> None:
    manifest = build_development_manifest(METADATA_PATH)
    dataset = ZipManifestDataset(
        metadata_path=METADATA_PATH,
        repo_id=None,
        archive_dir=FIXTURE_DIR,
    )
    generator = MixtureGenerator(dataset)
    enhancer = MockEnhancer()

    runner = ManifestBenchmarkRunner(
        enhancer,
        generator,
        streaming_delay_samples=1440,
        modes=(BenchmarkMode.OFFLINE,),
    )

    report = runner.run(
        manifest,
        select_smoke_cases(manifest)[:1],
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        json_path = Path(tmp_dir) / "report.json"
        csv_path = Path(tmp_dir) / "report.csv"

        report.save_json(json_path)
        report.save_csv(csv_path)

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["successful_cases"] == 1
        assert csv_path.read_text(encoding="utf-8").count("\n") >= 2


def test_df3_smoke_benchmark_integration() -> None:
    if os.environ.get("SIH26_INTEGRATION") != "1":
        return

    import urllib.request

    from drdo_anc.enhancement import create_enhancer, get_model_config

    url = (
        "https://huggingface.co/datasets/"
        "Panav-Payappagoudar/sih-26-processed-audio/"
        "resolve/main/metadata.csv"
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        metadata_path = Path(tmp_dir) / "metadata.csv"
        with urllib.request.urlopen(url, timeout=180) as resp:
            metadata_path.write_bytes(resp.read())

        manifest = build_development_manifest(metadata_path)
        validate_development_manifest(manifest)

        dataset = ZipManifestDataset(metadata_path=metadata_path)
        generator = MixtureGenerator(dataset)

        model_config = get_model_config("DeepFilterNet3")
        enhancer = create_enhancer("DeepFilterNet3")

        runner = ManifestBenchmarkRunner(
            enhancer,
            generator,
            streaming_delay_samples=(
                model_config.streaming_delay_samples
            ),
            modes=(
                BenchmarkMode.OFFLINE,
                BenchmarkMode.STREAMING,
            ),
        )

        report = runner.run(
            manifest,
            select_smoke_cases(manifest),
        )

    offline = [
        result
        for result in report.successful_results()
        if result.mode == "offline"
    ]
    streaming = [
        result
        for result in report.successful_results()
        if result.mode == "streaming"
    ]

    assert len(offline) == 2
    assert len(streaming) == 2

    for result in offline:
        assert result.delay_samples == 0
        assert result.si_sdr is not None
        assert result.si_sdr > 0.0

    for result in streaming:
        assert result.delay_samples == 1440
        assert result.si_sdr is not None
        assert result.si_sdr > 0.0


def main() -> None:
    print("=" * 70)
    print("DRDO-ANC | DF3 Manifest Benchmark Tests")
    print("=" * 70)

    build_fixtures()

    tests = [
        test_model_registry_lists_deepfilternet3,
        test_manifest_validation,
        test_smoke_case_selection,
        test_resample_mixture_for_enhancer,
        test_delay_compensation_configuration,
        test_offline_and_streaming_receive_identical_noisy_input,
        test_smoke_benchmark_with_mock_enhancer,
        test_streaming_flush_length_with_mock_enhancer,
        test_result_serialization,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    if os.environ.get("SIH26_INTEGRATION") == "1":
        test_df3_smoke_benchmark_integration()
        print("PASS: test_df3_smoke_benchmark_integration")
    else:
        print("SKIP: test_df3_smoke_benchmark_integration")

    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
