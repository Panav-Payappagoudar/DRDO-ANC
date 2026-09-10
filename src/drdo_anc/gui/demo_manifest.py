from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

SCENARIO_FILE = Path(__file__).with_name("demo_scenarios.json")

MIN_DURATION_S = 1.0
MIN_RMS = 0.005
MIN_PEAK = 0.02
EXPECTED_CHANNELS = 1
MIN_AUDIBLE_NOISE_RMS = 0.015
SNR_TOLERANCE_DB = 2.0


class DemoManifestError(Exception):
    """Raised when a demo scenario or asset fails validation."""


@dataclass(frozen=True)
class DemoScenario:
    id: str
    label: str
    wav_path: Path
    clean_reference_path: Path | None = None
    enhanced_wav_path: Path | None = None
    enhanced_playback: str = "live"
    noise_type: str | None = None
    target_snr_db: float | None = None


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ValidatedAudioAsset:
    path: Path
    sample_rate: int
    num_samples: int
    duration_s: float
    rms: float
    peak: float


@dataclass(frozen=True)
class ValidatedDemoCatalog:
    sample_rate: int
    scenarios: tuple[DemoScenario, ...]


def _inspect_wav(path: Path) -> ValidatedAudioAsset:
    if not path.is_file():
        raise DemoManifestError(f"Demo asset not found: {path}")

    try:
        info = sf.info(path)
        audio, sample_rate = sf.read(path, dtype="float32")
    except Exception as exc:
        raise DemoManifestError(
            f"Demo asset is not a readable WAV: {path} ({exc})"
        ) from exc

    if info.channels != EXPECTED_CHANNELS:
        raise DemoManifestError(
            f"Demo asset must be mono ({EXPECTED_CHANNELS} channel): "
            f"{path} has {info.channels} channels"
        )

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    if audio.size == 0:
        raise DemoManifestError(f"Demo asset is empty: {path}")

    if not np.isfinite(audio).all():
        raise DemoManifestError(f"Demo asset contains NaN/Inf samples: {path}")

    duration_s = audio.size / sample_rate
    if duration_s < MIN_DURATION_S:
        raise DemoManifestError(
            f"Demo asset too short ({duration_s:.2f}s < {MIN_DURATION_S}s): {path}"
        )

    rms = float(np.sqrt(np.mean(audio**2)))
    peak = float(np.max(np.abs(audio)))

    if rms < MIN_RMS:
        raise DemoManifestError(
            f"Demo asset has insufficient speech energy (RMS={rms:.6f}): {path}"
        )

    if peak < MIN_PEAK:
        raise DemoManifestError(
            f"Demo asset peak level too low (peak={peak:.6f}): {path}"
        )

    return ValidatedAudioAsset(
        path=path,
        sample_rate=int(sample_rate),
        num_samples=int(audio.size),
        duration_s=duration_s,
        rms=rms,
        peak=peak,
    )


def _validate_training_triplet(
    entry: dict,
    *,
    root: Path,
    noisy_path: Path,
    input_asset: ValidatedAudioAsset,
) -> tuple[Path | None, Path | None]:
    scenario_id = str(entry["id"])
    clean_rel = entry.get("clean_reference")
    enhanced_rel = entry.get("enhanced_wav")
    target_snr = entry.get("target_snr_db")

    clean_path: Path | None = None
    enhanced_path: Path | None = None

    if clean_rel:
        clean_path = (root / str(clean_rel)).resolve()
        clean_asset = _inspect_wav(clean_path)
        if clean_asset.sample_rate != input_asset.sample_rate:
            raise DemoManifestError(
                f"Scenario '{scenario_id}' clean reference sample rate "
                f"mismatch: {clean_path}"
            )
        if clean_asset.num_samples != input_asset.num_samples:
            raise DemoManifestError(
                f"Scenario '{scenario_id}' clean reference length "
                f"({clean_asset.num_samples}) does not match noisy input "
                f"({input_asset.num_samples}): {clean_path}"
            )

    if enhanced_rel:
        enhanced_path = (root / str(enhanced_rel)).resolve()
        enhanced_asset = _inspect_wav(enhanced_path)
        if enhanced_asset.sample_rate != input_asset.sample_rate:
            raise DemoManifestError(
                f"Scenario '{scenario_id}' enhanced reference sample rate "
                f"mismatch: {enhanced_path}"
            )
        if enhanced_asset.num_samples != input_asset.num_samples:
            raise DemoManifestError(
                f"Scenario '{scenario_id}' enhanced reference length "
                f"({enhanced_asset.num_samples}) does not match noisy input "
                f"({input_asset.num_samples}): {enhanced_path}"
            )

    if clean_path is not None and target_snr is not None:
        noisy_audio, _ = sf.read(noisy_path, dtype="float32")
        clean_audio, _ = sf.read(clean_path, dtype="float32")
        noisy_audio = np.asarray(noisy_audio, dtype=np.float32).reshape(-1)
        clean_audio = np.asarray(clean_audio, dtype=np.float32).reshape(-1)
        noise_estimate = noisy_audio - clean_audio
        clean_power = float(np.mean(clean_audio**2))
        noise_power = float(np.mean(noise_estimate**2))

        if noise_power > 0 and clean_power > 0:
            achieved_snr = 10.0 * np.log10(clean_power / noise_power)
            delta = abs(achieved_snr - float(target_snr))

            if delta > SNR_TOLERANCE_DB:
                raise DemoManifestError(
                    f"Scenario '{scenario_id}' noisy/clean SNR "
                    f"({achieved_snr:.1f} dB) differs from documented target "
                    f"({float(target_snr):.1f} dB) by more than "
                    f"{SNR_TOLERANCE_DB:.1f} dB: {noisy_path}"
                )

    playback_mode = str(entry.get("enhanced_playback", "live"))
    if playback_mode not in {"live", "reference"}:
        raise DemoManifestError(
            f"Scenario '{scenario_id}' has unsupported enhanced_playback "
            f"mode '{playback_mode}'."
        )

    return clean_path, enhanced_path


def _validate_scenario_entry(
    entry: dict,
    *,
    root: Path,
    expected_sample_rate: int,
) -> DemoScenario:
    scenario_id = str(entry["id"])
    label = str(entry["label"])
    wav_rel = str(entry["wav"])
    wav_path = (root / wav_rel).resolve()

    input_asset = _inspect_wav(wav_path)
    if input_asset.sample_rate != expected_sample_rate:
        raise DemoManifestError(
            f"Scenario '{scenario_id}' input sample rate "
            f"({input_asset.sample_rate} Hz) does not match manifest "
            f"({expected_sample_rate} Hz): {wav_path}"
        )

    enhanced_path: Path | None = None
    enhanced_rel = entry.get("enhanced_wav")

    clean_path, enhanced_path = _validate_training_triplet(
        entry,
        root=root,
        noisy_path=wav_path,
        input_asset=input_asset,
    )

    source = str(entry.get("source", "file"))
    if source not in {"file"}:
        raise DemoManifestError(
            f"Scenario '{scenario_id}' uses unsupported source kind '{source}'. "
            "Only explicit WAV files are allowed in demo mode."
        )

    return DemoScenario(
        id=scenario_id,
        label=label,
        wav_path=wav_path,
        clean_reference_path=clean_path,
        enhanced_wav_path=enhanced_path,
        enhanced_playback=str(entry.get("enhanced_playback", "live")),
        noise_type=entry.get("noise_type"),
        target_snr_db=(
            float(entry["target_snr_db"])
            if entry.get("target_snr_db") is not None
            else None
        ),
    )


def compute_demo_reference_metrics(
    scenario: DemoScenario,
    *,
    delay_samples: int = 0,
) -> dict[str, float]:
    """Offline metrics for noisy/enhanced reference WAVs vs clean reference.

    Uses ``delay_samples=0`` by default because ``train_enh_snr5.wav`` is an
    offline/batch DF3 reference aligned sample-for-sample with the noisy clip.
    """

    if scenario.clean_reference_path is None:
        raise DemoManifestError(
            f"Scenario '{scenario.id}' has no clean reference for metrics."
        )

    if scenario.enhanced_wav_path is None:
        raise DemoManifestError(
            f"Scenario '{scenario.id}' has no enhanced reference for metrics."
        )

    from drdo_anc.audio.io import load_mono_wav
    from drdo_anc.evaluation.metrics import evaluate_model

    clean, sample_rate = load_mono_wav(scenario.clean_reference_path)
    noisy, noisy_sr = load_mono_wav(scenario.wav_path)
    enhanced, enhanced_sr = load_mono_wav(scenario.enhanced_wav_path)

    if noisy_sr != sample_rate or enhanced_sr != sample_rate:
        raise DemoManifestError(
            f"Scenario '{scenario.id}' reference metrics require a common "
            "sample rate across clean/noisy/enhanced assets."
        )

    return evaluate_model(
        clean,
        noisy,
        enhanced,
        sample_rate,
        delay_samples=delay_samples,
    )


def load_validated_demo_catalog(
    manifest_path: Path | None = None,
) -> ValidatedDemoCatalog:
    """Load and validate every demo scenario from the manifest.

    Raises ``DemoManifestError`` on the first invalid scenario. Does not
    substitute or skip entries.
    """

    path = manifest_path or SCENARIO_FILE
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_sample_rate = int(payload.get("sample_rate", 48_000))
    root = project_root()

    scenarios: list[DemoScenario] = []
    for entry in payload.get("scenarios", []):
        scenarios.append(
            _validate_scenario_entry(
                entry,
                root=root,
                expected_sample_rate=expected_sample_rate,
            )
        )

    if not scenarios:
        raise DemoManifestError(f"No demo scenarios defined in {path}")

    return ValidatedDemoCatalog(
        sample_rate=expected_sample_rate,
        scenarios=tuple(scenarios),
    )


def get_scenario_by_index(
    catalog: ValidatedDemoCatalog,
    index: int,
) -> DemoScenario:
    if index < 0 or index >= len(catalog.scenarios):
        raise DemoManifestError(
            f"Demo scenario index {index} is out of range "
            f"(0..{len(catalog.scenarios) - 1})."
        )

    return catalog.scenarios[index]
