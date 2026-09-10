"""Recording-safe SIH-26 evaluation for fine-tuned vs pretrained DF3.

The teammate artifact does not ship a train/validation file list, so this
module never claims the selected clips are held out from fine-tuning unless
an explicit training-source manifest is provided.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from drdo_anc.benchmark.case import BenchmarkCase
from drdo_anc.benchmark.evaluation_manifest import EvaluationManifest
from drdo_anc.classification.source_identity import recording_source_id
from drdo_anc.dataset.manifest import (
    SIH26_REPO_ID,
    load_metadata_rows,
    row_to_source_sample,
)
from drdo_anc.dataset.source_pool import (
    DEVELOPMENT_NOISE_CATEGORIES,
    DEVELOPMENT_NUM_CLEAN_SPEAKERS,
    DEVELOPMENT_SNR_LEVELS_DB,
    ENGLISH_SOURCE,
    NOISE_CATEGORY_SOURCES,
    derive_mixing_seed,
    english_speaker_id,
    is_clean_source,
    is_noise_source,
    source_sample_id_from_row,
)
from drdo_anc.dataset.source_sample import SourceSample
from drdo_anc.enhancement.finetuned import (
    project_root,
    resolve_finetuned_artifact_root,
)


HELD_OUT_RULES_VERSION = "sih26-finetuned-recording-safe-v1"
HELD_OUT_SPLIT_NAME = "recording_safe_independent_eval"
HELD_OUT_SELECTION_SEED = 43
EXPERIMENT_VERSION = "dfn3-finetuned-recording-safe-v1"
OPTIONAL_TRAINING_MANIFEST_NAMES = (
    "training_sources.json",
    "train_files.json",
    "training_file_list.json",
)


@dataclass(frozen=True)
class TrainingProvenance:
    holdout_status: str
    holdout_claim: bool
    reasons: tuple[str, ...]
    blocked_clean_sample_ids: frozenset[str]
    blocked_noise_sample_ids: frozenset[str]
    blocked_noise_recording_ids: frozenset[str]
    blocked_speaker_ids: frozenset[str]
    training_manifest_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "training_holdout_status": self.holdout_status,
            "training_holdout_claim": self.holdout_claim,
            "reasons": list(self.reasons),
            "training_manifest_path": self.training_manifest_path,
            "num_blocked_clean_sample_ids": len(self.blocked_clean_sample_ids),
            "num_blocked_noise_sample_ids": len(self.blocked_noise_sample_ids),
            "num_blocked_noise_recording_ids": len(
                self.blocked_noise_recording_ids
            ),
            "num_blocked_speaker_ids": len(self.blocked_speaker_ids),
        }


@dataclass(frozen=True)
class DevelopmentBlocklist:
    speaker_ids: frozenset[str]
    clean_sample_ids: frozenset[str]
    noise_sample_ids: frozenset[str]
    noise_recording_ids: frozenset[str]


def inspect_training_provenance(
    artifact_root: Path | None = None,
    training_manifest_path: Path | None = None,
) -> TrainingProvenance:
    """Inspect whether fine-tune training sources can be identified."""

    reasons: list[str] = []
    resolved_manifest: Path | None = training_manifest_path

    try:
        root = resolve_finetuned_artifact_root(artifact_root)
    except FileNotFoundError as exc:
        reasons.append(str(exc))
        root = None

    if root is not None:
        mvp = root / "data" / "mvp" / "finetune" / "dfn3-custom"
        if not mvp.exists():
            reasons.append(
                "Fine-tune working directory data/mvp/finetune/dfn3-custom "
                "is not present in the extracted artifact "
                "(convert-onnx.py refers to it, but it was not shipped)."
            )
        if resolved_manifest is None:
            for name in OPTIONAL_TRAINING_MANIFEST_NAMES:
                candidate = root / name
                if candidate.exists():
                    resolved_manifest = candidate
                    break

    checkpoint = None
    if root is not None:
        checkpoint = (
            root
            / "models"
            / "dfn3-epoch-130-onnx"
            / "_export_model"
            / "checkpoints"
            / "model_130.ckpt"
        )
        if checkpoint.exists():
            reasons.append(
                "Checkpoint is a PyTorch state_dict (weights only); "
                "it contains no dataset file list."
            )

    config = None
    if root is not None:
        config = (
            root
            / "models"
            / "dfn3-epoch-130-onnx"
            / "_export_model"
            / "config.ini"
        )
        if config.exists():
            reasons.append(
                "Export config.ini has training hyperparameters "
                "(seed=43, max_sample_len_s=3.0, dataloader_snrs) "
                "but no source-file paths."
            )

    if resolved_manifest is None:
        reasons.append(
            "No training_sources.json / train file list was found next to "
            "the artifact, so clips cannot be proven unused in fine-tuning."
        )
        return TrainingProvenance(
            holdout_status="unverified",
            holdout_claim=False,
            reasons=tuple(reasons),
            blocked_clean_sample_ids=frozenset(),
            blocked_noise_sample_ids=frozenset(),
            blocked_noise_recording_ids=frozenset(),
            blocked_speaker_ids=frozenset(),
            training_manifest_path=None,
        )

    payload = json.loads(
        Path(resolved_manifest).read_text(encoding="utf-8")
    )
    return TrainingProvenance(
        holdout_status="excluded_listed_training_sources",
        holdout_claim=True,
        reasons=tuple(
            reasons
            + [
                f"Loaded explicit training-source manifest: {resolved_manifest}"
            ]
        ),
        blocked_clean_sample_ids=frozenset(
            payload.get("clean_sample_ids", [])
        ),
        blocked_noise_sample_ids=frozenset(
            payload.get("noise_sample_ids", [])
        ),
        blocked_noise_recording_ids=frozenset(
            payload.get("noise_recording_ids", [])
        ),
        blocked_speaker_ids=frozenset(payload.get("speaker_ids", [])),
        training_manifest_path=str(Path(resolved_manifest).resolve()),
    )


def development_blocklist(
    development_manifest: EvaluationManifest,
) -> DevelopmentBlocklist:
    speakers: set[str] = set()
    clean_ids: set[str] = set()
    noise_ids: set[str] = set()
    noise_recordings: set[str] = set()

    for case in development_manifest.cases:
        speakers.add(english_speaker_id(case.clean_source))
        clean_ids.add(case.clean_source.sample_id)
        noise_ids.add(case.noise_source.sample_id)
        noise_recordings.add(
            recording_source_id(
                case.noise_source,
                case.noise_category,
            )
        )

    return DevelopmentBlocklist(
        speaker_ids=frozenset(speakers),
        clean_sample_ids=frozenset(clean_ids),
        noise_sample_ids=frozenset(noise_ids),
        noise_recording_ids=frozenset(noise_recordings),
    )


def _select_english_clean_sources(
    clean_rows: list[dict[str, str]],
    num_speakers: int,
    blocked_speakers: set[str],
    blocked_sample_ids: set[str],
) -> list[SourceSample]:
    english_rows = [
        row for row in clean_rows if row["dataset_source"] == ENGLISH_SOURCE
    ]
    by_speaker: dict[str, list[dict[str, str]]] = {}
    for row in english_rows:
        speaker = english_speaker_id(row)
        by_speaker.setdefault(speaker, []).append(row)

    selected: list[SourceSample] = []
    for speaker in sorted(by_speaker):
        if speaker in blocked_speakers:
            continue
        rows = sorted(by_speaker[speaker], key=source_sample_id_from_row)
        chosen = None
        for row in rows:
            sample = row_to_source_sample(row)
            if sample.sample_id in blocked_sample_ids:
                continue
            chosen = sample
            break
        if chosen is None:
            continue
        selected.append(chosen)
        if len(selected) >= num_speakers:
            break

    if len(selected) < num_speakers:
        raise ValueError(
            f"Requested {num_speakers} English speakers disjoint from the "
            f"blocked set, but only found {len(selected)}."
        )
    return selected


def _select_noise_source(
    noise_rows: list[dict[str, str]],
    category: str,
    clean_index: int,
    category_index: int,
    blocked_recording_ids: set[str],
    blocked_sample_ids: set[str],
) -> SourceSample:
    by_subclass: dict[str, list[dict[str, str]]] = {}
    for row in noise_rows:
        by_subclass.setdefault(row["inferred_subclass"], []).append(row)

    subclasses = sorted(by_subclass)
    if not subclasses:
        raise ValueError(f"No noise rows available for {category}.")

    subclass = subclasses[(clean_index + category_index) % len(subclasses)]
    rows = sorted(by_subclass[subclass], key=source_sample_id_from_row)
    for row in rows:
        sample = row_to_source_sample(row)
        recording_id = recording_source_id(sample, category)
        if recording_id in blocked_recording_ids:
            continue
        if sample.sample_id in blocked_sample_ids:
            continue
        return sample

    raise ValueError(
        f"No unblocked {category} noise source in subclass {subclass!r}."
    )


def assert_recording_disjoint(
    development_manifest: EvaluationManifest,
    evaluation_manifest: EvaluationManifest,
) -> None:
    blocked = development_blocklist(development_manifest)
    speakers = {
        english_speaker_id(case.clean_source)
        for case in evaluation_manifest.cases
    }
    noise_recordings = {
        recording_source_id(case.noise_source, case.noise_category)
        for case in evaluation_manifest.cases
    }
    clean_ids = {
        case.clean_source.sample_id for case in evaluation_manifest.cases
    }
    noise_ids = {
        case.noise_source.sample_id for case in evaluation_manifest.cases
    }

    speaker_overlap = speakers & blocked.speaker_ids
    noise_rec_overlap = noise_recordings & blocked.noise_recording_ids
    clean_overlap = clean_ids & blocked.clean_sample_ids
    noise_overlap = noise_ids & blocked.noise_sample_ids
    if speaker_overlap or noise_rec_overlap or clean_overlap or noise_overlap:
        raise ValueError(
            "Recording-safe evaluation overlaps the 60-case development "
            f"set: speakers={sorted(speaker_overlap)}, "
            f"noise_recordings={sorted(noise_rec_overlap)}, "
            f"clean_ids={sorted(clean_overlap)}, "
            f"noise_ids={sorted(noise_overlap)}."
        )


def build_recording_safe_manifest(
    metadata_path: Path,
    development_manifest: EvaluationManifest,
    *,
    provenance: TrainingProvenance | None = None,
    dataset_revision: str | None = None,
) -> EvaluationManifest:
    """Build an independent SIH-26 eval disjoint from the 60-case development set."""

    if provenance is None:
        provenance = inspect_training_provenance()

    blocked = development_blocklist(development_manifest)
    blocked_speakers = set(blocked.speaker_ids) | set(provenance.blocked_speaker_ids)
    blocked_clean = set(blocked.clean_sample_ids) | set(
        provenance.blocked_clean_sample_ids
    )
    blocked_noise_ids = set(blocked.noise_sample_ids) | set(
        provenance.blocked_noise_sample_ids
    )
    blocked_noise_rec = set(blocked.noise_recording_ids) | set(
        provenance.blocked_noise_recording_ids
    )

    rows = load_metadata_rows(metadata_path)
    clean_rows = sorted(
        [row for row in rows if is_clean_source(row)],
        key=source_sample_id_from_row,
    )
    noise_rows = sorted(
        [row for row in rows if is_noise_source(row)],
        key=source_sample_id_from_row,
    )

    clean_sources = _select_english_clean_sources(
        clean_rows,
        DEVELOPMENT_NUM_CLEAN_SPEAKERS,
        blocked_speakers,
        blocked_clean,
    )

    cases: list[BenchmarkCase] = []
    case_counter = 0
    for clean_index, clean_source in enumerate(clean_sources):
        for category_index, category in enumerate(DEVELOPMENT_NOISE_CATEGORIES):
            dataset_source = NOISE_CATEGORY_SOURCES[category]
            category_rows = [
                row
                for row in noise_rows
                if row["dataset_source"] == dataset_source
            ]
            noise_source = _select_noise_source(
                category_rows,
                category,
                clean_index,
                category_index,
                blocked_noise_rec,
                blocked_noise_ids,
            )
            for snr_db in DEVELOPMENT_SNR_LEVELS_DB:
                case_counter += 1
                case_id = (
                    f"recording_safe_{HELD_OUT_RULES_VERSION}_"
                    f"{case_counter:05d}"
                )
                cases.append(
                    BenchmarkCase(
                        case_id=case_id,
                        clean_source=clean_source,
                        noise_source=noise_source,
                        noise_category=category,
                        snr_db=float(snr_db),
                        mixing_seed=derive_mixing_seed(case_id),
                    )
                )

    manifest = EvaluationManifest(
        dataset_repo_id=SIH26_REPO_ID,
        dataset_revision=dataset_revision,
        selection_seed=HELD_OUT_SELECTION_SEED,
        rules_version=HELD_OUT_RULES_VERSION,
        split_name=HELD_OUT_SPLIT_NAME,
        snr_levels_db=DEVELOPMENT_SNR_LEVELS_DB,
        noise_categories=DEVELOPMENT_NOISE_CATEGORIES,
        cases=tuple(cases),
    )
    assert_recording_disjoint(development_manifest, manifest)
    return manifest


def timing_summary(report: dict[str, Any]) -> dict[str, float]:
    rows = [
        row
        for row in report.get("case_results", [])
        if row.get("status") == "success"
    ]
    inference = [
        float(row["inference_s"])
        for row in rows
        if row.get("inference_s") is not None
    ]
    rtf = [float(row["rtf"]) for row in rows if row.get("rtf") is not None]
    from statistics import mean, median

    result: dict[str, float] = {}
    if inference:
        result["mean_inference_s"] = mean(inference)
        result["median_inference_s"] = median(inference)
    if rtf:
        result["mean_rtf"] = mean(rtf)
        result["median_rtf"] = median(rtf)
    return result


def default_output_dir() -> Path:
    return (
        project_root()
        / "data"
        / "benchmark_results"
        / "dfn3_finetuned_recording_safe"
    )
