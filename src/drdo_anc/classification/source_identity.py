"""Recording-level source identity for leakage-safe dataset splits."""

from __future__ import annotations

import re

from drdo_anc.dataset.source_sample import SourceSample


_UUID_RE = re.compile(
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
    re.IGNORECASE,
)
_ESC50_RE = re.compile(r"^(\d+)-(\d+)-([A-Z])-(\d+)\.wav$")


def recording_source_id(
    source: SourceSample,
    category: str,
) -> str:
    """
    Return a stable recording identity for split grouping.

    Rules (derived from SIH-26 filename conventions):

    * ``vehicle_engine`` (ESC-50): group takes ``A/B/...`` of the same clip ID
      together (``{fold}-{clip}-{take}-{class}.wav`` → ``esc50_clip_{clip}``).
    * ``impulsive_firearms``: when a UUID is present, group channel/version
      variants of that recording; otherwise treat each file as its own source.
    * ``uav_drone``: each WAV is a distinct clip/source.

    If a category has no reliable multi-window grouping cue, the full
    ``sample_id`` is used so random window splitting is never invented.
    """

    if category == "vehicle_engine":
        match = _ESC50_RE.match(source.filename)
        if match is not None:
            return f"esc50_clip_{match.group(2)}"

    if category == "impulsive_firearms":
        match = _UUID_RE.search(source.filename)
        if match is not None:
            return f"firearm_rec_{match.group(1).lower()}"

    return f"{category}:{source.sample_id}"
