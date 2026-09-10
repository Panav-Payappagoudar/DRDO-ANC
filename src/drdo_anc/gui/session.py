from __future__ import annotations

import argparse

from drdo_anc.gui.bridge import GUIBridge
from drdo_anc.gui.demo import DemoAudioController, load_benchmark_summary
from drdo_anc.gui.demo_manifest import (
    DemoManifestError,
    compute_demo_reference_metrics,
    load_validated_demo_catalog,
)


def _parse_device(value: str | None) -> int | str | None:
    if value is None:
        return None

    try:
        return int(value)
    except ValueError:
        return value


class ApplicationSession:
    """Coordinates demo and live audio controllers for the GUI."""

    def __init__(
        self,
        bridge: GUIBridge,
        args: argparse.Namespace,
        *,
        live_controller,
    ) -> None:
        self._bridge = bridge
        self._args = args
        self._live_controller = live_controller
        self._mode = "demo"

        try:
            catalog = load_validated_demo_catalog()
            self._demo_controller = DemoAudioController(
                bridge,
                model_name=args.model,
                chunk_size=args.chunk_size,
                output_device=_parse_device(args.output_device),
            )
        except DemoManifestError as exc:
            bridge.set_error(f"Demo manifest invalid: {exc}")
            bridge.set_audio_status("Error")
            raise

        labels = [scenario.label for scenario in catalog.scenarios]
        self._bridge.set_scenario_labels(labels)
        self._bridge.set_selected_scenario_index(0)

        scenario = catalog.scenarios[0]
        self._bridge.set_demo_assets(
            clean_file=scenario.clean_reference_path.name
            if scenario.clean_reference_path
            else "",
            noisy_file=scenario.wav_path.name,
            enhanced_ref_file=scenario.enhanced_wav_path.name
            if scenario.enhanced_wav_path
            else "",
            enhanced_playback=(
                "Live DF3"
                if scenario.enhanced_playback == "live"
                else "Offline Reference"
            ),
        )

        if (
            scenario.clean_reference_path is not None
            and scenario.enhanced_wav_path is not None
        ):
            metrics = compute_demo_reference_metrics(scenario)
            self._bridge.set_demo_reference_metrics(metrics)

        dev_cases, evaluations = load_benchmark_summary()
        self._bridge.set_benchmark_summary(dev_cases, evaluations)
        self._bridge.set_operation_mode("demo")
        self._bridge.set_stream_metadata(
            model_name=args.model,
            sample_rate=catalog.sample_rate,
        )
        self._bridge.set_demo_scenario(labels[0])
        self._demo_controller.set_ab_mode("raw")
        self._bridge.set_audio_status("Ready")

    def set_demo_mode(self) -> None:
        if self._mode == "demo":
            return

        self._live_controller.stop()
        self._mode = "demo"
        self._bridge.set_operation_mode("demo")
        self._bridge.set_audio_status("Ready")

    def set_live_mode(self) -> None:
        if self._mode == "live":
            return

        self._demo_controller.stop()
        self._mode = "live"
        self._bridge.set_operation_mode("live")
        self._live_controller.start()
        self._bridge.set_audio_status("Live")

    def play(self) -> None:
        if self._mode == "demo":
            self._demo_controller.play()
        else:
            self._live_controller.start()

    def pause(self) -> None:
        if self._mode == "demo":
            self._demo_controller.pause()

    def stop(self) -> None:
        if self._mode == "demo":
            self._demo_controller.stop()
        else:
            self._live_controller.stop()

    def set_scenario(self, index: int) -> None:
        if self._mode != "demo":
            self.set_demo_mode()

        try:
            self._demo_controller.set_scenario_index(index)
        except DemoManifestError as exc:
            self._bridge.set_error(f"Demo scenario unavailable: {exc}")
            self._bridge.set_audio_status("Error")

    def set_ab_raw(self) -> None:
        self._demo_controller.set_ab_mode("raw")

    def set_ab_enhanced(self) -> None:
        self._demo_controller.set_ab_mode("enhanced")

    def shutdown(self) -> None:
        self._demo_controller.shutdown()
        self._live_controller.stop()
