import collections
import math
import threading
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import Property, QObject, QTimer, Signal, Slot

from drdo_anc.gui.telemetry import AudioTelemetry
from drdo_anc.gui.waveform import WaveformProcessor


@dataclass
class _TelemetrySnapshot:
  """Latest-value handoff from the audio thread to the GUI timer."""

  input_chunk: np.ndarray | None = None
  output_chunk: np.ndarray | None = None
  processing_time_s: float = 0.0
  stats: dict[str, float | int] = field(default_factory=dict)
  has_update: bool = False


class GUIBridge(QObject):
  """
  Bridge between background audio threads and the QML frontend.

  The audio thread only stores a small snapshot. Waveform reduction and
  level calculations run on the GUI timer thread.
  """

  telemetryUpdated = Signal()
  inputWaveformUpdated = Signal(list)
  outputWaveformUpdated = Signal(list)
  historyUpdated = Signal()
  errorChanged = Signal()
  demoStateChanged = Signal()

  def __init__(self, fps: int = 60) -> None:
    super().__init__()
    self._fps = fps
    self._timer = QTimer(self)
    self._timer.timeout.connect(self._on_timeout)

    self._latest_telemetry = AudioTelemetry()
    self._error_message = ""

    self._waveform_processor = WaveformProcessor(target_points=500)
    self._latest_input_waveform: list[float] = []
    self._latest_output_waveform: list[float] = []

    self._history_len = 100
    self._proc_time_hist = collections.deque(
      [0.0] * self._history_len,
      maxlen=self._history_len,
    )
    self._buffer_fill_hist = collections.deque(
      [0.0] * self._history_len,
      maxlen=self._history_len,
    )
    self._dropped_hist = collections.deque(
      [0.0] * self._history_len,
      maxlen=self._history_len,
    )
    self._rtf_hist = collections.deque(
      [0.0] * self._history_len,
      maxlen=self._history_len,
    )

    self._phase = 0.0
    self._telemetry_lock = threading.Lock()
    self._pending_snapshot = _TelemetrySnapshot()
    self._session = None
    self._use_fake_visuals = False

    self._operation_mode = "demo"
    self._playback_state = "stopped"
    self._demo_scenario = "Speech Only"
    self._selected_scenario_index = 0
    self._scenario_labels: list[str] = []
    self._ab_mode = "raw"
    self._pipeline_stage = "input"
    self._audio_status = "Ready"
    self._development_cases = -1
    self._evaluations = -1
    self._demo_clean_file = ""
    self._demo_noisy_file = ""
    self._demo_enhanced_ref_file = ""
    self._demo_enhanced_playback = "Live DF3"
    self._demo_input_label = "NOISY INPUT"
    self._demo_output_label = "LIVE DF3 ENHANCED"
    self._demo_metrics_available = False
    self._demo_noisy_snr = 0.0
    self._demo_enhanced_ref_snr = 0.0
    self._demo_noisy_si_sdr = 0.0
    self._demo_enhanced_ref_si_sdr = 0.0
    self._demo_noisy_stoi = 0.0
    self._demo_enhanced_ref_stoi = 0.0
    self._demo_noisy_pesq = 0.0
    self._demo_enhanced_ref_pesq = 0.0

  def start_timer(self) -> None:
    interval = int(1000 / self._fps)
    self._timer.start(interval)

  def stop_timer(self) -> None:
    self._timer.stop()

  @Property(float, notify=telemetryUpdated)
  def inputLevelDb(self) -> float:
    return self._latest_telemetry.input_level_db

  @Property(float, notify=telemetryUpdated)
  def outputLevelDb(self) -> float:
    return self._latest_telemetry.output_level_db

  @Property(float, notify=telemetryUpdated)
  def inputPeakDb(self) -> float:
    return self._latest_telemetry.input_peak_db

  @Property(float, notify=telemetryUpdated)
  def outputPeakDb(self) -> float:
    return self._latest_telemetry.output_peak_db

  @Property(float, notify=telemetryUpdated)
  def processingTimeMs(self) -> float:
    return self._latest_telemetry.processing_time_ms

  @Property(float, notify=telemetryUpdated)
  def realtimeFactor(self) -> float:
    return self._latest_telemetry.realtime_factor

  @Property(float, notify=telemetryUpdated)
  def bufferFillPercent(self) -> float:
    return self._latest_telemetry.buffer_fill_percent

  @Property(int, notify=telemetryUpdated)
  def droppedFrames(self) -> int:
    return self._latest_telemetry.dropped_frames

  @Property(str, notify=telemetryUpdated)
  def modelName(self) -> str:
    return self._latest_telemetry.model_name

  @Property(int, notify=telemetryUpdated)
  def sampleRate(self) -> int:
    return self._latest_telemetry.sample_rate

  @Property(bool, notify=telemetryUpdated)
  def isLive(self) -> bool:
    return self._latest_telemetry.is_live

  @Property(str, notify=errorChanged)
  def errorMessage(self) -> str:
    return self._error_message

  @Property(str, notify=demoStateChanged)
  def operationMode(self) -> str:
    return self._operation_mode

  @Property(str, notify=demoStateChanged)
  def playbackState(self) -> str:
    return self._playback_state

  @Property(str, notify=demoStateChanged)
  def demoScenario(self) -> str:
    return self._demo_scenario

  @Property(int, notify=demoStateChanged)
  def selectedScenarioIndex(self) -> int:
    return self._selected_scenario_index

  @Property(list, notify=demoStateChanged)
  def scenarioLabels(self) -> list[str]:
    return list(self._scenario_labels)

  @Property(str, notify=demoStateChanged)
  def abMode(self) -> str:
    return self._ab_mode

  @Property(str, notify=demoStateChanged)
  def pipelineStage(self) -> str:
    return self._pipeline_stage

  @Property(str, notify=demoStateChanged)
  def audioStatus(self) -> str:
    return self._audio_status

  @Property(int, notify=demoStateChanged)
  def developmentCases(self) -> int:
    return self._development_cases

  @Property(int, notify=demoStateChanged)
  def evaluations(self) -> int:
    return self._evaluations

  @Property(bool, notify=demoStateChanged)
  def showBenchmarkSummary(self) -> bool:
    return self._development_cases > 0 and self._evaluations > 0

  @Property(str, notify=demoStateChanged)
  def demoCleanFile(self) -> str:
    return self._demo_clean_file

  @Property(str, notify=demoStateChanged)
  def demoNoisyFile(self) -> str:
    return self._demo_noisy_file

  @Property(str, notify=demoStateChanged)
  def demoEnhancedRefFile(self) -> str:
    return self._demo_enhanced_ref_file

  @Property(str, notify=demoStateChanged)
  def demoEnhancedPlayback(self) -> str:
    return self._demo_enhanced_playback

  @Property(str, notify=demoStateChanged)
  def demoInputLabel(self) -> str:
    return self._demo_input_label

  @Property(str, notify=demoStateChanged)
  def demoOutputLabel(self) -> str:
    return self._demo_output_label

  @Property(bool, notify=demoStateChanged)
  def showDemoMetrics(self) -> bool:
    return self._demo_metrics_available

  @Property(float, notify=demoStateChanged)
  def demoNoisySnr(self) -> float:
    return self._demo_noisy_snr

  @Property(float, notify=demoStateChanged)
  def demoEnhancedRefSnr(self) -> float:
    return self._demo_enhanced_ref_snr

  @Property(float, notify=demoStateChanged)
  def demoNoisySiSdr(self) -> float:
    return self._demo_noisy_si_sdr

  @Property(float, notify=demoStateChanged)
  def demoEnhancedRefSiSdr(self) -> float:
    return self._demo_enhanced_ref_si_sdr

  @Property(float, notify=demoStateChanged)
  def demoNoisyStoi(self) -> float:
    return self._demo_noisy_stoi

  @Property(float, notify=demoStateChanged)
  def demoEnhancedRefStoi(self) -> float:
    return self._demo_enhanced_ref_stoi

  @Property(float, notify=demoStateChanged)
  def demoNoisyPesq(self) -> float:
    return self._demo_noisy_pesq

  @Property(float, notify=demoStateChanged)
  def demoEnhancedRefPesq(self) -> float:
    return self._demo_enhanced_ref_pesq

  def set_demo_assets(
    self,
    *,
    clean_file: str,
    noisy_file: str,
    enhanced_ref_file: str,
    enhanced_playback: str,
  ) -> None:
    self._demo_clean_file = clean_file
    self._demo_noisy_file = noisy_file
    self._demo_enhanced_ref_file = enhanced_ref_file
    self._demo_enhanced_playback = enhanced_playback
    self._demo_input_label = f"NOISY INPUT ({noisy_file})"
    self._demo_output_label = f"{enhanced_playback} OUTPUT"
    self.demoStateChanged.emit()

  def set_demo_reference_metrics(self, metrics: dict[str, float]) -> None:
    self._demo_metrics_available = True
    self._demo_noisy_snr = float(metrics.get("noisy_snr", 0.0))
    self._demo_enhanced_ref_snr = float(metrics.get("enhanced_snr", 0.0))
    self._demo_noisy_si_sdr = float(metrics.get("noisy_si_sdr", 0.0))
    self._demo_enhanced_ref_si_sdr = float(metrics.get("enhanced_si_sdr", 0.0))
    self._demo_noisy_stoi = float(metrics.get("noisy_stoi", 0.0))
    self._demo_enhanced_ref_stoi = float(metrics.get("enhanced_stoi", 0.0))
    self._demo_noisy_pesq = float(metrics.get("noisy_pesq", 0.0))
    self._demo_enhanced_ref_pesq = float(metrics.get("enhanced_pesq", 0.0))
    self.demoStateChanged.emit()

  def set_session(self, session) -> None:
    self._session = session

  def enable_fake_visuals(self, enabled: bool = True) -> None:
    self._use_fake_visuals = enabled

  def set_operation_mode(self, mode: str) -> None:
    self._operation_mode = mode
    self.demoStateChanged.emit()

  def set_playback_state(self, state: str) -> None:
    self._playback_state = state
    self.demoStateChanged.emit()

  def set_demo_scenario(self, label: str) -> None:
    self._demo_scenario = label
    self.demoStateChanged.emit()

  def set_selected_scenario_index(self, index: int) -> None:
    self._selected_scenario_index = index
    self.demoStateChanged.emit()

  def set_scenario_labels(self, labels: list[str]) -> None:
    self._scenario_labels = list(labels)
    self.demoStateChanged.emit()

  def set_ab_mode(self, mode: str) -> None:
    self._ab_mode = mode
    self.demoStateChanged.emit()

  def set_pipeline_stage(self, stage: str) -> None:
    self._pipeline_stage = stage
    self.demoStateChanged.emit()

  def set_audio_status(self, status: str) -> None:
    self._audio_status = status
    self.demoStateChanged.emit()

  def set_benchmark_summary(
    self,
    development_cases: int | None,
    evaluations: int | None,
  ) -> None:
    self._development_cases = development_cases if development_cases else -1
    self._evaluations = evaluations if evaluations else -1
    self.demoStateChanged.emit()

  @Slot()
  def setDemoMode(self) -> None:
    if self._session is not None:
      self._session.set_demo_mode()

  @Slot()
  def setLiveMode(self) -> None:
    if self._session is not None:
      self._session.set_live_mode()

  @Slot()
  def play(self) -> None:
    if self._session is not None:
      self._session.play()

  @Slot()
  def pause(self) -> None:
    if self._session is not None:
      self._session.pause()

  @Slot()
  def stop(self) -> None:
    if self._session is not None:
      self._session.stop()

  @Slot(int)
  def selectScenario(self, index: int) -> None:
    if self._session is not None:
      self._session.set_scenario(index)

  @Slot()
  def selectAbRaw(self) -> None:
    if self._session is not None:
      self._session.set_ab_raw()

  @Slot()
  def selectAbEnhanced(self) -> None:
    if self._session is not None:
      self._session.set_ab_enhanced()

  @Property(list, notify=historyUpdated)
  def procTimeHistory(self) -> list[float]:
    return list(self._proc_time_hist)

  @Property(list, notify=historyUpdated)
  def bufferFillHistory(self) -> list[float]:
    return list(self._buffer_fill_hist)

  @Property(list, notify=historyUpdated)
  def droppedHistory(self) -> list[float]:
    return list(self._dropped_hist)

  @Property(list, notify=historyUpdated)
  def rtfHistory(self) -> list[float]:
    return list(self._rtf_hist)

  def set_error(self, message: str) -> None:
    self._error_message = message
    self.errorChanged.emit()
    self.telemetryUpdated.emit()

  def clear_error(self) -> None:
    if self._error_message:
      self._error_message = ""
      self.errorChanged.emit()

  def set_stream_metadata(self, *, model_name: str, sample_rate: int) -> None:
    self._latest_telemetry.model_name = model_name
    self._latest_telemetry.sample_rate = sample_rate
    self.telemetryUpdated.emit()

  def publish_data(
    self,
    input_chunk: np.ndarray,
    output_chunk: np.ndarray,
    proc_time_s: float,
    stats: dict | None = None,
  ) -> None:
    """Called from the audio thread with the latest chunk snapshot."""

    snapshot = _TelemetrySnapshot(
      input_chunk=(
        np.array(input_chunk, dtype=np.float32, copy=True)
        if len(input_chunk) > 0
        else None
      ),
      output_chunk=(
        np.array(output_chunk, dtype=np.float32, copy=True)
        if len(output_chunk) > 0
        else None
      ),
      processing_time_s=proc_time_s,
      stats=dict(stats) if stats else {},
      has_update=True,
    )

    with self._telemetry_lock:
      self._pending_snapshot = snapshot

  def _consume_snapshot(self) -> _TelemetrySnapshot | None:
    with self._telemetry_lock:
      if not self._pending_snapshot.has_update:
        return None

      snapshot = self._pending_snapshot
      self._pending_snapshot = _TelemetrySnapshot()
      return snapshot

  def _apply_snapshot(self, snapshot: _TelemetrySnapshot) -> None:
    t = self._latest_telemetry
    t.is_live = True
    t.processing_time_ms = snapshot.processing_time_s * 1000.0

    input_chunk = snapshot.input_chunk
    output_chunk = snapshot.output_chunk

    if input_chunk is not None and len(input_chunk) > 0:
      rms_in = float(np.sqrt(np.mean(input_chunk**2) + 1e-10))
      peak_in = float(np.max(np.abs(input_chunk)) + 1e-10)
      t.input_level_db = 20 * math.log10(rms_in)
      t.input_peak_db = 20 * math.log10(peak_in)

    if output_chunk is not None and len(output_chunk) > 0:
      rms_out = float(np.sqrt(np.mean(output_chunk**2) + 1e-10))
      peak_out = float(np.max(np.abs(output_chunk)) + 1e-10)
      t.output_level_db = 20 * math.log10(rms_out)
      t.output_peak_db = 20 * math.log10(peak_out)

    if snapshot.stats:
      t.dropped_frames = int(
        snapshot.stats.get("input_overflows", 0)
        + snapshot.stats.get("output_underflows", 0)
      )

      chunk_samples = len(input_chunk) if input_chunk is not None else 0
      chunk_duration = chunk_samples / max(1, t.sample_rate)
      if chunk_duration > 0:
        t.realtime_factor = snapshot.processing_time_s / chunk_duration
        t.buffer_fill_percent = min(
          100.0,
          max(0.0, t.realtime_factor * 100.0),
        )

    input_reduced, output_reduced = self._waveform_processor.process(
      input_chunk if input_chunk is not None else np.array([], dtype=np.float32),
      output_chunk if output_chunk is not None else np.array([], dtype=np.float32),
    )
    self._latest_input_waveform = input_reduced.tolist()
    self._latest_output_waveform = output_reduced.tolist()

  @Slot()
  def _on_timeout(self) -> None:
    snapshot = self._consume_snapshot()

    if snapshot is not None:
      self._apply_snapshot(snapshot)
    elif self._use_fake_visuals and not self._latest_telemetry.is_live:
      self._generate_fake_data()

    t = self._latest_telemetry
    self._proc_time_hist.append(t.processing_time_ms)
    self._buffer_fill_hist.append(t.buffer_fill_percent)
    self._dropped_hist.append(float(t.dropped_frames))
    self._rtf_hist.append(t.realtime_factor)

    self.telemetryUpdated.emit()
    self.historyUpdated.emit()
    self.inputWaveformUpdated.emit(self._latest_input_waveform)
    self.outputWaveformUpdated.emit(self._latest_output_waveform)

  def _generate_fake_data(self) -> None:
    self._phase += 0.15
    t = self._latest_telemetry

    envelope = max(0, math.sin(self._phase * 0.4) * math.sin(self._phase * 0.13))
    envelope = envelope**2

    base_vol = -50 + (envelope * 45)
    t.input_level_db = base_vol
    t.input_peak_db = min(0, base_vol + 6 + np.random.uniform(0, 4))

    out_vol = base_vol - 2
    t.output_level_db = out_vol
    t.output_peak_db = min(0, out_vol + 4 + np.random.uniform(0, 3))

    t.processing_time_ms = (
      4.1 + math.sin(self._phase * 0.5) * 0.3 + np.random.uniform(0, 0.1)
    )
    t.realtime_factor = (
      0.45 + math.sin(self._phase * 0.3) * 0.02 + np.random.uniform(-0.01, 0.01)
    )
    t.buffer_fill_percent = max(
      0,
      min(100, 15 + math.sin(self._phase * 1.5) * 4 + np.random.uniform(0, 2)),
    )
    t.dropped_frames = 0 if np.random.random() > 0.02 else 1
    t.model_name = "DeepFilterNet3"
    t.sample_rate = 48000

    x = np.linspace(0, 10 * np.pi, 500)
    carrier = (
      np.sin(x * 3.5) + 0.5 * np.sin(x * 7.2) + 0.25 * np.sin(x * 15.1)
    )
    clean_speech = carrier * envelope * 0.8
    noise_envelope = 0.15 + 0.05 * math.sin(self._phase * 0.1)
    noise = np.random.normal(0, noise_envelope, 500)
    noisy_speech = clean_speech + noise

    self._latest_input_waveform = noisy_speech.tolist()
    self._latest_output_waveform = clean_speech.tolist()
