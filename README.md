# DRDO-ANC

DRDO-ANC is a robust, end-to-end framework for Active Noise Cancellation (ANC) model evaluation, testing, and real-time visualization. It provides a deterministic offline benchmark pipeline and a low-latency real-time telemetry GUI, specifically designed for integrating and testing models like **DeepFilterNet3** and custom fine-tuned ANC architectures.

---

## 🚀 Key Features

* **Deterministic Offline Benchmarking**: A complete pipeline that downloads datasets (via Hugging Face manifests), generates clean/noise mixtures, resamples audio at model boundaries (e.g. 16kHz to 48kHz), and computes standardized performance metrics.
* **Extensible Model Registry**: Easily register new fine-tuned ANC models. The registry ensures fair benchmarking across different architectures by handling per-model streaming delays transparently.
* **Real-Time PySide6 + QML GUI**: A "Dark Informative Telemetry" dashboard that visualizes real-time performance. Built on a fully decoupled architecture, the GUI visualizes live audio oscilloscopes, multi-stop LED volume meters, and history sparklines without ever blocking the critical audio processing thread.
* **Low-Latency Streaming Pipeline**: A dedicated background daemon thread captures live audio from your hardware (e.g., dual-microphones) via `sounddevice`, processes it through the active ANC model frame-by-frame, and outputs to your speakers in real-time.

---

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SIH-2026-27/DRDO-ANC.git
   cd DRDO-ANC
   ```

2. **Install core dependencies:**
   The framework requires `numpy`, `torch`, `sounddevice`, `soundfile`, and `PySide6`. 
   ```bash
   pip install -e .
   pip install PySide6 sounddevice soundfile numpy torch
   ```

3. **Install DeepFilterNet:**
   To use the default provided enhancer model:
   ```bash
   pip install deepfilternet
   ```

---

## 🖥️ Usage

### Running the Real-Time GUI

The GUI allows you to monitor the active noise cancellation happening on your live hardware microphones.

1. **Production Mode (Live ANC Enhancement):**
   Runs your primary microphone through the DeepFilterNet3 model and visualizes the telemetry in real-time.
   ```bash
   python scripts/run_live_gui.py --model DeepFilterNet3
   ```

2. **Pass-through Mode (Hardware Diagnostic):**
   Routes your microphone directly to your speakers without applying ML models. Use this to measure raw I/O latency and test the Qt visualizer.
   ```bash
   python scripts/run_live_gui.py --passthrough
   ```

### Running Benchmarks (Offline)

To benchmark registered models against standard datasets:
```bash
python scripts/test_live_audio.py
```
*Note: Refer to `PROJECT_STATUS.md` for deep technical details on the evaluation pipeline and manifest generation.*

---

## 🏗️ Architecture Overview

The system strictly decouples the audio-critical path from the visualization path:

1. **Audio Daemon Thread**: Uses a `StreamingPipeline` to ingest arbitrary-sized audio chunks, buffers them into strict frames for the model (e.g., DeepFilterNet native Rust backend), and writes the enhanced audio to the speaker buffer.
2. **Telemetry Callbacks**: When a chunk finishes processing, non-blocking telemetry data (Latency, RTF, Peak RMS, Dropped frames) is immediately fired to a thread-safe property bridge.
3. **Main Qt Thread**: Runs a 60 FPS `QTimer` that consumes the latest available telemetry snapshot and pushes it to the GPU-accelerated QML Canvas for rendering. **The audio pipeline never waits for a GUI frame.**

---

## 📄 Documentation

For full details on the development process, setup nuances, and historical architectural decisions, refer to:
- [`PROJECT_STATUS.md`](./PROJECT_STATUS.md): Current project state and roadmap.
- [`implementation_plan.md`](./implementation_plan.md): The team execution runbook and dependency setup guide.
- [`implementation_report.md`](./implementation_report.md): The technical report on the decoupled GUI design.

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.