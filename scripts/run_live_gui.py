#!/usr/bin/env python3

import argparse

import os

import sys

import threading

import traceback

from pathlib import Path



src_dir = Path(__file__).resolve().parent.parent / "src"

sys.path.insert(0, os.fspath(src_dir))



from drdo_anc.audio.live import (

  StreamingPipeline,

  close_sounddevice_io,

  format_device_listing,

  open_sounddevice_io,

)

from drdo_anc.gui.app import run_gui

from drdo_anc.gui.bridge import GUIBridge

from drdo_anc.gui.session import ApplicationSession



DEFAULT_MODEL_NAME = "DeepFilterNet3"

DEFAULT_READ_CHUNK_SIZE = 1024





def _parse_device(value: str | None) -> int | str | None:

  if value is None:

    return None



  try:

    return int(value)

  except ValueError:

    return value





def _build_parser() -> argparse.ArgumentParser:

  parser = argparse.ArgumentParser(description="DRDO-ANC Real-Time GUI")

  parser.add_argument(

    "--list-devices",

    action="store_true",

    help="List host audio devices and exit.",

  )

  parser.add_argument(

    "--model",

    default=DEFAULT_MODEL_NAME,

    help="Registered enhancer model name.",

  )

  parser.add_argument(

    "--passthrough",

    action="store_true",

    help="Copy microphone input directly to the speaker (live mode).",

  )

  parser.add_argument(

    "--sample-rate",

    type=int,

    default=None,

    help="Audio sample rate in Hz.",

  )

  parser.add_argument(

    "--chunk-size",

    type=int,

    default=DEFAULT_READ_CHUNK_SIZE,

    help="Samples requested per AudioInput.read() call.",

  )

  parser.add_argument(

    "--input-device",

    default=None,

    help="Input device index or name.",

  )

  parser.add_argument(

    "--output-device",

    default=None,

    help="Output device index or name.",

  )

  parser.add_argument(

    "--fake",

    action="store_true",

    help="Run animated placeholder visuals only (no pipeline).",

  )

  parser.add_argument(

    "--live-on-start",

    action="store_true",

    help="Start live microphone capture immediately instead of demo mode.",

  )

  return parser





class LiveAudioController:

  """Owns the background live microphone thread and pipeline lifecycle."""



  def __init__(self, args: argparse.Namespace, bridge: GUIBridge) -> None:

    self._args = args

    self._bridge = bridge

    self._pipeline: StreamingPipeline | None = None

    self._audio_input = None

    self._audio_output = None

    self._audio_thread: threading.Thread | None = None

    self._started = False



  def start(self) -> None:

    if self._started:

      return



    self._started = True



    try:

      self._start_audio()

    except Exception as exc:

      self._bridge.set_error(f"Audio startup failed: {exc}")

      self._bridge.set_audio_status("Error")

      print(f"Audio startup failed: {exc}", file=sys.stderr)

      traceback.print_exc()

      self._started = False



  def stop(self) -> None:

    if self._pipeline is not None:

      self._pipeline.request_stop()



    if self._audio_thread is not None and self._audio_thread.is_alive():

      self._audio_thread.join(timeout=5.0)



    if self._audio_input is not None and self._audio_output is not None:

      close_sounddevice_io(self._audio_input, self._audio_output)



    self._pipeline = None

    self._audio_input = None

    self._audio_output = None

    self._audio_thread = None

    self._started = False

    self._bridge.set_audio_status("Stopped")

    self._bridge.set_pipeline_stage("input")



  def _start_audio(self) -> None:

    args = self._args



    if args.passthrough:

      sample_rate = args.sample_rate or 48_000

      enhancer = None

      mode_label = "pass-through"

      model_name = "Pass-Through"

    else:

      from drdo_anc.enhancement import create_enhancer



      enhancer = create_enhancer(args.model)

      sample_rate = args.sample_rate or enhancer.sample_rate()

      mode_label = args.model

      model_name = args.model



    if sample_rate <= 0:

      raise ValueError("--sample-rate must be positive.")



    input_device = _parse_device(args.input_device)

    output_device = _parse_device(args.output_device)



    print("=" * 70)

    print("DRDO-ANC | Live GUI Streaming")

    print("=" * 70)

    print(f"Mode:        {mode_label}")

    print(f"Sample rate: {sample_rate} Hz")

    print(f"Chunk size:  {args.chunk_size} samples/read")

    print(f"Input dev:   {input_device if input_device is not None else 'default'}")

    print(

      f"Output dev:  {output_device if output_device is not None else 'default'}"

    )



    audio_input, audio_output = open_sounddevice_io(

      sample_rate,

      input_device=input_device,

      output_device=output_device,

      blocksize=args.chunk_size,

    )



    self._bridge.set_stream_metadata(

      model_name=model_name,

      sample_rate=sample_rate,

    )

    self._bridge.clear_error()

    self._bridge.set_audio_status("Live")

    self._bridge.set_pipeline_stage("capture")



    def on_telemetry(in_chunk, out_chunk, proc_time):

      stats = getattr(audio_input, "stats", None)

      stats_dict = stats.as_dict() if stats is not None else {}

      self._bridge.publish_data(in_chunk, out_chunk, proc_time, stats=stats_dict)

      self._bridge.set_pipeline_stage("df3")



    pipeline = StreamingPipeline(

      audio_input,

      audio_output,

      enhancer,

      read_chunk_size=args.chunk_size,

      passthrough=args.passthrough,

      telemetry_callback=on_telemetry,

    )



    self._audio_input = audio_input

    self._audio_output = audio_output

    self._pipeline = pipeline



    def run_audio_thread() -> None:

      print("Audio thread starting...")

      try:

        pipeline.run()

      except Exception as exc:

        self._bridge.set_error(f"Audio pipeline error: {exc}")

        self._bridge.set_audio_status("Error")

        print(f"Audio pipeline error: {exc}", file=sys.stderr)

        traceback.print_exc()

      finally:

        close_sounddevice_io(audio_input, audio_output)

        print("Audio thread stopped.")



    audio_thread = threading.Thread(

      target=run_audio_thread,

      name="drdo-anc-audio",

      daemon=False,

    )

    self._audio_thread = audio_thread

    audio_thread.start()





def main() -> None:

  parser = _build_parser()

  args, _unknown_args = parser.parse_known_args()



  if args.list_devices:

    print(format_device_listing())

    return



  bridge = GUIBridge()



  if args.fake:

    print("Starting DRDO-ANC Real-Time GUI (Fake Visual Mode)...")

    bridge.enable_fake_visuals(True)

    run_gui(bridge=bridge)

    return



  live_controller = LiveAudioController(args, bridge)

  session = ApplicationSession(bridge, args, live_controller=live_controller)

  bridge.set_session(session)



  def on_ready() -> None:

    if args.live_on_start:

      session.set_live_mode()



  print("Starting PySide6 GUI (default: Demo Mode)...")

  run_gui(

    bridge=bridge,

    on_ready=on_ready,

    on_shutdown=session.shutdown,

  )





if __name__ == "__main__":

  try:

    main()

  except KeyboardInterrupt:

    sys.exit(0)

