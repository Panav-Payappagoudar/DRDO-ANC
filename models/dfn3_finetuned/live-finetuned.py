import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch


# Add the DeepFilterNet source directory.
DFN_PATH = r""

# Directory containing config.ini and checkpoints/.
MODEL_DIR = r""

# Smaller values reduce delay but may reduce quality.
BLOCK_SECONDS = 0.02


sys.path.insert(0, DFN_PATH)

from df.enhance import enhance, init_df


def main():
    if not DFN_PATH:
        raise ValueError("Set DFN_PATH first.")

    if not MODEL_DIR:
        raise ValueError("Set MODEL_DIR first.")

    model, df_state, _, epoch = init_df(
        model_base_dir=MODEL_DIR,
        epoch="best",
        log_file=None,
    )

    sample_rate = df_state.sr()
    block_size = int(sample_rate * BLOCK_SECONDS)

    model.eval()

    print(f"Loaded checkpoint epoch: {epoch}")
    print(f"Sample rate: {sample_rate} Hz")
    print("Speak into the microphone. Press Ctrl+C to stop.")

    with sd.Stream(
        samplerate=sample_rate,
        blocksize=block_size,
        channels=1,
        dtype="float32",
    ) as stream:
        while True:
            microphone_audio, _ = stream.read(block_size)

            audio = torch.from_numpy(
                np.asarray(
                    microphone_audio[:, 0],
                    dtype=np.float32,
                )
            ).unsqueeze(0)

            with torch.no_grad():
                enhanced_audio = enhance(
                    model,
                    df_state,
                    audio,
                    pad=True,
                )

            speaker_audio = (
                enhanced_audio.squeeze(0)
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            stream.write(speaker_audio.reshape(-1, 1))


if __name__ == "__main__":
    main()