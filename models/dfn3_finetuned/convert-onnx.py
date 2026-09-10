from pathlib import Path
import re
import shutil
import subprocess
import sys
import os

root = Path(__file__).resolve().parent
out = root / "models/dfn3-epoch-130-onnx"
model = out / "_export_model"
(model / "checkpoints").mkdir(parents=True, exist_ok=True)

config_path = root / "data/mvp/finetune/dfn3-custom/config.ini"
config = config_path.read_text(encoding="utf-8")
config = re.sub(r"(?im)^(\s*device\s*=\s*).*$", r"\1cpu", config)
(model / "config.ini").write_text(config, encoding="utf-8")

checkpoint_candidates = [
    root / "data/mvp/finetune/dfn3-custom/checkpoints/model_130.ckpt.best",
    root / "data/mvp/finetune/dfn3-custom/checkpoints/model_130.ckpt",
]
checkpoint = next((p for p in checkpoint_candidates if p.exists()), checkpoint_candidates[0])
shutil.copy2(checkpoint, model / "checkpoints" / checkpoint.name)

env = os.environ.copy()
env["PYTHONPATH"] = str(root / "dfn3-model-files/deepfilternet/DeepFilterNet")

subprocess.run(
    [
        sys.executable,
        "-m",
        "df.scripts.export",
        "--model-base-dir",
        str(model),
        "--epoch",
        "130",
        str(out / "onnx"),
        "--opset",
        "14",
    ],
    cwd=root / "dfn3-model-files/deepfilternet/DeepFilterNet",
    env=env,
    check=True,
)
