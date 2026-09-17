import os
import subprocess
from pathlib import Path

import torch

from .model import DTLN


def export_onnx(model: DTLN, onnx_path: str, input_shape: tuple[int, int] = (1, 16000)):
    model.eval()
    dummy_input = torch.randn(*input_shape)
    onnx_file = Path(onnx_path)
    onnx_file.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_file),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input_audio"],
        output_names=["enhanced_audio"],
    )

    print(f"Exported ONNX model to {onnx_file}")
    return str(onnx_file)


def export_tensorrt(onnx_path: str, engine_path: str, workspace_size_mb: int = 4096):
    trtexec = "trtexec"
    if os.system(f"which {trtexec} >/dev/null 2>&1") != 0:
        raise RuntimeError("TensorRT was not found on PATH. Install TensorRT or use the ONNX export only.")

    engine_file = Path(engine_path)
    engine_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={str(engine_file)}",
        f"--workspace={workspace_size_mb}",
        "--fp16",
    ]
    print("Running TensorRT export:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Saved TensorRT engine to {engine_file}")
    return str(engine_file)
