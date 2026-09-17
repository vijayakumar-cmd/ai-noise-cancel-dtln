import os
import subprocess
from pathlib import Path


def jetson_export_script():
    script = """#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${1:-checkpoints/dtln.pt}
ONNX_PATH=${2:-checkpoints/dtln.onnx}
ENGINE_PATH=${3:-checkpoints/dtln.engine}

python -m ai_noise_cancel_dtln.cli export --model-path "$MODEL_PATH" --onnx-path "$ONNX_PATH"
if command -v trtexec >/dev/null 2>&1; then
  trtexec --onnx="$ONNX_PATH" --saveEngine="$ENGINE_PATH" --fp16 --workspace=4096
else
  echo "trtexec not found; ONNX export completed successfully."
fi
"""
    return script


def main():
    path = Path("scripts/jetson_export.sh")
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(jetson_export_script())
    os.chmod(path, 0o755)
    print(f"Created {path}")


if __name__ == "__main__":
    main()
