# AI Noise Cancel DTLN

PyTorch-based DTLN AI noise cancellation pipeline for synthetic data generation, model training, streaming inference, ONNX export, and Jetson-friendly deployment.

## Features

- Synthetic noise + speech mixture generation
- Lightweight DTLN-style model in PyTorch
- Training loop with configurable hyperparameters
- Streaming inference with overlap-add buffering
- ONNX export workflow
- TensorRT export entry point for NVIDIA deployment
- Jetson deployment notes and config

## Project structure

- `src/ai_noise_cancel_dtln/` - Python package
- `tests/` - basic validation tests
- `README.md` - project documentation

## Quick start

1. Create a virtual environment and install dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

2. Train the model.

```bash
python -m ai_noise_cancel_dtln.cli train --epochs 10 --batch-size 8 --output-dir checkpoints
```

3. Run a quick inference demo.

```bash
python -m ai_noise_cancel_dtln.cli infer --model-path checkpoints/dtln.pt --sample-length 16000
```

4. Export ONNX.

```bash
python -m ai_noise_cancel_dtln.cli export --model-path checkpoints/dtln.pt --onnx-path checkpoints/dtln.onnx
```

## Model overview

The project implements a compact DTLN-inspired architecture:

- 1D convolutional encoder
- LSTM bottleneck
- 1D convolutional decoder
- residual skip connection for stable training

This design keeps inference efficient while allowing real-time enhancement on embedded devices.

## Synthetic dataset

The built-in dataset synthesizes:

- speech-like tones and amplitude modulated carriers
- background noise with Gaussian and colored components
- impulsive bursts and low-frequency hum

This makes the project useful as a baseline development pipeline even without a real labeled dataset.

## Jetson deployment notes

For NVIDIA Jetson platforms:

- export the model to ONNX first
- convert to TensorRT using `trtexec`
- use FP16/INT8 mixed precision when supported
- keep inference buffers small for low-latency streaming

Example:

```bash
trtexec --onnx=checkpoints/dtln.onnx --saveEngine=checkpoints/dtln.engine --fp16 --workspace=4096
```

## License

MIT
