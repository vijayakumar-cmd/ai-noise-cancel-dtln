# AI Noise Cancel DTLN

PyTorch-based DTLN AI noise cancellation pipeline for synthetic data mixing, perceptual losses, streaming inference, ONNX/TensorRT export, and Jetson deployment support.

## Features

- Synthetic data generator for noisy speech-like waveforms
- Compact DTLN-inspired denoising model in PyTorch
- Perceptual + spectral loss training objective
- Streaming inference with overlap-add buffering
- ONNX export tooling
- TensorRT export entry point for NVIDIA deployment
- Jetson deployment notes and helper scripts
- CLI for training, inference, export, and demo workflows

## Repository layout

- `src/ai_noise_cancel_dtln/` — Python package
- `tests/` — smoke tests and validation
- `scripts/` — deployment and benchmarking helpers
- `docs/` — deployment notes
- `examples/` — sample usage scripts

## Quick start

1. Create a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

2. Train a model.

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

5. Run a demo script.

```bash
python examples/quick_demo.py
```

## Model architecture

The package implements a compact DTLN-inspired pipeline:

- Temporal encoder using 1D convolutional layers
- LSTM bottleneck for temporal modeling
- Decoder with residual enhancement path
- Learned denoising with waveform regression objective

## Loss design

The training pipeline uses a combined objective:

- L1 waveform loss
- MSE waveform loss
- Spectral magnitude loss
- Optional perceptual-style aggregation of waveform + spectral components

This gives a more stable training signal than pure samplewise loss alone.

## Streaming inference

Streaming inference is handled by a `StreamingEnhancer` class that keeps a partial buffer, processes fixed frames, and merges results via overlap-add. This makes it suitable for real-time or low-latency enhancement pipelines.

## Export workflow

- Export ONNX with `torch.onnx.export`
- Optionally convert to TensorRT with `trtexec`
- Keep model weights compatible with Jetson and embedded inference stacks

Example:

```bash
trtexec --onnx=checkpoints/dtln.onnx --saveEngine=checkpoints/dtln.engine --fp16 --workspace=4096
```

## Jetson deployment notes

For NVIDIA Jetson devices:

- export to ONNX first
- convert to TensorRT with `trtexec`
- prefer FP16 for throughput
- keep frame sizes small for ultra-low latency
- validate with short real-world audio clips before deployment

See `docs/jetson.md` and `scripts/jetson_export.sh`.

## License

MIT
