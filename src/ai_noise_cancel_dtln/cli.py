import argparse
import numpy as np
import torch

from .config import TrainConfig
from .dataset import SyntheticNoiseDataset
from .export import export_onnx, export_tensorrt
from .inference import StreamingEnhancer
from .model import DTLN
from .train import train_model


def build_parser():
    parser = argparse.ArgumentParser(description="DTLN noise cancellation CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--output-dir", type=str, default="checkpoints")
    train_parser.add_argument("--sample-rate", type=int, default=16000)
    train_parser.add_argument("--sample-length", type=int, default=16000)
    train_parser.add_argument("--num-samples", type=int, default=256)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--hidden-dim", type=int, default=64)
    train_parser.add_argument("--kernel-size", type=int, default=5)
    train_parser.add_argument("--lstm-layers", type=int, default=2)
    train_parser.add_argument("--seed", type=int, default=42)

    infer_parser = subparsers.add_parser("infer", help="Run synthetic inference")
    infer_parser.add_argument("--model-path", type=str, default="checkpoints/dtln.pt")
    infer_parser.add_argument("--sample-length", type=int, default=16000)

    export_parser = subparsers.add_parser("export", help="Export model to ONNX/TensorRT")
    export_parser.add_argument("--model-path", type=str, default="checkpoints/dtln.pt")
    export_parser.add_argument("--onnx-path", type=str, default="checkpoints/dtln.onnx")
    export_parser.add_argument("--engine-path", type=str, default="checkpoints/dtln.engine")
    export_parser.add_argument("--tensorrt", action="store_true")

    demo_parser = subparsers.add_parser("demo", help="Run a quick streaming demo")
    demo_parser.add_argument("--sample-length", type=int, default=16000)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        config = TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            sample_rate=args.sample_rate,
            sample_length=args.sample_length,
            num_samples=args.num_samples,
            learning_rate=args.learning_rate,
            hidden_dim=args.hidden_dim,
            kernel_size=args.kernel_size,
            lstm_layers=args.lstm_layers,
            seed=args.seed,
        )
        train_model(config)
        return

    if args.command == "infer":
        sample = np.random.randn(args.sample_length).astype("float32")
        enhancer = StreamingEnhancer(model_path=args.model_path)
        out = enhancer.process(sample)
        print(f"Input length: {len(sample)}")
        print(f"Output length: {len(out)}")
        return

    if args.command == "export":
        checkpoint = torch.load(args.model_path, map_location="cpu")
        model = DTLN()
        model.load_state_dict(checkpoint["model_state"])
        export_onnx(model, args.onnx_path)
        if args.tensorrt:
            export_tensorrt(args.onnx_path, args.engine_path)
        return

    if args.command == "demo":
        dataset = SyntheticNoiseDataset(num_samples=1, sample_length=args.sample_length)
        noisy, clean = dataset[0]
        enhancer = StreamingEnhancer(model=DTLN())
        out = enhancer.process(noisy.numpy())
        print(f"Demo sample length: {len(noisy)}")
        print(f"Enhanced output length: {len(out)}")
        return


if __name__ == "__main__":
    main()
