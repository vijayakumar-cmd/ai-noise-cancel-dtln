from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    sample_rate: int = 16000
    sample_length: int = 16000
    batch_size: int = 8
    epochs: int = 20
    learning_rate: float = 1e-3
    hidden_dim: int = 64
    kernel_size: int = 5
    lstm_layers: int = 2
    output_dir: str = "checkpoints"
    seed: int = 42
    num_samples: int = 256

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
