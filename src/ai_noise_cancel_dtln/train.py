import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import TrainConfig
from .dataset import SyntheticNoiseDataset
from .losses import perceptual_loss
from .model import DTLN


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(config: TrainConfig):
    set_seed(config.seed)
    dataset = SyntheticNoiseDataset(
        num_samples=config.num_samples,
        sample_rate=config.sample_rate,
        sample_length=config.sample_length,
        seed=config.seed,
    )

    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    model = DTLN(hidden_dim=config.hidden_dim, kernel_size=config.kernel_size, lstm_layers=config.lstm_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        for noisy, clean in loader:
            optimizer.zero_grad()
            pred = model(noisy)
            loss = perceptual_loss(pred, clean)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(loader))
        print(f"epoch={epoch + 1}/{config.epochs} loss={avg_loss:.6f}")

    checkpoint_path = config.output_dir / "dtln.pt"
    state = {
        "model_state": model.state_dict(),
        "config": config,
    }
    torch.save(state, checkpoint_path)
    print(f"Saved model to {checkpoint_path}")
    return checkpoint_path
