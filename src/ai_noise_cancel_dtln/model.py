import torch
import torch.nn as nn


class DTLN(nn.Module):
    """Compact DTLN-inspired denoising model for 1D waveforms."""

    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 64,
        kernel_size: int = 5,
        lstm_layers: int = 2,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=lstm_layers, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, in_channels, kernel_size, padding=kernel_size // 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        encoded = self.encoder(x)
        encoded = encoded.transpose(1, 2)
        encoded, _ = self.lstm(encoded)
        encoded = encoded.transpose(1, 2)

        residual = self.decoder(encoded)
        enhanced = x + torch.tanh(residual) * 0.75
        return enhanced.squeeze(1)

    @torch.no_grad()
    def export_state_dict(self, path: str):
        torch.save(self.state_dict(), path)
