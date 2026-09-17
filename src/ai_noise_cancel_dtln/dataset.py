import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticNoiseDataset(Dataset):
    """Generate synthetic waveform pairs for denoising experiments."""

    def __init__(
        self,
        num_samples: int = 1024,
        sample_rate: int = 16000,
        sample_length: int = 16000,
        noise_scale: float = 0.8,
        seed: int | None = None,
    ):
        self.num_samples = int(num_samples)
        self.sample_rate = int(sample_rate)
        self.sample_length = int(sample_length)
        self.noise_scale = float(noise_scale)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _generate_clean_signal(self, length: int, sr: int) -> np.ndarray:
        t = np.linspace(0.0, length / sr, length, endpoint=False)

        tone_a = 0.8 * np.sin(2 * np.pi * 220.0 * t)
        tone_b = 0.35 * np.sin(2 * np.pi * 330.0 * t + 0.7)
        tone_c = 0.25 * np.sin(2 * np.pi * 440.0 * t + 1.3)
        voiced = tone_a + tone_b + tone_c

        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2.7 * t + 0.4)
        envelope *= 0.5 + 0.5 * np.exp(-((t - 0.35) ** 2) / 0.04)
        signal = voiced * envelope
        return signal.astype(np.float32)

    def _generate_noise(self, length: int, sr: int) -> np.ndarray:
        base = self.rng.normal(0.0, 1.0, size=length).astype(np.float32)
        kernel = np.linspace(1.0, 0.1, 32)
        colored = np.convolve(base, kernel, mode="same")
        hum = 0.15 * np.sin(2 * np.pi * 60.0 * np.linspace(0.0, length / sr, length, endpoint=False))

        impulse_count = max(1, length // 1000)
        impulses = np.zeros(length, dtype=np.float32)
        impulse_positions = self.rng.integers(0, length, size=impulse_count)
        impulses[impulse_positions] = self.rng.uniform(0.4, 0.9, size=impulse_count)

        noise = 0.6 * base + 0.4 * colored + hum + impulses
        return noise.astype(np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        clean = self._generate_clean_signal(self.sample_length, self.sample_rate)
        noise = self._generate_noise(self.sample_length, self.sample_rate)
        gain = float(self.rng.uniform(0.25, 1.5))
        noisy = clean + gain * self.noise_scale * noise
        noisy = noisy / (np.max(np.abs(noisy)) + 1e-8)
        clean = clean / (np.max(np.abs(clean)) + 1e-8)
        return (
            torch.tensor(noisy, dtype=torch.float32),
            torch.tensor(clean, dtype=torch.float32),
        )
