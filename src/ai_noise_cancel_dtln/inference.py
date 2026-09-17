import numpy as np
import torch

from .model import DTLN


class StreamingEnhancer:
    """Simple streaming denoising helper based on frame-by-frame overlap-add."""

    def __init__(
        self,
        model_path: str | None = None,
        model: DTLN | None = None,
        frame_size: int = 4096,
        hop_size: int = 2048,
    ):
        if model is None:
            if model_path is None:
                raise ValueError("Either model or model_path must be set")
            state = torch.load(model_path, map_location="cpu")
            model = DTLN()
            model.load_state_dict(state["model_state"])

        self.model = model.eval()
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.buffer = np.zeros(0, dtype=np.float32)

    def process(self, waveform: np.ndarray) -> np.ndarray:
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim != 1:
            raise ValueError("Input waveform must be a 1D NumPy array")

        self.buffer = np.concatenate([self.buffer, waveform])
        outputs = []

        while len(self.buffer) >= self.frame_size:
            frame = self.buffer[: self.frame_size]
            self.buffer = self.buffer[self.hop_size :]
            with torch.no_grad():
                frame_tensor = torch.tensor(frame, dtype=torch.float32).unsqueeze(0)
                enhanced = self.model(frame_tensor).squeeze(0).numpy()
            outputs.append(enhanced)

        if outputs:
            return np.concatenate(outputs)
        return np.zeros(0, dtype=np.float32)

    def flush(self) -> np.ndarray:
        if len(self.buffer) == 0:
            return np.zeros(0, dtype=np.float32)

        remaining = self.buffer.copy()
        self.buffer = np.zeros(0, dtype=np.float32)
        with torch.no_grad():
            tensor = torch.tensor(remaining, dtype=torch.float32).unsqueeze(0)
            return self.model(tensor).squeeze(0).numpy()
