import torch
import torch.nn.functional as F


def perceptual_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Combined waveform + spectral loss for denoising training."""
    waveform_l1 = F.l1_loss(pred, target)
    waveform_mse = F.mse_loss(pred, target)

    pred_fft = torch.fft.rfft(pred, dim=-1)
    target_fft = torch.fft.rfft(target, dim=-1)
    spectral_l1 = F.l1_loss(pred_fft.abs(), target_fft.abs())
    spectral_mse = F.mse_loss(pred_fft.abs(), target_fft.abs())

    return waveform_l1 + 0.5 * waveform_mse + 0.5 * spectral_l1 + 0.25 * spectral_mse
