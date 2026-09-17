import torch

from ai_noise_cancel_dtln.losses import perceptual_loss
from ai_noise_cancel_dtln.model import DTLN


def test_output_shape():
    model = DTLN()
    x = torch.randn(2, 16000)
    y = model(x)
    assert y.shape == x.shape


def test_model_is_trainable():
    model = DTLN()
    x = torch.randn(2, 16000)
    y = model(x)
    assert y.requires_grad


def test_perceptual_loss_runs():
    pred = torch.randn(2, 16000)
    target = torch.randn(2, 16000)
    loss = perceptual_loss(pred, target)
    assert torch.isfinite(loss)
