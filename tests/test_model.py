import torch

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
