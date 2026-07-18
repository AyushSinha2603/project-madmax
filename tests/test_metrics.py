import torch
from src.metrics import masked_mae


def test_masked_mae_ignores_zeros():
    labels = torch.tensor([[10.0, 0.0, 20.0]])   # middle one is MISSING
    preds = torch.tensor([[12.0, 99.0, 24.0]])   # wild guess where missing
    # errors on real values: |12-10|=2, |24-20|=4 -> mean 3.0
    assert abs(masked_mae(preds, labels).item() - 3.0) < 1e-5


def test_masked_differs_from_unmasked():
    labels = torch.tensor([[10.0, 0.0, 20.0]])
    preds = torch.tensor([[12.0, 99.0, 24.0]])
    unmasked = (preds - labels).abs().mean().item()
    assert abs(unmasked - masked_mae(preds, labels).item()) > 1.0


def test_all_real_matches_plain_mae():
    labels = torch.tensor([[10.0, 15.0, 20.0]])
    preds = torch.tensor([[11.0, 15.0, 22.0]])
    assert abs(masked_mae(preds, labels).item() - 1.0) < 1e-5
