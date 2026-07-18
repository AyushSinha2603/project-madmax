"""Masked error metrics. All in mph (inverse-transform predictions first)."""
from __future__ import annotations
import torch

NULL_VAL = 0.0


def _mask(labels: torch.Tensor, null_val: float = NULL_VAL) -> torch.Tensor:
    """1.0 where real, 0.0 where missing; rescaled so .mean() over ALL
    elements equals the mean over MASKED elements."""
    mask = (labels != null_val).float()
    mask = mask / mask.mean()
    return torch.nan_to_num(mask)


def masked_mae(preds, labels, null_val: float = NULL_VAL):
    m = _mask(labels, null_val)
    return torch.nan_to_num(torch.abs(preds - labels) * m).mean()


def masked_rmse(preds, labels, null_val: float = NULL_VAL):
    m = _mask(labels, null_val)
    return torch.sqrt(torch.nan_to_num(torch.square(preds - labels) * m).mean())


def masked_mape(preds, labels, null_val: float = NULL_VAL):
    m = _mask(labels, null_val)
    return torch.nan_to_num(torch.abs((preds - labels) / labels) * m).mean()


def all_metrics(preds, labels) -> dict:
    return {"mae": masked_mae(preds, labels).item(),
            "rmse": masked_rmse(preds, labels).item(),
            "mape": masked_mape(preds, labels).item() * 100.0}
