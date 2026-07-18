"""Model-agnostic training + per-horizon evaluation. Loss is masked MAE in mph."""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.metrics import all_metrics, masked_mae

HORIZON_STEPS = {3: '15min', 6: '30min', 12: '60min'}


def make_loader(npz_path, batch_size=64, shuffle=False) -> DataLoader:
    """Shuffling WITHIN a split is fine -- the chronological split already
    happened in prepare.py. Never shuffle before splitting."""
    d = np.load(npz_path)
    x = torch.from_numpy(d['x']).float()
    y = torch.from_numpy(d['y'][..., 0]).float()   # speed channel, raw mph
    return DataLoader(TensorDataset(x, y), batch_size=batch_size,
                      shuffle=shuffle, drop_last=False)


@torch.no_grad()
def evaluate(model, loader, scaler, device):
    model.eval()
    preds, reals = [], []
    for x, y in loader:
        out = model(x.to(device))
        preds.append(scaler.inverse_transform(out).cpu())
        reals.append(y)
    preds, reals = torch.cat(preds), torch.cat(reals)
    res = {lab: all_metrics(preds[:, s - 1, :], reals[:, s - 1, :])
           for s, lab in HORIZON_STEPS.items()}
    res['avg'] = all_metrics(preds, reals)
    return res, preds, reals


def train(model, train_loader, val_loader, scaler, device, epochs=60, lr=1e-3,
          weight_decay=1e-4, clip=5.0, patience=12, ckpt='results/best.pt'):
    Path(ckpt).parent.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [20, 35, 50], gamma=0.1)

    best, bad, history = float('inf'), 0, []
    for ep in range(1, epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = scaler.inverse_transform(model(x))   # -> mph
            loss = masked_mae(pred, y)
            loss.backward()
            # DCRNN's own README warns the loss can explode. Clip.
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            losses.append(loss.item())
        sched.step()

        val, _, _ = evaluate(model, val_loader, scaler, device)
        vm = val['avg']['mae']
        history.append({'epoch': ep, 'train_mae': float(np.mean(losses)),
                        'val_mae': vm})
        print(f'epoch {ep:3d} | train {np.mean(losses):.4f} | val {vm:.4f} '
              f'| {time.time()-t0:.1f}s')

        if vm < best - 1e-4:
            best, bad = vm, 0
            torch.save(model.state_dict(), ckpt)   # ckpt is in Drive -> survives
        else:
            bad += 1
            if bad >= patience:
                print(f'early stop @ {ep} (best val {best:.4f})')
                break

    model.load_state_dict(torch.load(ckpt))
    return model, history


def save_run(path, name, config, results, history, seed):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump({'model': name, 'seed': seed, 'config': config,
                   'results': results, 'history': history}, f, indent=2)
