"""metr-la.h5 -> windowed chronological splits. Speed in mph. Missing = 0.0."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

SEQ_IN, SEQ_OUT = 12, 12


class StandardScaler:
    """Z-score. Statistics MUST come from the training split only."""

    def __init__(self, mean: float, std: float) -> None:
        self.mean, self.std = float(mean), float(std)

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse_transform(self, x):
        return x * self.std + self.mean


def load_scaler(path='data/processed/scaler.npz') -> StandardScaler:
    s = np.load(path)
    return StandardScaler(float(s['mean']), float(s['std']))


def load_frame(h5_path) -> pd.DataFrame:
    df = pd.read_hdf(h5_path)
    assert df.shape == (34272, 207), f'unexpected shape {df.shape}'
    return df


def prepare(h5_path, out_dir='data/processed') -> StandardScaler:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_frame(h5_path)
    print(f'missing (zeros): {(df.values == 0).mean():.3%}   [expect ~8%]')
    print(f'range: {df.index.min()} -> {df.index.max()}')

    n_t, n_nodes = df.shape
    speed = np.expand_dims(df.values, -1)
    day_frac = ((df.index.values - df.index.values.astype('datetime64[D]'))
                / np.timedelta64(1, 'D'))
    tod = np.tile(day_frac, [1, n_nodes, 1]).transpose((2, 1, 0))
    data = np.concatenate([speed, tod], -1).astype(np.float32)  # (T, N, 2)

    x_off = np.arange(-(SEQ_IN - 1), 1)
    y_off = np.arange(1, SEQ_OUT + 1)
    lo, hi = abs(x_off.min()), n_t - y_off.max()

    xs = np.stack([data[t + x_off] for t in range(lo, hi)]).astype(np.float32)
    ys = np.stack([data[t + y_off] for t in range(lo, hi)]).astype(np.float32)
    # timestamps of the TARGET steps -- needed by the historical-average baseline
    ts = np.stack([df.index.values[t + y_off] for t in range(lo, hi)])
    print(f'windows: x={xs.shape} y={ys.shape}')

    n = xs.shape[0]
    n_test, n_train = round(n * 0.2), round(n * 0.7)
    n_val = n - n_test - n_train

    # GAP between splits. Windows overlap: the last train window's targets and
    # the first val window's inputs can cover the same timestamps. DCRNN splits
    # on window index and tolerates this. We drop SEQ_IN+SEQ_OUT windows at each
    # boundary instead, which makes the splits strictly non-overlapping in time.
    # Costs ~24 of 34249 windows (0.07%) -- no measurable effect on comparability.
    gap = SEQ_IN + SEQ_OUT
    bounds = {'train': (0, n_train - gap),
              'val': (n_train, n_train + n_val - gap),
              'test': (n - n_test, n)}

    # Scaler: TRAIN split only, speed channel only.
    # Known impurity: zeros (missing) are included in mean/std, exactly as
    # DCRNN and Graph-WaveNet do. Kept for comparability. Note it in the report.
    xtr = xs[:n_train]
    scaler = StandardScaler(xtr[..., 0].mean(), xtr[..., 0].std())
    print(f'scaler: mean={scaler.mean:.4f} std={scaler.std:.4f}')

    for name, (a, b) in bounds.items():
        xa = xs[a:b].copy()
        xa[..., 0] = scaler.transform(xa[..., 0])   # inputs normalized
        # y stays raw mph: the masked loss is computed in mph.
        np.savez_compressed(out_dir / f'{name}.npz', x=xa, y=ys[a:b],
                            y_time=ts[a:b].astype('datetime64[ns]').astype(np.int64))
        print(f'{name:5s} x={xa.shape}  {pd.Timestamp(ts[a][0])} -> '
              f'{pd.Timestamp(ts[b-1][-1])}')

    np.savez(out_dir / 'scaler.npz', mean=scaler.mean, std=scaler.std)
    return scaler
