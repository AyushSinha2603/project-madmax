import numpy as np
import torch
from src.models.stgnn import STGNN, build_supports
from src.models.lstm_baseline import LSTMBaseline

N = 20


def _fake_adj(n=N):
    rng = np.random.default_rng(0)
    a = rng.random((n, n)).astype(np.float32)
    a[a < 0.7] = 0.0
    np.fill_diagonal(a, 1.0)
    return a


def test_shapes():
    m = STGNN(build_supports(_fake_adj(), 'road'))
    x = torch.randn(4, 12, N, 2)
    assert m(x).shape == (4, 12, N)


def test_param_count_matches_identity_control():
    """The headline claim depends on this. If road and identity differ in
    params, the comparison measures capacity, not the graph."""
    a = STGNN(build_supports(_fake_adj(), 'road'))
    b = STGNN(build_supports(_fake_adj(), 'identity'))
    pa = sum(p.numel() for p in a.parameters())
    pb = sum(p.numel() for p in b.parameters())
    assert pa == pb, f'{pa} != {pb}'


def test_stgnn_has_more_params_than_lstm():
    """Sanity: the graph layer really does add capacity -- which is exactly why
    the identity control is needed."""
    s = sum(p.numel() for p in STGNN(build_supports(_fake_adj(), 'road')).parameters())
    l = sum(p.numel() for p in LSTMBaseline().parameters())
    assert s > l


def test_road_supports_are_row_stochastic():
    p_f, p_b = build_supports(_fake_adj(), 'road')
    assert torch.allclose(p_f.sum(1), torch.ones(N), atol=1e-5)
    assert torch.allclose(p_b.sum(1), torch.ones(N), atol=1e-5)


def test_identity_support_does_not_mix():
    """With identity supports, node i's output must not depend on node j."""
    torch.manual_seed(0)
    m = STGNN(build_supports(_fake_adj(), 'identity')).eval()
    x = torch.randn(1, 12, N, 2)
    y1 = m(x)
    x2 = x.clone()
    x2[:, :, 5, :] += 10.0          # perturb ONE node
    y2 = m(x2)
    changed = (y1 - y2).abs().sum(dim=(0, 1)) > 1e-5
    assert changed[5], 'perturbed node should change'
    assert not changed[[i for i in range(N) if i != 5]].any(), \
        'identity control LEAKED across nodes'


def test_road_support_does_mix():
    """Sanity check on the mirror image: the road graph MUST mix."""
    torch.manual_seed(0)
    m = STGNN(build_supports(_fake_adj(), 'road')).eval()
    x = torch.randn(1, 12, N, 2)
    y1 = m(x)
    x2 = x.clone()
    x2[:, :, 5, :] += 10.0
    y2 = m(x2)
    others = (y1 - y2).abs().sum(dim=(0, 1))[[i for i in range(N) if i != 5]]
    assert others.max() > 1e-5, 'road graph did not mix -- graph is not connected?'
