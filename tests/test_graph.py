import numpy as np
import pytest
import yaml

from src.data.build_graph import build_adjacency, symmetrize


@pytest.fixture(scope="module")
def cfg():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def adj(cfg):
    return build_adjacency(
        cfg["paths"]["distances_csv"],
        cfg["paths"]["sensor_ids_txt"],
        k=cfg["graph"]["k"],
    )


def test_shape(adj, cfg):
    n = cfg["data"]["n_sensors"]
    assert adj.shape == (n, n)


def test_asymmetric(adj):
    assert not np.allclose(adj, adj.T)


def test_unit_diagonal(adj):
    assert np.allclose(adj.diagonal(), 1.0)


def test_values_in_unit_interval(adj):
    assert adj.min() >= 0.0
    assert adj.max() <= 1.0


def test_sparse_after_threshold(adj, cfg):
    density = (adj > 0).mean()
    assert 0.0 < density < 0.2
    assert (adj[adj > 0].min()) >= cfg["graph"]["k"]


def test_symmetrize_is_symmetric(adj):
    sym = symmetrize(adj)
    assert np.allclose(sym, sym.T)
