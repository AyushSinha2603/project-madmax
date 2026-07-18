import pickle
import numpy as np
from src.data.build_graph import build_adjacency

REF = 'tests/fixtures/adj_mx.pkl'
DIST = 'data/raw/distances_la_2012.csv'
IDS = 'data/raw/graph_sensor_ids.txt'


def test_adjacency_matches_reference():
    with open(REF, 'rb') as f:
        _, _, ref = pickle.load(f, encoding='latin1')
    mine = build_adjacency(DIST, IDS)
    assert mine.shape == ref.shape == (207, 207)
    assert np.abs(mine - np.asarray(ref)).max() < 1e-5


def test_adjacency_is_directed():
    """Road distance A->B != B->A. If this is symmetric, something collapsed."""
    mine = build_adjacency(DIST, IDS)
    assert not np.allclose(mine, mine.T), 'adjacency unexpectedly symmetric'
