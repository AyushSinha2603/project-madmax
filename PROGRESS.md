# Progress Log

Reproduction/analysis study: does the road graph help traffic-speed prediction
more as the horizon grows (15 -> 60 min)? Deliverable = the LSTM-vs-STGNN gap
plotted against horizon. Numbers compared to the DCRNN repo's post-bugfix table,
not the paper's (CLAUDE.md 8).

Compute note: Weeks 1 tasks are CPU-only. Only LSTM (Week 2) and STGNN (Weeks 3-4)
training need a GPU — planned on Google Colab free T4.

---

## Step 1 — Foundation (scaffolding, config, data layout)

- **Repo layout** created per CLAUDE.md 9: `src/`, `tests/fixtures/`, `scripts/`,
  `results/`, `data/raw/`, `data/processed/`.
- **Data placed** in `data/raw/`: `metr-la.h5` (speeds), `distances_la_2012.csv`
  (road distances = the graph), `graph_sensor_ids.txt` (canonical sensor order).
- **Test oracle** `adj_mx.pkl` downloaded to `tests/fixtures/`. Used only to verify
  our own graph builder later; never imported by `src/` (CLAUDE.md 5.2).
- **`config.yaml`** holds every tunable (horizons, split ratios, graph `k`, seed,
  model dims) so there are no magic numbers in code (CLAUDE.md 10).
- **`requirements.txt`** lists the stack; torch/PyG deferred to the Colab training
  phase; `torch_geometric_temporal` deliberately excluded (hides the pipeline).
- **`.gitignore`** ignores raw data, venv, caches, and the oracle (all downloadable).
- **Env gotcha:** local pandas 3.0.3 cannot `read_hdf` this file
  (`TypeError: a bytes-like object is required`). Worked around by reading the HDF5
  directly with PyTables — robust across pandas/Colab versions.

## Step 2 — Data loader (`src/data/load.py`)

- **Direct PyTables reader** returns the `(34272, 207)` speed DataFrame (mph),
  sidestepping the pandas 3.0 bug.
- **Column alignment** (the silent killer, CLAUDE.md 6.1): speed columns are
  reindexed to `graph_sensor_ids.txt` order so they line up with adjacency rows.
  Verified already in canonical order, but the guard makes misalignment impossible.
- **Missing-value mask**: zeros treated as missing (a real 0 mph jam is
  indistinguishable), giving a boolean validity mask. Missing rate = **8.11%**.
- **Chronological split** 70/10/20, no shuffle. Slices tile the timeline exactly:
  - train 23990 steps: 2012-03-01 00:00 -> 05-23 07:05
  - val    3427 steps: 05-23 07:10 -> 06-04 04:40
  - test   6855 steps: 06-04 04:45 -> 06-27 23:55
- **Train-only z-score**: mean=54.41 mph, std=19.49 mph, computed on train alone.
  Sanity check: normalized train split has mean -0.0000, std 1.0000. No leakage.
- Code carries no inline comments; docstrings retained for units/shapes.
