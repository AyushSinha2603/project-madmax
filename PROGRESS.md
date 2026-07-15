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

## Step 3 — Sanity gate (`scripts/check_data.py`)

- Fail-loud script turning the CLAUDE.md 6.1 table into asserts; exits non-zero on
  any failure. Read-only. **8/8 checks pass.**
- Verified: speed shape (34272, 207); uniform 5-min resolution; date range
  2012-03-01 -> 2012-06-27; missing rate 8.11%; non-zero speeds [0.3, 70.0] mph.
- **Column order == graph_sensor_ids.txt exactly (207/207)** — the silent killer
  is confirmed dead before any model is built.
- Distances CSV checked: 295374 road-pairs, `from,to,cost` columns, costs finite
  and >= 0 (0-21255 m), all 207 sensor ids referenced. This is the graph's input.
- Adjacency shape/asymmetry deferred to Step 5 (checked on our own builder, not
  the oracle, per CLAUDE.md 5.2).

## Step 4 — Masked metrics (`src/metrics.py` + `tests/test_metrics.py`)

- `masked_mae`, `masked_rmse`, `masked_mape` (+ `all_metrics`). Mask derived from
  `target != 0` (missing) unless an explicit mask is passed. MAE/RMSE in mph,
  MAPE in percent. Return NaN when no valid entries.
- These are the ruler every model is judged by, so correctness is nailed with a
  **hand-computed 2x3 test** (targets in mph, two masked zeros):
  MAE=2.75, RMSE=sqrt(8.25)=2.8723, MAPE=12.5%.
- Key guard test: one masked cell carries a 100 mph error; masked MAE (2.75) must
  differ from the naive unmasked mean (~19.33). Proves zeros are truly excluded.
- Added root `conftest.py` so pytest can import `src`. **7/7 tests pass.**

## Step 5 — Road-graph adjacency (`src/data/build_graph.py` + `tests/test_graph.py`)

- Implements DCRNN's thresholded Gaussian kernel over road-network distances:
  `A[i,j] = exp(-(dist/sigma)^2)` if `>= k` else 0. `sigma` is the computed std of
  observed distances (not chosen); `k = 0.1` sparsifies.
- Row/column order follows `graph_sensor_ids.txt`, so the matrix aligns with the
  speed columns (same canonical order verified in Step 3).
- Result is **directed and asymmetric** (road distance A->B != B->A); not
  symmetrized. A `symmetrize()` helper is provided for models that require it,
  to be applied explicitly and logged (CLAUDE.md 5.1).
- Built matrix: shape (207, 207), asymmetric, unit diagonal, weights in [0, 1],
  density 4.02% nonzero after the threshold.
- `test_graph.py` asserts these structural properties (shape, asymmetry, unit
  diagonal, value range, sparsity, and that `symmetrize` is symmetric). This also
  covers the adjacency checks deferred from Step 3. **All property tests pass.**

## Step 6 — Windowing (`src/data/windowing.py` + `tests/test_windowing.py`)

- Sliding windows of 12 steps in (1h history) -> 12 steps out (1h ahead), built
  **within each split** so no window straddles a train/val/test boundary.
- Shape convention (batch, time, nodes, features), asserted at construction:
  x = (num, 12, 207, F), y = (num, 12, 207), y_mask, y_time (ns per target step).
- Features F=2: normalized speed + time-of-day in [0,1). Day-of-week available via
  config (off by default). Time-of-day matters a lot for traffic.
- Saved compressed to `data/processed/{train,val,test}.npz` + `metadata.json`
  (scaler mean/std, sensor order, feature names, horizon->minute map). npz are
  git-ignored. Sizes: train 29 MB, val 4 MB, test 8 MB.
- Sample counts = split_len - 23: train 23967, val 3404, test 6832. Valid targets
  92.9% / 93.0% / 87.9%.
- Tests pin the off-by-one alignment (target = steps immediately after input),
  shapes, and the time-of-day feature. A real-data check confirms target time
  ranges are strictly ordered and non-overlapping (no leakage). **18/18 tests pass.**
