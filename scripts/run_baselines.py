from __future__ import annotations

import logging
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.data.load import load_config, load_metrla
from src.data.windowing import load_windows
from src.evaluate import per_horizon_metrics, upsert_metrics, write_run_log
from src.models.copy_last import CopyLastValue
from src.models.historical_average import HistoricalAverage

logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    np.random.seed(cfg["seed"])

    ds = load_metrla(cfg)
    horizons = cfg["windowing"]["report_horizons"]
    freq = cfg["data"]["freq_minutes"]
    out_steps = cfg["windowing"]["output_steps"]
    results_dir = cfg["paths"]["results_dir"]
    metrics_csv = f"{results_dir}/metrics.csv"

    tr = ds.splits["train"]
    train_times = ds.timestamps[tr].to_numpy(dtype="datetime64[ns]").view("int64")
    ha = HistoricalAverage(freq_minutes=freq).fit(
        ds.speed[tr], ds.mask[tr], train_times
    )
    copy_last = CopyLastValue()

    all_metrics: dict[str, dict] = {}
    for split in ["val", "test"]:
        w = load_windows(cfg["paths"]["processed_dir"], split)

        preds = {
            copy_last.name: copy_last.predict(w["x"], out_steps, ds.scaler),
            ha.name: ha.predict(w["y_time"]),
        }
        for model_name, p in preds.items():
            rows = per_horizon_metrics(
                p, w["y"], w["y_mask"], ds.scaler, horizons, freq
            )
            upsert_metrics(metrics_csv, model_name, split, rows)
            all_metrics.setdefault(model_name, {})[split] = rows

    write_run_log(
        results_dir, cfg, cfg["seed"], all_metrics, extra={"stage": "baselines"}
    )

    print("\n=== Baselines on TEST (masked, mph) ===")
    print(f"{'model':<18}{'horizon':>9}{'MAE':>8}{'RMSE':>8}{'MAPE%':>8}")
    for model_name in [copy_last.name, ha.name]:
        for r in all_metrics[model_name]["test"]:
            print(
                f"{model_name:<18}{r['horizon_min']:>7}m{r['mae']:>8.2f}"
                f"{r['rmse']:>8.2f}{r['mape']:>8.2f}"
            )
    print(f"\nWrote {metrics_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
