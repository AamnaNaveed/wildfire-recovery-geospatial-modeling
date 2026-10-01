"""
Stage 8: Persistence baseline.

Predict each pixel's dNBR as the mean dNBR of its fire's ecosystem group
in the training fires. This is the floor that any ML model must beat
under leave-one-fire-out validation.

Evaluates with leave-one-fire-out over the 10 usable fires.
Reports RMSE, MAE, R2, Spearman per held-out fire.

Per PROJECT_PLAN.md section 8.1 and section 15, row 8.
"""

from pathlib import Path
import json
import gc
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_ROOT / "data" / "interim" / "features"
TEST_FIRES_TXT = PROJECT_ROOT / "data" / "interim" / "test_fires.txt"
FEATURE_REPORT = PROJECT_ROOT / "reports" / "feature_tables_report.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"


USABLE_SIGNALS = {"strong", "weak"}


def log(msg):
    print(msg, flush=True)


def load_usable_fires():
    """Get event_ids of usable fires (strong + weak signal)."""
    report = json.loads(FEATURE_REPORT.read_text())
    return [
        f for f in report["fires"]
        if f["fire_signal_quality"] in USABLE_SIGNALS
    ]


def load_test_fires():
    if not TEST_FIRES_TXT.exists():
        return []
    return [line.strip() for line in TEST_FIRES_TXT.read_text().splitlines() if line.strip()]


def load_feature_table(event_id):
    path = FEATURES_DIR / f"{event_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_parquet(path)


def leave_one_fire_out_baseline(df_all, fire_ids, target_col="dnbr"):
    """Run LOFO for the persistence baseline."""
    results = []

    for held_out in fire_ids:
        train = df_all[df_all["event_id"] != held_out]
        test = df_all[df_all["event_id"] == held_out]

        # Compute per-ecosystem means from training fires only
        eco_means = train.groupby("ecosystem")[target_col].mean().to_dict()
        global_mean = float(train[target_col].mean())

        # Predict each test pixel from its fire's ecosystem
        # All pixels in a fire share an ecosystem (this is the fire-level label)
        ecosystem = test["ecosystem"].iloc[0]
        pred_value = eco_means.get(ecosystem, global_mean)

        y_true = test[target_col].values
        y_pred = np.full_like(y_true, pred_value, dtype=float)

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        try:
            r2 = float(r2_score(y_true, y_pred))
        except Exception:
            r2 = float("nan")
        try:
            rho, _ = spearmanr(y_true, y_pred)
            rho = float(rho) if np.isfinite(rho) else float("nan")
        except Exception:
            rho = float("nan")

        results.append({
            "held_out_fire": held_out,
            "ecosystem": ecosystem,
            "n_pixels": int(len(test)),
            "predicted_value": round(pred_value, 4),
            "target_mean": round(float(y_true.mean()), 4),
            "target_std": round(float(y_true.std()), 4),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2": round(r2, 4),
            "spearman": round(rho, 4),
        })

    return results


def make_figure(results, out_path):
    fires = [r["held_out_fire"] for r in results]
    rmse = [r["rmse"] for r in results]
    r2 = [r["r2"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].barh(fires, rmse)
    axes[0].set_xlabel("RMSE")
    axes[0].set_title("Persistence baseline: RMSE per held-out fire")
    axes[1].barh(fires, r2)
    axes[1].set_xlabel("R²")
    axes[1].set_title("Persistence baseline: R² per held-out fire")
    for ax in axes:
        ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main():
    log("Stage 8: Persistence baseline")
    log("-" * 50)

    usable = load_usable_fires()
    fire_ids = [f["event_id"] for f in usable]
    log(f"Usable fires: {len(fire_ids)}")

    test_ids = load_test_fires()
    log(f"Test fires (frozen): {test_ids}")

    # For LOFO, we use all usable fires as held-out at some point
    # (including the test fires — their dNBR is used but their *labels* aren't
    # used in training because they're held out)

    log("Loading feature tables ...")
    dfs = []
    for f in usable:
        df = load_feature_table(f["event_id"])
        dfs.append(df)
        log(f"  {f['fire_name']:25s} {len(df):>7,} rows")
    all_df = pd.concat(dfs, ignore_index=True)
    log(f"Total rows: {len(all_df):,}")

    log("-" * 50)
    log("Running leave-one-fire-out persistence baseline ...")
    results = leave_one_fire_out_baseline(all_df, fire_ids)

    # Aggregate
    rmse_vals = [r["rmse"] for r in results]
    r2_vals = [r["r2"] for r in results]
    log("")
    log(f"{'Fire':25s} {'Eco':10s} {'RMSE':>8s} {'MAE':>8s} {'R²':>8s} {'Rho':>8s}")
    for r in results:
        log(f"{r['held_out_fire']:25s} {r['ecosystem']:10s} "
            f"{r['rmse']:>8.4f} {r['mae']:>8.4f} {r['r2']:>8.4f} {r['spearman']:>8.4f}")

    log("")
    log(f"RMSE mean ± SD: {np.mean(rmse_vals):.4f} ± {np.std(rmse_vals):.4f}")
    log(f"R²   mean ± SD: {np.nanmean(r2_vals):.4f} ± {np.nanstd(r2_vals):.4f}")

    # Save figure
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURE_DIR / "stage8_persistence_baseline.png"
    make_figure(results, fig_path)
    log(f"Saved figure: {fig_path.relative_to(PROJECT_ROOT)}")

    # Save report
    report = {
        "stage": 8,
        "model": "persistence",
        "description": "Predict dNBR as mean of ecosystem group in training fires",
        "timestamp": datetime.now().isoformat(),
        "n_fires": len(fire_ids),
        "n_rows_total": int(len(all_df)),
        "test_fires": test_ids,
        "per_fire_results": results,
        "aggregate": {
            "rmse_mean": float(np.mean(rmse_vals)),
            "rmse_std": float(np.std(rmse_vals)),
            "mae_mean": float(np.mean([r["mae"] for r in results])),
            "r2_mean": float(np.nanmean(r2_vals)),
            "r2_std": float(np.nanstd(r2_vals)),
            "spearman_mean": float(np.nanmean([r["spearman"] for r in results])),
        },
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "stage8_baseline.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"Saved report: {report_path.relative_to(PROJECT_ROOT)}")

    log("-" * 50)
    log("Stage 8 complete.")


if __name__ == "__main__":
    main()