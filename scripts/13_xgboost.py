"""
Stage 10: XGBoost with leave-one-fire-out cross-validation.

Trains an XGBoost regressor to predict dNBR from pre-fire vegetation,
terrain, climate, and landcover. Evaluates with leave-one-fire-out
cross-validation over the 10 usable fires.

Configuration fixed in PROTOCOL.md section 6.3:
  n_estimators=800, learning_rate=0.03, max_depth=6,
  min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
  reg_lambda=1, objective=reg:squarederror, random_state=42

No early stopping against test fires. Early stopping is only permitted
against training-fold validation splits, and is not used here (fixed
n_estimators as per protocol).

Per PROJECT_PLAN.md section 8.4 and section 15, row 10.
"""

from pathlib import Path
import json
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_ROOT / "data" / "interim" / "features"
TEST_FIRES_TXT = PROJECT_ROOT / "data" / "interim" / "test_fires.txt"
FEATURE_REPORT = PROJECT_ROOT / "reports" / "feature_tables_report.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"


USABLE_SIGNALS = {"strong", "weak"}

FEATURES = [
    "pre_nbr", "pre_ndvi",
    "elevation", "slope", "aspect_sin", "aspect_cos",
    "temperature", "precipitation",
    "landcover_class",
]
TARGET = "dnbr"

XGB_PARAMS = {
    "n_estimators": 800,
    "learning_rate": 0.03,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
}


def log(msg):
    print(msg, flush=True)


def load_usable_fires():
    report = json.loads(FEATURE_REPORT.read_text())
    return [f for f in report["fires"] if f["fire_signal_quality"] in USABLE_SIGNALS]


def load_test_fires():
    if not TEST_FIRES_TXT.exists():
        return []
    return [line.strip() for line in TEST_FIRES_TXT.read_text().splitlines() if line.strip()]


def load_features():
    usable = load_usable_fires()
    dfs = [pd.read_parquet(FEATURES_DIR / f"{f['event_id']}.parquet") for f in usable]
    return pd.concat(dfs, ignore_index=True), [f["event_id"] for f in usable]


def train_eval_fold(train_df, test_df):
    X_train = train_df[FEATURES].values
    y_train = train_df[TARGET].values
    X_test = test_df[FEATURES].values
    y_test = test_df[TARGET].values

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    try:
        rho, _ = spearmanr(y_test, y_pred)
        rho = float(rho) if np.isfinite(rho) else float("nan")
    except Exception:
        rho = float("nan")

    return {
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "target_mean": float(y_test.mean()),
        "target_std": float(y_test.std()),
        "pred_mean": float(y_pred.mean()),
        "pred_std": float(y_pred.std()),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "spearman": round(rho, 4),
    }, model, y_test, y_pred


def make_figure(results, importances_arr, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    fires = [r["held_out_fire"] for r in results]
    rmse = [r["rmse"] for r in results]
    r2 = [r["r2"] for r in results]

    axes[0].barh(fires, rmse)
    axes[0].set_xlabel("RMSE")
    axes[0].set_title("XGBoost: RMSE per held-out fire")
    axes[0].tick_params(axis="y", labelsize=8)

    axes[1].barh(fires, r2)
    axes[1].set_xlabel("R²")
    axes[1].set_title("XGBoost: R² per held-out fire")
    axes[1].tick_params(axis="y", labelsize=8)

    importances = importances_arr.mean(axis=0)
    idx = np.argsort(importances)
    axes[2].barh([FEATURES[i] for i in idx], importances[idx])
    axes[2].set_xlabel("Mean importance across folds")
    axes[2].set_title("XGBoost: Feature importance")

    plt.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main():
    log("Stage 10: XGBoost")
    log("-" * 50)

    df_all, fire_ids = load_features()
    log(f"Total rows: {len(df_all):,}")
    log(f"Fires: {len(fire_ids)}")

    test_ids = load_test_fires()
    log(f"Frozen test fires: {test_ids}")

    log("-" * 50)
    log("Running leave-one-fire-out ...")

    results = []
    importances = []
    for i, held_out in enumerate(fire_ids, 1):
        train_df = df_all[df_all["event_id"] != held_out]
        test_df = df_all[df_all["event_id"] == held_out]
        log(f"  [{i}/{len(fire_ids)}] hold out {held_out} "
            f"(train {len(train_df):,}, test {len(test_df):,})")

        res, model, y_true, y_pred = train_eval_fold(train_df, test_df)
        importances.append(model.feature_importances_)
        res["held_out_fire"] = held_out
        res["ecosystem"] = test_df["ecosystem"].iloc[0]
        results.append(res)

    log("")
    log(f"{'Fire':25s} {'Eco':10s} {'RMSE':>8s} {'MAE':>8s} {'R²':>8s} {'Rho':>8s}")
    for r in results:
        log(f"{r['held_out_fire']:25s} {r['ecosystem']:10s} "
            f"{r['rmse']:>8.4f} {r['mae']:>8.4f} {r['r2']:>8.4f} {r['spearman']:>8.4f}")

    rmse_vals = [r["rmse"] for r in results]
    r2_vals = [r["r2"] for r in results]
    rho_vals = [r["spearman"] for r in results]

    log("")
    log(f"RMSE mean ± SD: {np.mean(rmse_vals):.4f} ± {np.std(rmse_vals):.4f}")
    log(f"R²   mean ± SD: {np.nanmean(r2_vals):.4f} ± {np.nanstd(r2_vals):.4f}")
    log(f"Rho  mean ± SD: {np.nanmean(rho_vals):.4f} ± {np.nanstd(rho_vals):.4f}")

    importances_arr = np.array(importances)
    mean_importance = importances_arr.mean(axis=0)
    log("")
    log("Feature importance (mean across folds):")
    order = np.argsort(mean_importance)[::-1]
    for i in order:
        log(f"  {FEATURES[i]:20s} {mean_importance[i]:.4f}")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURE_DIR / "stage10_xgboost.png"
    make_figure(results, importances_arr, fig_path)
    log(f"Saved figure: {fig_path.relative_to(PROJECT_ROOT)}")

    report = {
        "stage": 10,
        "model": "xgboost",
        "params": XGB_PARAMS,
        "features": FEATURES,
        "timestamp": datetime.now().isoformat(),
        "n_fires": len(fire_ids),
        "n_rows_total": int(len(df_all)),
        "test_fires": test_ids,
        "per_fire_results": results,
        "aggregate": {
            "rmse_mean": float(np.mean(rmse_vals)),
            "rmse_std": float(np.std(rmse_vals)),
            "mae_mean": float(np.mean([r["mae"] for r in results])),
            "r2_mean": float(np.nanmean(r2_vals)),
            "r2_std": float(np.nanstd(r2_vals)),
            "spearman_mean": float(np.nanmean(rho_vals)),
            "spearman_std": float(np.nanstd(rho_vals)),
        },
        "feature_importance_mean": {
            FEATURES[i]: float(mean_importance[i]) for i in range(len(FEATURES))
        },
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "stage10_xgboost.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"Saved report: {report_path.relative_to(PROJECT_ROOT)}")

    log("-" * 50)
    log("Stage 10 complete.")


if __name__ == "__main__":
    main()