"""
Stage 14: Final test evaluation.

Evaluate the frozen Random Forest and XGBoost models on the three
pre-registered test fires (CREEK, BOBCAT, TUCKER). This is the
single evaluation run permitted on the test set.

Test fires were frozen in Stage 7 and are read from
data/interim/test_fires.txt. No tuning is performed in this script.

Both models were trained on the other 7 usable fires.

Per PROJECT_PLAN.md section 10 and section 15, row 14.
"""

from pathlib import Path
import json
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
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

RF_PARAMS = {
    "n_estimators": 500,
    "max_features": "sqrt",
    "min_samples_leaf": 5,
    "random_state": 42,
    "n_jobs": -1,
}

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
    return [line.strip() for line in TEST_FIRES_TXT.read_text().splitlines() if line.strip()]


def load_features(event_ids):
    dfs = [pd.read_parquet(FEATURES_DIR / f"{e}.parquet") for e in event_ids]
    return pd.concat(dfs, ignore_index=True)


def metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    try:
        rho, _ = spearmanr(y_true, y_pred)
        rho = float(rho) if np.isfinite(rho) else float("nan")
    except Exception:
        rho = float("nan")
    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "spearman": round(rho, 4) if np.isfinite(rho) else None,
    }


def make_figure(all_results, out_path):
    df = pd.DataFrame(all_results)
    fires = df["fire_name"].unique()
    models = df["model"].unique()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    x = np.arange(len(fires))
    width = 0.35

    for i, model_name in enumerate(models):
        sub = df[df["model"] == model_name].set_index("fire_name").reindex(fires)
        axes[0].bar(x + i * width, sub["rmse"], width, label=model_name)
        axes[1].bar(x + i * width, sub["r2"], width, label=model_name)
        axes[2].bar(x + i * width, sub["spearman"], width, label=model_name)

    for ax, title, ylabel in zip(
        axes,
        ["Final test: RMSE", "Final test: R²", "Final test: Spearman"],
        ["RMSE", "R²", "Spearman"],
    ):
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(fires)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.tick_params(axis="x", labelsize=8)
        if ylabel in ("R²", "Spearman"):
            ax.axhline(0, color="grey", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main():
    log("Stage 14: Final test evaluation")
    log("-" * 50)

    usable = load_usable_fires()
    usable_ids = [f["event_id"] for f in usable]
    log(f"Usable fires: {len(usable_ids)}")

    test_ids = load_test_fires()
    train_ids = [e for e in usable_ids if e not in test_ids]

    log(f"Test fires (frozen):  {test_ids}")
    log(f"Training fires:       {train_ids}")

    if not set(test_ids).issubset(set(usable_ids)):
        missing = set(test_ids) - set(usable_ids)
        raise ValueError(f"Test fires not in usable pool: {missing}")

    log("-" * 50)
    log("Loading feature tables ...")
    train_df = load_features(train_ids)
    test_df = load_features(test_ids)
    log(f"  train rows: {len(train_df):,}")
    log(f"  test rows:  {len(test_df):,}")

    X_train = train_df[FEATURES].values
    y_train = train_df[TARGET].values

    results = []

    # Random Forest
    log("-" * 50)
    log("Random Forest ...")
    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X_train, y_train)
    for event_id in test_ids:
        test_fire = test_df[test_df["event_id"] == event_id]
        y_true = test_fire[TARGET].values
        y_pred = rf.predict(test_fire[FEATURES].values)
        m = metrics(y_true, y_pred)
        m.update({
            "model": "RandomForest",
            "event_id": event_id,
            "fire_name": test_fire["fire_name"].iloc[0],
            "ecosystem": test_fire["ecosystem"].iloc[0],
            "n_pixels": int(len(test_fire)),
            "target_mean": round(float(y_true.mean()), 4),
            "pred_mean": round(float(y_pred.mean()), 4),
        })
        results.append(m)
        log(f"  {m['fire_name']:15s} RMSE={m['rmse']:.4f} R²={m['r2']:.4f} Rho={m['spearman']}")

    # XGBoost
    log("-" * 50)
    log("XGBoost ...")
    xgb_model = xgb.XGBRegressor(**XGB_PARAMS)
    xgb_model.fit(X_train, y_train)
    for event_id in test_ids:
        test_fire = test_df[test_df["event_id"] == event_id]
        y_true = test_fire[TARGET].values
        y_pred = xgb_model.predict(test_fire[FEATURES].values)
        m = metrics(y_true, y_pred)
        m.update({
            "model": "XGBoost",
            "event_id": event_id,
            "fire_name": test_fire["fire_name"].iloc[0],
            "ecosystem": test_fire["ecosystem"].iloc[0],
            "n_pixels": int(len(test_fire)),
            "target_mean": round(float(y_true.mean()), 4),
            "pred_mean": round(float(y_pred.mean()), 4),
        })
        results.append(m)
        log(f"  {m['fire_name']:15s} RMSE={m['rmse']:.4f} R²={m['r2']:.4f} Rho={m['spearman']}")

    # Summary
    log("-" * 50)
    log("Summary by model (across 3 test fires):")
    for model_name in ["RandomForest", "XGBoost"]:
        sub = [r for r in results if r["model"] == model_name]
        rmse_m = float(np.mean([r["rmse"] for r in sub]))
        r2_m = float(np.mean([r["r2"] for r in sub]))
        rho_vals = [r["spearman"] for r in sub if r["spearman"] is not None]
        rho_m = float(np.mean(rho_vals)) if rho_vals else float("nan")
        log(f"  {model_name:15s} RMSE={rmse_m:.4f}  R²={r2_m:.4f}  Rho={rho_m:.4f}")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURE_DIR / "stage14_final_test.png"
    make_figure(results, fig_path)
    log(f"Saved figure: {fig_path.relative_to(PROJECT_ROOT)}")

    # Persistence baseline on test fires for comparison
    log("-" * 50)
    log("Persistence baseline on test fires (reference):")
    eco_means = train_df.groupby("ecosystem")[TARGET].mean().to_dict()
    global_mean = float(train_df[TARGET].mean())
    baseline_results = []
    for event_id in test_ids:
        test_fire = test_df[test_df["event_id"] == event_id]
        y_true = test_fire[TARGET].values
        eco = test_fire["ecosystem"].iloc[0]
        pred_value = eco_means.get(eco, global_mean)
        y_pred = np.full_like(y_true, pred_value, dtype=float)
        m = metrics(y_true, y_pred)
        m["fire_name"] = test_fire["fire_name"].iloc[0]
        m["model"] = "persistence"
        baseline_results.append(m)
        log(f"  {m['fire_name']:15s} RMSE={m['rmse']:.4f} R²={m['r2']:.4f}")

    report = {
        "stage": 14,
        "description": "Final test evaluation on frozen test fires",
        "timestamp": datetime.now().isoformat(),
        "test_fires": test_ids,
        "train_fires": train_ids,
        "features": FEATURES,
        "rf_params": RF_PARAMS,
        "xgb_params": XGB_PARAMS,
        "results": results,
        "baseline_results": baseline_results,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "stage14_final_test.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"Saved report: {report_path.relative_to(PROJECT_ROOT)}")

    log("-" * 50)
    log("Stage 14 complete.")


if __name__ == "__main__":
    main()