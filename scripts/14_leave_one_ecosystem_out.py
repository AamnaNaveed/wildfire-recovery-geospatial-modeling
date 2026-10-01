"""
Stage 11: Leave-one-ecosystem-out cross-validation.

Train on two ecosystem groups, predict the third. This tests whether
the model generalizes across ecosystem types, not just across fires.

Runs for both Random Forest and XGBoost.

Per PROJECT_PLAN.md section 9 and section 15, row 11.
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
FEATURE_REPORT = PROJECT_ROOT / "reports" / "feature_tables_report.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

USABLE_SIGNALS = {"strong", "weak"}
ECOSYSTEMS = ["forest", "shrubland", "grassland"]

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


def load_usable():
    report = json.loads(FEATURE_REPORT.read_text())
    return [f for f in report["fires"] if f["fire_signal_quality"] in USABLE_SIGNALS]


def load_features():
    usable = load_usable()
    dfs = [pd.read_parquet(FEATURES_DIR / f"{f['event_id']}.parquet") for f in usable]
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
        "spearman": round(rho, 4),
    }


def evaluate_loeo(df_all, model_factory, model_name):
    """Hold out each ecosystem, train on the other two."""
    results = []
    for held_out_eco in ECOSYSTEMS:
        train = df_all[df_all["ecosystem"] != held_out_eco]
        test = df_all[df_all["ecosystem"] == held_out_eco]

        if len(train) == 0 or len(test) == 0:
            log(f"  [skip] {held_out_eco}: empty train or test")
            continue

        X_train = train[FEATURES].values
        y_train = train[TARGET].values
        X_test = test[FEATURES].values
        y_test = test[TARGET].values

        model = model_factory()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        m = metrics(y_test, y_pred)
        m.update({
            "model": model_name,
            "held_out_ecosystem": held_out_eco,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "train_ecosystems": sorted(train["ecosystem"].unique().tolist()),
            "target_mean": float(y_test.mean()),
            "target_std": float(y_test.std()),
            "pred_mean": float(y_pred.mean()),
            "pred_std": float(y_pred.std()),
        })
        results.append(m)

        log(f"  [{model_name}] hold out {held_out_eco}: "
            f"train={len(train):,}, test={len(test):,}, "
            f"RMSE={m['rmse']:.4f}, R²={m['r2']:.4f}, Rho={m['spearman']:.4f}")

    return results


def make_figure(all_results, out_path):
    """Two panels: RMSE and R² for each model x held-out ecosystem."""
    df = pd.DataFrame(all_results)
    ecosystems = df["held_out_ecosystem"].unique()
    models = df["model"].unique()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    width = 0.35
    x = np.arange(len(ecosystems))

    for i, model_name in enumerate(models):
        sub = df[df["model"] == model_name].set_index("held_out_ecosystem")
        sub = sub.reindex(ecosystems)
        axes[0].bar(x + i * width, sub["rmse"], width, label=model_name)
        axes[1].bar(x + i * width, sub["r2"], width, label=model_name)

    axes[0].set_xticks(x + width / 2)
    axes[0].set_xticklabels(ecosystems)
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("Leave-one-ecosystem-out: RMSE")
    axes[0].legend()

    axes[1].set_xticks(x + width / 2)
    axes[1].set_xticklabels(ecosystems)
    axes[1].set_ylabel("R²")
    axes[1].set_title("Leave-one-ecosystem-out: R²")
    axes[1].axhline(0, color="grey", linestyle="--", linewidth=0.5)
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main():
    log("Stage 11: Leave-one-ecosystem-out")
    log("-" * 50)

    df_all = load_features()
    log(f"Total rows: {len(df_all):,}")
    log(f"Ecosystem counts: {df_all.groupby('ecosystem').size().to_dict()}")

    log("-" * 50)
    log("Random Forest ...")
    rf_results = evaluate_loeo(df_all, lambda: RandomForestRegressor(**RF_PARAMS), "RandomForest")

    log("-" * 50)
    log("XGBoost ...")
    xgb_results = evaluate_loeo(df_all, lambda: xgb.XGBRegressor(**XGB_PARAMS), "XGBoost")

    all_results = rf_results + xgb_results

    # Aggregate summaries
    def agg(results, model_name):
        sub = [r for r in results if r["model"] == model_name]
        if not sub:
            return {}
        return {
            "rmse_mean": float(np.mean([r["rmse"] for r in sub])),
            "rmse_std": float(np.std([r["rmse"] for r in sub])),
            "r2_mean": float(np.mean([r["r2"] for r in sub])),
            "r2_std": float(np.std([r["r2"] for r in sub])),
            "spearman_mean": float(np.nanmean([r["spearman"] for r in sub])),
        }

    # Figure
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURE_DIR / "stage11_loeo.png"
    make_figure(all_results, fig_path)
    log(f"Saved figure: {fig_path.relative_to(PROJECT_ROOT)}")

    report = {
        "stage": 11,
        "description": "Leave-one-ecosystem-out cross-validation",
        "timestamp": datetime.now().isoformat(),
        "features": FEATURES,
        "n_rows_total": int(len(df_all)),
        "per_fold_results": all_results,
        "aggregate_rf": agg(all_results, "RandomForest"),
        "aggregate_xgb": agg(all_results, "XGBoost"),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "stage11_loeo.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"Saved report: {report_path.relative_to(PROJECT_ROOT)}")

    log("-" * 50)
    log("Summary:")
    log(f"  RF  RMSE mean ± SD: {report['aggregate_rf']['rmse_mean']:.4f} ± {report['aggregate_rf']['rmse_std']:.4f}")
    log(f"  XGB RMSE mean ± SD: {report['aggregate_xgb']['rmse_mean']:.4f} ± {report['aggregate_xgb']['rmse_std']:.4f}")
    log(f"  RF  R²   mean ± SD: {report['aggregate_rf']['r2_mean']:.4f} ± {report['aggregate_rf']['r2_std']:.4f}")
    log(f"  XGB R²   mean ± SD: {report['aggregate_xgb']['r2_mean']:.4f} ± {report['aggregate_xgb']['r2_std']:.4f}")
    log("Stage 11 complete.")


if __name__ == "__main__":
    main()