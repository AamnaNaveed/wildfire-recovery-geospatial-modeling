"""
Stage 12: Uncertainty quantification.

For each model (RF, XGBoost), train five models with different random
seeds on the same training split and measure prediction variance.
Report mean ± SD per held-out fire, prediction intervals, and the
correlation between uncertainty and prediction error.

Uses leave-one-fire-out split for consistency with Stage 9/10.

Per PROJECT_PLAN.md section 9 and section 15, row 12.
"""

from pathlib import Path
import json
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_ROOT / "data" / "interim" / "features"
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
SEEDS = [42, 123, 7, 256, 999]

RF_BASE = {
    "n_estimators": 500,
    "max_features": "sqrt",
    "min_samples_leaf": 5,
    "n_jobs": -1,
}

XGB_BASE = {
    "n_estimators": 800,
    "learning_rate": 0.03,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "n_jobs": -1,
    "tree_method": "hist",
}


def log(msg):
    print(msg, flush=True)


def load_usable_fires():
    report = json.loads(FEATURE_REPORT.read_text())
    return [f for f in report["fires"] if f["fire_signal_quality"] in USABLE_SIGNALS]


def load_features():
    usable = load_usable_fires()
    dfs = [pd.read_parquet(FEATURES_DIR / f"{f['event_id']}.parquet") for f in usable]
    return pd.concat(dfs, ignore_index=True), [f["event_id"] for f in usable]


def train_seeds(model_type, X_train, y_train, X_test, seeds=SEEDS):
    """Train model with multiple seeds, return array of predictions (n_seeds, n_test)."""
    preds = []
    for seed in seeds:
        if model_type == "rf":
            params = {**RF_BASE, "random_state": seed}
            model = RandomForestRegressor(**params)
        else:
            params = {**XGB_BASE, "random_state": seed}
            model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        preds.append(model.predict(X_test))
    return np.array(preds)


def summarize_fold(model_type, held_out, y_true, preds):
    """preds: shape (n_seeds, n_test)."""
    pred_mean = preds.mean(axis=0)
    pred_std = preds.std(axis=0)

    rmse_mean = float(np.sqrt(mean_squared_error(y_true, pred_mean)))

    # Per-pixel absolute error of the mean prediction
    abs_err = np.abs(y_true - pred_mean)

    # Correlation between uncertainty and error (higher = uncertainty tracks error)
    if pred_std.std() > 0:
        rho, _ = spearmanr(pred_std, abs_err)
        rho = float(rho) if np.isfinite(rho) else float("nan")
    else:
        rho = float("nan")

    # 95% prediction interval width (assume ~normal; ±1.96 * std)
    pi_width_mean = float((1.96 * pred_std * 2).mean())

    # Coverage: fraction of y_true within ±1.96*std of pred_mean
    lower = pred_mean - 1.96 * pred_std
    upper = pred_mean + 1.96 * pred_std
    coverage = float(((y_true >= lower) & (y_true <= upper)).mean())

    return {
        "model": model_type,
        "held_out_fire": held_out,
        "n_test": int(len(y_true)),
        "rmse_mean_pred": round(rmse_mean, 4),
        "pred_std_mean": round(float(pred_std.mean()), 4),
        "pred_std_max": round(float(pred_std.max()), 4),
        "pi_width_mean": round(pi_width_mean, 4),
        "coverage_95": round(coverage, 4),
        "uncertainty_error_spearman": round(rho, 4) if np.isfinite(rho) else None,
        "target_std": round(float(y_true.std()), 4),
    }


def make_figure(all_results, out_path):
    df = pd.DataFrame(all_results)
    fires = df["held_out_fire"].unique()
    models = df["model"].unique()

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    x = np.arange(len(fires))
    width = 0.35

    for i, model_name in enumerate(models):
        sub = df[df["model"] == model_name].set_index("held_out_fire").reindex(fires)
        axes[0].bar(x + i * width, sub["pred_std_mean"], width, label=model_name)
        axes[1].bar(x + i * width, sub["coverage_95"], width, label=model_name)
        axes[2].bar(x + i * width, sub["uncertainty_error_spearman"], width, label=model_name)

    axes[0].set_xticks(x + width / 2); axes[0].set_xticklabels(fires, rotation=45, ha="right", fontsize=7)
    axes[0].set_ylabel("Mean prediction std")
    axes[0].set_title("Mean uncertainty per fire")
    axes[0].legend()

    axes[1].set_xticks(x + width / 2); axes[1].set_xticklabels(fires, rotation=45, ha="right", fontsize=7)
    axes[1].set_ylabel("Fraction of y in 95% PI")
    axes[1].axhline(0.95, color="red", linestyle="--", linewidth=1, label="Nominal 95%")
    axes[1].set_title("Prediction interval coverage")
    axes[1].legend()

    axes[2].set_xticks(x + width / 2); axes[2].set_xticklabels(fires, rotation=45, ha="right", fontsize=7)
    axes[2].set_ylabel("Spearman(std, |err|)")
    axes[2].set_title("Uncertainty-error correlation")
    axes[2].axhline(0, color="grey", linestyle="--", linewidth=0.5)
    axes[2].legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main():
    log("Stage 12: Uncertainty quantification")
    log("-" * 50)

    df_all, fire_ids = load_features()
    log(f"Total rows: {len(df_all):,}, fires: {len(fire_ids)}")
    log(f"Seeds: {SEEDS}")

    all_results = []
    for model_type in ["rf", "xgb"]:
        log(f"\n=== {model_type.upper()} ===")
        for i, held_out in enumerate(fire_ids, 1):
            train = df_all[df_all["event_id"] != held_out]
            test = df_all[df_all["event_id"] == held_out]
            X_train = train[FEATURES].values
            y_train = train[TARGET].values
            X_test = test[FEATURES].values
            y_test = test[TARGET].values

            log(f"  [{i}/{len(fire_ids)}] hold out {held_out} ({len(test)} test rows, {len(SEEDS)} seeds)")
            preds = train_seeds(model_type, X_train, y_train, X_test)
            res = summarize_fold(model_type, held_out, y_test, preds)
            res["ecosystem"] = test["ecosystem"].iloc[0]
            all_results.append(res)

    # Summary table
    log("")
    log(f"{'Model':5s} {'Fire':25s} {'Eco':10s} {'pred_std':>9s} {'PI95':>7s} {'cover':>7s} {'rho':>7s}")
    for r in all_results:
        log(f"{r['model']:5s} {r['held_out_fire']:25s} {r['ecosystem']:10s} "
            f"{r['pred_std_mean']:>9.4f} {r['pi_width_mean']:>7.4f} {r['coverage_95']:>7.4f} "
            f"{(r['uncertainty_error_spearman'] or float('nan')):>7.4f}")

    # Aggregate
    for model_type in ["rf", "xgb"]:
        sub = [r for r in all_results if r["model"] == model_type]
        log("")
        log(f"{model_type.upper()} aggregates:")
        log(f"  mean prediction std:      {np.mean([r['pred_std_mean'] for r in sub]):.4f}")
        log(f"  mean PI width (95%):      {np.mean([r['pi_width_mean'] for r in sub]):.4f}")
        log(f"  mean coverage (95%):      {np.mean([r['coverage_95'] for r in sub]):.4f}")
        log(f"  mean uncertainty-error ρ: {np.nanmean([r['uncertainty_error_spearman'] for r in sub if r['uncertainty_error_spearman'] is not None]):.4f}")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURE_DIR / "stage12_uncertainty.png"
    make_figure(all_results, fig_path)
    log(f"\nSaved figure: {fig_path.relative_to(PROJECT_ROOT)}")

    report = {
        "stage": 12,
        "description": "Uncertainty quantification across 5 seeds",
        "seeds": SEEDS,
        "features": FEATURES,
        "n_rows_total": int(len(df_all)),
        "per_fire_results": all_results,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "stage12_uncertainty.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"Saved report: {report_path.relative_to(PROJECT_ROOT)}")

    log("-" * 50)
    log("Stage 12 complete.")


if __name__ == "__main__":
    main()