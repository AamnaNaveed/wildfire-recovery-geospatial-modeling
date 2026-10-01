"""
Stage 6b: Build per-fire feature tables.

Loads aligned rasters, samples 20,000 random valid pixels per fire,
writes a Parquet per fire, and computes fire-level signal quality.

Signal classification rule:
  insufficient_data: n_valid < 1000
  invalid:           mean dNBR < 0.02 OR mean pre_NBR < 0.05
  strong:            mean dNBR >= 0.10 AND mean pre_NBR >= 0.10
  weak:              mean dNBR >= 0.02
Note: post_NBR is NOT required to be positive — severe burns legitimately
produce negative post-fire NBR.
"""

from pathlib import Path
import json
import gc
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import rioxarray
from rasterio.enums import Resampling


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIRES_DIR = PROJECT_ROOT / "data" / "interim" / "fires"
SELECTED_CSV = PROJECT_ROOT / "data" / "interim" / "selected_fires.csv"
FEATURES_DIR = PROJECT_ROOT / "data" / "interim" / "features"
REPORTS_DIR = PROJECT_ROOT / "reports"

SAMPLES_PER_FIRE = 20000
RANDOM_SEED = 42
MIN_VALID_PIXELS = 1000

OPTICAL_RASTERS = ["dnbr", "pre_nbr", "post_nbr", "pre_ndvi", "post_ndvi"]
AUX_RASTERS = ["dem", "slope", "aspect", "temp", "precip", "landcover"]


def log(msg):
    print(msg, flush=True)


def load_raster_stack(fire_dir):
    opened = {}
    shapes = []
    for name in OPTICAL_RASTERS + AUX_RASTERS:
        path = fire_dir / f"{name}.tif"
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}.tif in {fire_dir}")
        da = rioxarray.open_rasterio(path, masked=True).squeeze()
        opened[name] = da
        shapes.append(da.shape)

    modal_shape = Counter(shapes).most_common(1)[0][0]

    ref_da = None
    for name in OPTICAL_RASTERS + AUX_RASTERS:
        if opened[name].shape == modal_shape:
            ref_da = opened[name]
            break

    if ref_da is None:
        raise ValueError(f"No raster matches modal shape {modal_shape}")

    arrays = {}
    for name in OPTICAL_RASTERS + AUX_RASTERS:
        da = opened[name]
        if da.shape != modal_shape:
            log(f"      reprojecting {name} from {da.shape} to {modal_shape}")
            categorical = (name == "landcover")
            rs = Resampling.nearest if categorical else Resampling.bilinear
            da = da.rio.reproject_match(ref_da, resampling=rs)
        arrays[name] = da.values.astype("float32")
        da.close()

    return arrays, modal_shape, str(ref_da.rio.crs), ref_da.rio.transform()


def derive_aspect_components(aspect):
    rad = np.deg2rad(aspect)
    return np.sin(rad), np.cos(rad)


def classify_fire_signal(dnbr_mean, pre_nbr_mean, post_nbr_mean, n_valid):
    """
    Fire-level signal quality, based on mean values across all sampled pixels.
    post_NBR is NOT required to be positive — severe burns legitimately produce
    negative post-fire NBR.
    """
    if n_valid < MIN_VALID_PIXELS:
        return "insufficient_data"
    if not (np.isfinite(dnbr_mean) and np.isfinite(pre_nbr_mean)):
        return "invalid"
    if dnbr_mean < 0.02:
        return "invalid"
    if pre_nbr_mean < 0.05:
        return "invalid"
    if dnbr_mean >= 0.10 and pre_nbr_mean >= 0.10:
        return "strong"
    return "weak"


def sample_pixels(fire_dir, fire_meta, n_samples=SAMPLES_PER_FIRE, seed=RANDOM_SEED):
    arrays, shape, crs, transform = load_raster_stack(fire_dir)
    h, w = shape

    valid = np.ones(shape, dtype=bool)
    for name in ["dnbr", "pre_nbr", "post_nbr", "dem", "temp", "landcover"]:
        valid &= np.isfinite(arrays[name])

    n_valid = int(valid.sum())
    if n_valid == 0:
        raise ValueError(f"No valid pixels in {fire_dir}")

    n_take = min(n_samples, n_valid)
    rng = np.random.default_rng(seed)
    valid_flat = np.flatnonzero(valid.ravel())
    chosen = rng.choice(valid_flat, size=n_take, replace=False)
    rows, cols = np.unravel_index(chosen, shape)

    aspect = arrays["aspect"][rows, cols]
    aspect_sin, aspect_cos = derive_aspect_components(aspect)

    xs = transform.c + (cols + 0.5) * transform.a
    ys = transform.f + (rows + 0.5) * transform.e

    df = pd.DataFrame({
        "event_id": fire_meta["event_id"],
        "fire_name": fire_meta["fire_name"],
        "year": int(fire_meta["year"]),
        "ecosystem": fire_meta["ecosystem"],
        "dnbr": arrays["dnbr"][rows, cols].astype("float32"),
        "pre_nbr": arrays["pre_nbr"][rows, cols].astype("float32"),
        "post_nbr": arrays["post_nbr"][rows, cols].astype("float32"),
        "pre_ndvi": arrays["pre_ndvi"][rows, cols].astype("float32"),
        "post_ndvi": arrays["post_ndvi"][rows, cols].astype("float32"),
        "elevation": arrays["dem"][rows, cols].astype("float32"),
        "slope": arrays["slope"][rows, cols].astype("float32"),
        "aspect_sin": aspect_sin.astype("float32"),
        "aspect_cos": aspect_cos.astype("float32"),
        "temperature": arrays["temp"][rows, cols].astype("float32"),
        "precipitation": arrays["precip"][rows, cols].astype("float32"),
        "landcover_class": arrays["landcover"][rows, cols].astype("int32"),
        "x": xs.astype("float64"),
        "y": ys.astype("float64"),
    })

    fire_signal = classify_fire_signal(
        dnbr_mean=float(df["dnbr"].mean()),
        pre_nbr_mean=float(df["pre_nbr"].mean()),
        post_nbr_mean=float(df["post_nbr"].mean()),
        n_valid=n_valid,
    )
    df["fire_signal_quality"] = fire_signal

    return df, n_valid, fire_signal


def main():
    log("Stage 6b: Build per-fire feature tables")
    log("-" * 50)

    sel = pd.read_csv(SELECTED_CSV)
    fires = sel.to_dict(orient="records")
    log(f"Fires: {len(fires)}")

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    report_fires = []
    errors = []
    for fire in fires:
        event_id = fire["event_id"]
        fire_dir = FIRES_DIR / event_id
        try:
            log(f"  [{event_id}] {fire['fire_name']}")
            df, n_valid, fire_signal = sample_pixels(fire_dir, fire)
            out_path = FEATURES_DIR / f"{event_id}.parquet"
            df.to_parquet(out_path, index=False)
            log(f"    valid: {n_valid:,}, sampled: {len(df):,}, signal: {fire_signal}")
            report_fires.append({
                "event_id": event_id,
                "fire_name": fire["fire_name"],
                "ecosystem": fire["ecosystem"],
                "year": int(fire["year"]),
                "n_valid_pixels": n_valid,
                "n_sampled": len(df),
                "fire_signal_quality": fire_signal,
                "dnbr_mean": float(df["dnbr"].mean()),
                "pre_nbr_mean": float(df["pre_nbr"].mean()),
                "post_nbr_mean": float(df["post_nbr"].mean()),
                "parquet_path": str(out_path.relative_to(PROJECT_ROOT)),
            })
        except Exception as e:
            log(f"    ERROR: {type(e).__name__}: {e}")
            errors.append({"event_id": event_id, "error": str(e)})
        finally:
            gc.collect()

    signal_counts = Counter(r["fire_signal_quality"] for r in report_fires)

    report = {
        "stage": "6b",
        "timestamp": datetime.now().isoformat(),
        "n_fires": len(fires),
        "n_succeeded": len(report_fires),
        "n_failed": len(errors),
        "signal_summary": dict(signal_counts),
        "min_valid_pixels": MIN_VALID_PIXELS,
        "errors": errors,
        "fires": report_fires,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "feature_tables_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"Saved report: {report_path.relative_to(PROJECT_ROOT)}")

    log("-" * 50)
    log(f"Stage 6b complete: {len(report_fires)} succeeded, {len(errors)} failed")
    log(f"Fire-level signal: {dict(signal_counts)}")
    log("")
    for r in report_fires:
        log(f"  {r['fire_name']:25s} signal={r['fire_signal_quality']:18s} "
            f"dNBR={r['dnbr_mean']:.3f} pre_NBR={r['pre_nbr_mean']:.3f} "
            f"post_NBR={r['post_nbr_mean']:.3f} n={r['n_sampled']:,}")


if __name__ == "__main__":
    main()
