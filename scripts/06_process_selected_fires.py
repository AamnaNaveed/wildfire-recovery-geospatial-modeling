"""
Stage 5, part 2: Process selected fires through the Sentinel-2 pipeline.

Uses MTBS pre_image_id and post_image_id dates to define per-fire windows
(+/- 30 days around each date). This matches MTBS methodology: same season,
year-apart imagery, avoiding burn-period overlap.

Batch 1 = first 5 fires, Batch 2 = remaining 10.
"""

from pathlib import Path
import json
import argparse
import gc
import time
import re
import traceback
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
import pystac_client
import planetary_computer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SELECTED_CSV = PROJECT_ROOT / "data" / "interim" / "selected_fires.csv"
PERIMETERS_CSV = PROJECT_ROOT / "data" / "interim" / "fire_perimeters.csv"
FIRES_DIR = PROJECT_ROOT / "data" / "interim" / "fires"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"


BATCH_SIZE = 5
BANDS = ["B04", "B08", "B12", "SCL"]
CATEGORICAL_BANDS = ["SCL"]
SCL_BAD_VALUES = [0, 9, 10]
MAX_SCENES_PER_TILE = 3
MAX_SCENE_CLOUD = 30
DNBR_CLIP = (-1.0, 1.0)
COMMON_CRS = "EPSG:5070"
WINDOW_DAYS = 30

MAX_RETRIES = 4
RETRY_WAIT = 3

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def log(msg):
    print(msg, flush=True)


# --- Date extraction from MTBS image ids ---
# Formats:
#   RANCH:     804503320170723        -> date = 20170723
#   KINCADE:   804503320190424        -> 20190424
#   MCKINNEY:  A10TEM20210830_30m     -> 20210830
_DATE_RE = re.compile(r"(20[0-9]{2})(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])")

def extract_date_from_image_id(image_id):
    """Return a pandas Timestamp or None."""
    if not isinstance(image_id, str):
        return None
    m = _DATE_RE.search(image_id)
    if not m:
        return None
    try:
        return pd.Timestamp(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
    except Exception:
        return None


def load_batch(batch_num):
    sel = pd.read_csv(SELECTED_CSV)
    perim = pd.read_csv(PERIMETERS_CSV)
    cat = pd.read_csv(PROJECT_ROOT / "data" / "interim" / "fire_catalog.csv")
    merged = sel.merge(perim, on="event_id", how="left")
    merged = merged.merge(
        cat[["event_id", "pre_image_id", "post_image_id"]],
        on="event_id", how="left",
    )
    if merged["bbox_minx"].isna().any():
        raise ValueError("Missing perimeter bboxes")
    if batch_num == 1:
        return merged.iloc[0:BATCH_SIZE].to_dict(orient="records")
    elif batch_num == 2:
        return merged.iloc[BATCH_SIZE:].to_dict(orient="records")
    raise ValueError(f"Unknown batch: {batch_num}")


_TILE_RE = re.compile(r"_(T[0-9]{2}[A-Z]{3})_")

def get_tile_id(item):
    m = _TILE_RE.search(item.id)
    if m:
        return m.group(1)
    gc_code = item.properties.get("grid:code")
    if gc_code:
        return gc_code
    return "unknown"


def select_scenes_by_tile(items, max_per_tile=MAX_SCENES_PER_TILE):
    by_tile = defaultdict(list)
    for item in items:
        by_tile[get_tile_id(item)].append(item)
    selected = []
    for tile, tile_items in by_tile.items():
        tile_items = sorted(tile_items, key=lambda i: i.properties.get("eo:cloud_cover", 100))
        selected.extend(tile_items[:max_per_tile])
    return selected, {tile: len(v) for tile, v in by_tile.items()}


def open_band_at_overview(href, overview_levels=(3, 2, 4, 1, 0)):
    last_err = None
    for attempt in range(MAX_RETRIES):
        retry_needed = False
        for level in overview_levels:
            try:
                da = rioxarray.open_rasterio(href, masked=True, overview_level=level).squeeze()
                return da.load()
            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                if any(x in err_str for x in ["403", "429", "forbidden", "rate", "timeout", "connection"]):
                    retry_needed = True
                    break
                continue
        if retry_needed and attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_WAIT * (attempt + 1))
            continue
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_WAIT * (attempt + 1))
            continue
        break
    raise RuntimeError(f"Could not open {href}: {last_err}")


def reproject_band_to_ref(da, ref, band_name):
    if band_name in CATEGORICAL_BANDS:
        return da.rio.reproject_match(ref, resampling=Resampling.nearest).load()
    return da.rio.reproject_match(ref).load()


def load_scene_bands(item):
    arrays = {}
    for band in BANDS:
        asset = item.assets.get(band)
        if asset is None:
            raise ValueError(f"Band {band} not in item")
        arrays[band] = open_band_at_overview(asset.href)
    return arrays


def build_composite(items, fire_name=""):
    if not items:
        raise ValueError(f"No items for {fire_name}")

    selected, tile_counts = select_scenes_by_tile(items)
    log(f"      Selected {len(selected)} scenes across tiles: {tile_counts}")

    by_tile = defaultdict(list)
    for item in selected:
        by_tile[get_tile_id(item)].append(item)

    tile_composites = []
    for tile_id, tile_items in by_tile.items():
        log(f"      Tile {tile_id}: {len(tile_items)} scenes")
        tile_arrays = []
        for i, item in enumerate(tile_items, 1):
            try:
                bands = load_scene_bands(item)
                tile_arrays.append(bands)
                log(f"        [{i}/{len(tile_items)}] {item.id[:48]}: OK")
            except Exception as e:
                log(f"        [{i}/{len(tile_items)}] {item.id[:48]}: FAILED {type(e).__name__}: {e}")
                continue

        if not tile_arrays:
            log(f"        Tile {tile_id}: no scenes loaded")
            continue

        ref = tile_arrays[0]["B04"]
        aligned_scenes = []
        for bands in tile_arrays:
            aligned = {}
            for band in BANDS:
                da = bands[band]
                if da.shape != ref.shape:
                    da = reproject_band_to_ref(da, ref, band)
                aligned[band] = da
            ds = xr.Dataset(aligned).rio.write_crs(ref.rio.crs)
            ds = ds.rio.write_transform(ref.rio.transform())
            mask = ~ds["SCL"].isin(SCL_BAD_VALUES)
            aligned_scenes.append(ds.where(mask))

        stacked = xr.concat(aligned_scenes, dim="scene")
        tile_comp = stacked.median(dim="scene", skipna=True)
        tile_comp = tile_comp.rio.write_crs(ref.rio.crs)
        tile_composites.append((tile_id, tile_comp))
        log(f"        Tile {tile_id}: composite shape {tile_comp['B04'].shape}")

    if not tile_composites:
        raise ValueError(f"No tile composites for {fire_name}")

    log(f"      Mosaicking {len(tile_composites)} tiles to {COMMON_CRS} ...")
    reprojected = []
    for tile_id, comp in tile_composites:
        comp_common = comp.rio.reproject(COMMON_CRS).load()
        reprojected.append(comp_common)

    target_grid = reprojected[0]["B04"]
    final_scenes = []
    for comp in reprojected:
        aligned = {}
        for band in BANDS:
            da = comp[band]
            if da.shape != target_grid.shape:
                da = reproject_band_to_ref(da, target_grid, band)
            aligned[band] = da
        final_scenes.append(xr.Dataset(aligned).rio.write_crs(COMMON_CRS))

    stacked = xr.concat(final_scenes, dim="tile")
    composite = stacked.median(dim="tile", skipna=True)
    composite = composite.rio.write_crs(COMMON_CRS)
    return composite


def compute_indices(ds):
    red = ds["B04"].astype("float32")
    nir = ds["B08"].astype("float32")
    swir = ds["B12"].astype("float32")
    ndvi = (nir - red) / (nir + red)
    nbr = (nir - swir) / (nir + swir)
    return ndvi.rename("NDVI"), nbr.rename("NBR")


def align_pair(pre_da, post_da):
    if pre_da.shape == post_da.shape and pre_da.rio.crs == post_da.rio.crs:
        pb = pre_da.rio.bounds()
        qb = post_da.rio.bounds()
        if all(abs(a - b) < 1 for a, b in zip(pb, qb)):
            return pre_da, post_da
    post_aligned = post_da.rio.reproject_match(pre_da, resampling=Resampling.bilinear).load()
    return pre_da, post_aligned


def spatial_coherence(da, threshold=0.1):
    arr = da.values
    valid = np.isfinite(arr)
    if valid.sum() == 0:
        return {"fraction_above_threshold": 0.0, "block_coherence": 0.0}
    frac_above = float((arr[valid] > threshold).sum() / valid.sum())
    h, w = arr.shape
    bh, bw = max(1, h // 10), max(1, w // 10)
    blocks = arr[:bh * 10, :bw * 10].reshape(10, bh, 10, bw).mean(axis=(1, 3))
    with np.errstate(all="ignore"):
        bc = float(np.nanmax(blocks) - np.nanmedian(blocks))
    if np.isnan(bc):
        bc = 0.0
    return {
        "fraction_above_threshold": round(frac_above, 4),
        "block_coherence": round(bc, 4),
    }


def summarize(da, name):
    arr = da.values
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return {"name": name, "valid_pixels": 0, "valid_fraction": 0.0}
    return {
        "name": name,
        "valid_pixels": int(valid.size),
        "total_pixels": int(arr.size),
        "valid_fraction": float(valid.size / arr.size),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
    }


def save_geotiff(da, path):
    da = da.astype("float32")
    path.parent.mkdir(parents=True, exist_ok=True)
    if da.rio.crs is None:
        raise ValueError(f"{da.name} has no CRS")
    da.rio.to_raster(path)


def search_sentinel2(bbox, date_range, max_cloud=MAX_SCENE_CLOUD):
    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    return list(search.items())


def process_one_fire(fire):
    event_id = fire["event_id"]
    fire_name = fire["fire_name"]

    pre_img_date = extract_date_from_image_id(fire["pre_image_id"])
    post_img_date = extract_date_from_image_id(fire["post_image_id"])
    if pre_img_date is None or post_img_date is None:
        raise ValueError(f"Could not parse MTBS dates for {event_id}: "
                         f"pre={fire['pre_image_id']}, post={fire['post_image_id']}")

    bbox = (
        float(fire["bbox_minx"]), float(fire["bbox_miny"]),
        float(fire["bbox_maxx"]), float(fire["bbox_maxy"]),
    )

    log(f"  [{event_id}] {fire_name} ({fire['ecosystem']}, {fire['year']})")
    log(f"    bbox: {[round(b,3) for b in bbox]}")
    log(f"    MTBS pre date:  {pre_img_date.date()}")
    log(f"    MTBS post date: {post_img_date.date()}")

    pre_start = (pre_img_date - pd.Timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d")
    pre_end = (pre_img_date + pd.Timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d")
    post_start = (post_img_date - pd.Timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d")
    post_end = (post_img_date + pd.Timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d")

    log(f"    pre window:  {pre_start} to {pre_end}")
    log(f"    post window: {post_start} to {post_end}")

    log(f"    Searching pre-fire ...")
    pre_items = search_sentinel2(bbox, f"{pre_start}/{pre_end}")
    log(f"      {len(pre_items)} scenes")
    log(f"    Searching post-fire ...")
    post_items = search_sentinel2(bbox, f"{post_start}/{post_end}")
    log(f"      {len(post_items)} scenes")

    if not pre_items or not post_items:
        raise ValueError(f"No imagery for {event_id}")

    log(f"    Building pre-fire composite ...")
    pre_ds = build_composite(pre_items, fire_name)
    log(f"      Pre-fire shape: {pre_ds['B04'].shape}")

    log(f"    Building post-fire composite ...")
    post_ds = build_composite(post_items, fire_name)
    log(f"      Post-fire shape: {post_ds['B04'].shape}")

    pre_ndvi, pre_nbr = compute_indices(pre_ds)
    post_ndvi, post_nbr = compute_indices(post_ds)
    pre_nbr, post_nbr = align_pair(pre_nbr, post_nbr)

    dnbr_raw = (pre_nbr - post_nbr).rename("dNBR")
    dnbr = dnbr_raw.clip(DNBR_CLIP[0], DNBR_CLIP[1])
    dnbr = dnbr.rio.write_crs(pre_nbr.rio.crs)

    out_dir = FIRES_DIR / event_id
    out_dir.mkdir(parents=True, exist_ok=True)
    save_geotiff(pre_ndvi, out_dir / "pre_ndvi.tif")
    save_geotiff(post_ndvi, out_dir / "post_ndvi.tif")
    save_geotiff(pre_nbr, out_dir / "pre_nbr.tif")
    save_geotiff(post_nbr, out_dir / "post_nbr.tif")
    save_geotiff(dnbr, out_dir / "dnbr.tif")

    coh = spatial_coherence(dnbr)

    meta = {
        "event_id": event_id,
        "fire_name": fire_name,
        "ecosystem": fire["ecosystem"],
        "year": int(fire["year"]),
        "ignition_date": fire["ignition_date"],
        "mtbs_pre_image_id": fire["pre_image_id"],
        "mtbs_post_image_id": fire["post_image_id"],
        "mtbs_pre_date": str(pre_img_date.date()),
        "mtbs_post_date": str(post_img_date.date()),
        "burn_area_ha": float(fire["burn_area_ha"]),
        "bbox": list(bbox),
        "common_crs": COMMON_CRS,
        "window_days": WINDOW_DAYS,
        "windows": {
            "pre_fire": [pre_start, pre_end],
            "post_fire": [post_start, post_end],
        },
        "stats": {
            "pre_nbr": summarize(pre_nbr, "pre_nbr"),
            "post_nbr": summarize(post_nbr, "post_nbr"),
            "dnbr": summarize(dnbr, "dnbr"),
            "pre_ndvi": summarize(pre_ndvi, "pre_ndvi"),
            "post_ndvi": summarize(post_ndvi, "post_ndvi"),
        },
        "coherence": coh,
        "processed_at": datetime.now().isoformat(),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def make_batch_figure(metas, out_path):
    n = len(metas)
    fig, axes = plt.subplots(n, 3, figsize=(14, 3.2 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    for i, meta in enumerate(metas):
        event_id = meta["event_id"]
        fire_dir = FIRES_DIR / event_id
        pre_nbr = rioxarray.open_rasterio(fire_dir / "pre_nbr.tif", masked=True).squeeze()
        post_nbr = rioxarray.open_rasterio(fire_dir / "post_nbr.tif", masked=True).squeeze()
        dnbr = rioxarray.open_rasterio(fire_dir / "dnbr.tif", masked=True).squeeze()
        for j, (da, title) in enumerate([(pre_nbr, "Pre NBR"), (post_nbr, "Post NBR"), (dnbr, "dNBR")]):
            axes[i, j].imshow(da.values, cmap="RdYlGn", vmin=-1, vmax=1)
            axes[i, j].set_title(f"{meta['fire_name']} - {title}", fontsize=9)
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
    plt.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, required=True, choices=[1, 2])
    args = parser.parse_args()

    log(f"Stage 5, part 2: Process selected fires (batch {args.batch})")
    log("-" * 50)

    fires = load_batch(args.batch)
    log(f"Processing {len(fires)} fires")

    metas = []
    errors = []
    for fire in fires:
        try:
            meta = process_one_fire(fire)
            metas.append(meta)
        except Exception as e:
            log(f"    ERROR on {fire['event_id']}: {type(e).__name__}: {e}")
            log(traceback.format_exc())
            errors.append({"event_id": fire["event_id"], "error": str(e), "type": type(e).__name__})
        finally:
            gc.collect()

    if metas:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        fig_path = FIGURE_DIR / f"batch{args.batch}_preview.png"
        make_batch_figure(metas, fig_path)
        log(f"Saved figure: {fig_path.relative_to(PROJECT_ROOT)}")

    report = {
        "stage": "5b",
        "batch": args.batch,
        "timestamp": datetime.now().isoformat(),
        "n_fires": len(fires),
        "n_succeeded": len(metas),
        "n_failed": len(errors),
        "errors": errors,
        "fires": [
            {
                "event_id": m["event_id"],
                "fire_name": m["fire_name"],
                "ecosystem": m["ecosystem"],
                "year": m["year"],
                "mtbs_pre_date": m["mtbs_pre_date"],
                "mtbs_post_date": m["mtbs_post_date"],
                "pre_nbr_mean": m["stats"]["pre_nbr"].get("mean"),
                "post_nbr_mean": m["stats"]["post_nbr"].get("mean"),
                "dNBR_mean": m["stats"]["dnbr"].get("mean"),
                "fraction_above_0.1": m["coherence"]["fraction_above_threshold"],
                "block_coherence": m["coherence"]["block_coherence"],
            }
            for m in metas
        ],
    }
    REPORT_PATH = REPORTS_DIR / f"batch{args.batch}_processing.json"
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"Saved report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")

    log("-" * 50)
    log(f"Batch {args.batch} complete: {len(metas)} succeeded, {len(errors)} failed")
    for m in metas:
        log(f"  {m['fire_name']}: pre={m['stats']['pre_nbr'].get('mean', 0):.3f} "
            f"post={m['stats']['post_nbr'].get('mean', 0):.3f} "
            f"dNBR={m['stats']['dnbr'].get('mean', 0):.3f} "
            f"frac>0.1={m['coherence']['fraction_above_threshold']:.3f}")


if __name__ == "__main__":
    main()
