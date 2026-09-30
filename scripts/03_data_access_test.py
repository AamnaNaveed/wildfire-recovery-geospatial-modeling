"""
Stage 3: Data-access test.

Process one small fire (PLANADA, 2018-06-15) end-to-end using
Sentinel-2 L2A imagery from Microsoft Planetary Computer.

Builds seasonal composites (median across multiple scenes from ALL
tiles), applies cloud masks via SCL, computes pre-fire and post-fire
NDVI and NBR, derives dNBR, saves GeoTIFFs and a preview figure, and
writes a QA report.

Per PROJECT_PLAN.md section 15, row 3.
"""

from pathlib import Path
import json
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


# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "interim" / "fire_catalog.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "interim" / "fire_planada"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
QA_PATH = PROJECT_ROOT / "reports" / "data_access_test_qa.json"


# --- Fire of interest ---
FIRE_EVENT_ID = "CA3735012030020180615"

# --- Sentinel-2 bands we need ---
BANDS = ["B04", "B08", "B12", "SCL"]  # Red, NIR, SWIR2, Scene Classification
CONTINUOUS_BANDS = ["B04", "B08", "B12"]
CATEGORICAL_BANDS = ["SCL"]

# --- Cloud mask values in SCL band to exclude (loosened) ---
# 0 = no data, 9 = cloud high probability, 10 = thin cirrus
SCL_BAD_VALUES = [0, 9, 10]

# --- How many scenes to composite per TILE per window ---
MAX_SCENES_PER_TILE = 5

# --- Max scene cloud cover to consider ---
MAX_SCENE_CLOUD = 30


def load_fire():
    df = pd.read_csv(CATALOG_PATH)
    row = df[df["event_id"] == FIRE_EVENT_ID]
    if row.empty:
        raise ValueError(f"Fire {FIRE_EVENT_ID} not found in catalog")
    return row.iloc[0].to_dict()


def bbox_from_fire(fire, buffer_deg=0.05):
    lon = float(fire["longitude"])
    lat = float(fire["latitude"])
    return (lon - buffer_deg, lat - buffer_deg, lon + buffer_deg, lat + buffer_deg)


def search_sentinel2(bbox, date_range, max_cloud=MAX_SCENE_CLOUD):
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    return list(search.items())


def get_tile_id(item):
    grid_code = item.properties.get("grid:code")
    if grid_code:
        return grid_code
    for part in item.id.split("_"):
        if part.startswith("T") and len(part) == 6:
            return part
    return "unknown"


def select_scenes_by_tile(items, max_per_tile=MAX_SCENES_PER_TILE):
    by_tile = defaultdict(list)
    for item in items:
        by_tile[get_tile_id(item)].append(item)

    selected = []
    for tile, tile_items in by_tile.items():
        tile_items = sorted(tile_items, key=lambda i: i.properties.get("eo:cloud_cover", 100))
        selected.extend(tile_items[:max_per_tile])

    return selected, {tile: len(items_list) for tile, items_list in by_tile.items()}


def open_band_at_overview(href):
    for level in (3, 2, 4, 1, 0):
        try:
            da = rioxarray.open_rasterio(
                href, masked=True, overview_level=level
            ).squeeze()
            return da.load()
        except Exception:
            continue
    raise RuntimeError(f"Could not open {href} at any overview level")


def reproject_band(da, ref, band_name):
    """
    Reproject a band to the reference grid.
    Categorical bands (SCL) use nearest-neighbor; continuous bands use bilinear.
    """
    if band_name in CATEGORICAL_BANDS:
        return da.rio.reproject_match(ref, resampling=Resampling.nearest).load()
    return da.rio.reproject_match(ref).load()


def load_scene(item):
    arrays = {}
    for band in BANDS:
        asset = item.assets.get(band)
        if asset is None:
            return None
        arrays[band] = open_band_at_overview(asset.href)

    ref = arrays["B04"]
    for band in BANDS:
        if arrays[band].shape != ref.shape:
            arrays[band] = reproject_band(arrays[band], ref, band)

    ds = xr.Dataset({band: arrays[band] for band in BANDS})
    ds = ds.rio.write_crs(ref.rio.crs)
    ds = ds.rio.write_transform(ref.rio.transform())
    return ds


def build_composite(items, max_per_tile=MAX_SCENES_PER_TILE):
    if not items:
        raise ValueError("No items to composite")

    selected, tile_counts = select_scenes_by_tile(items, max_per_tile)
    print(f"    Tiles available: {tile_counts}")
    print(f"    Scenes selected: {len(selected)}")

    ref_ds = load_scene(selected[0])
    if ref_ds is None:
        raise ValueError(f"Could not load reference scene {selected[0].id}")
    ref_grid = ref_ds["B04"]
    print(f"    Reference grid from: {selected[0].id[:50]}")

    masked_scenes = []
    used_items = []
    for item in selected:
        try:
            ds = load_scene(item)
            if ds is None:
                continue
            aligned = {}
            for band in BANDS:
                da = ds[band]
                if da.shape != ref_grid.shape or da.rio.crs != ref_grid.rio.crs:
                    da = reproject_band(da, ref_grid, band)
                aligned[band] = da
            ds_aligned = xr.Dataset(aligned).rio.write_crs(ref_grid.rio.crs)
            ds_aligned = ds_aligned.rio.write_transform(ref_grid.rio.transform())

            scl = ds_aligned["SCL"]
            mask = ~scl.isin(SCL_BAD_VALUES)
            masked = ds_aligned.where(mask)
            masked_scenes.append(masked)
            used_items.append(item)

            valid_frac = float(mask.values.mean())
            print(f"    Added [{get_tile_id(item)}]: {item.id[:48]} (cloud {item.properties.get('eo:cloud_cover'):.4f}%, valid {valid_frac:.3f})")
        except Exception as e:
            print(f"    Skipping {item.id[:48]}: {e}")
            continue

    if not masked_scenes:
        raise ValueError("No scenes loaded successfully")

    print(f"    Compositing {len(masked_scenes)} scenes (median) ...")
    stacked = xr.concat(masked_scenes, dim="scene")
    composite = stacked.median(dim="scene", skipna=True)
    composite = composite.rio.write_crs(ref_grid.rio.crs)
    composite = composite.rio.write_transform(ref_grid.rio.transform())
    return composite, used_items, tile_counts


def compute_indices(ds):
    red = ds["B04"].astype("float32")
    nir = ds["B08"].astype("float32")
    swir = ds["B12"].astype("float32")
    ndvi = (nir - red) / (nir + red)
    nbr = (nir - swir) / (nir + swir)
    return ndvi.rename("NDVI"), nbr.rename("NBR")


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
    }


def save_geotiff(da, path):
    da = da.astype("float32")
    da.rio.to_raster(path)
    return str(path.relative_to(PROJECT_ROOT))


def make_figure(pre_nbr, post_nbr, dnbr, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, da, title in zip(
        axes,
        [pre_nbr, post_nbr, dnbr],
        ["Pre-fire NBR", "Post-fire NBR", "dNBR (pre - post)"],
    ):
        im = ax.imshow(da.values, cmap="RdYlGn", vmin=-1, vmax=1)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    print("Stage 3: Data-access test")
    print("-" * 50)

    fire = load_fire()
    print(f"Fire: {fire['fire_name']} ({fire['event_id']})")
    print(f"  Ignition: {fire['ignition_date']}")
    print(f"  Area (ha): {fire['burn_area_ha']}")

    bbox = bbox_from_fire(fire, buffer_deg=0.05)
    print(f"  Bounding box: {bbox}")

    ignition = pd.to_datetime(fire["ignition_date"])
    pre_start = (ignition - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    pre_end = (ignition - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    post_start = (ignition + pd.Timedelta(days=15)).strftime("%Y-%m-%d")
    post_end = (ignition + pd.Timedelta(days=105)).strftime("%Y-%m-%d")

    print(f"  Pre-fire window:  {pre_start} to {pre_end}")
    print(f"  Post-fire window: {post_start} to {post_end}")

    print("-" * 50)
    print("Searching Sentinel-2 L2A on Planetary Computer ...")
    pre_items = search_sentinel2(bbox, f"{pre_start}/{pre_end}")
    post_items = search_sentinel2(bbox, f"{post_start}/{post_end}")
    print(f"  Pre-fire scenes:  {len(pre_items)}")
    print(f"  Post-fire scenes: {len(post_items)}")

    if not pre_items or not post_items:
        raise ValueError("No imagery found in one of the windows")

    print("-" * 50)
    print("Building pre-fire composite (stratified by tile) ...")
    pre_ds, pre_used, pre_tile_counts = build_composite(pre_items)
    print(f"  Pre-fire composite shape: {pre_ds['B04'].shape}")

    print("-" * 50)
    print("Building post-fire composite (stratified by tile) ...")
    post_ds, post_used, post_tile_counts = build_composite(post_items)
    print(f"  Post-fire composite shape: {post_ds['B04'].shape}")

    pre_ndvi, pre_nbr = compute_indices(pre_ds)
    post_ndvi, post_nbr = compute_indices(post_ds)
    dnbr = (pre_nbr - post_nbr).rename("dNBR")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "pre_ndvi": save_geotiff(pre_ndvi, OUTPUT_DIR / "pre_ndvi.tif"),
        "post_ndvi": save_geotiff(post_ndvi, OUTPUT_DIR / "post_ndvi.tif"),
        "pre_nbr": save_geotiff(pre_nbr, OUTPUT_DIR / "pre_nbr.tif"),
        "post_nbr": save_geotiff(post_nbr, OUTPUT_DIR / "post_nbr.tif"),
        "dnbr": save_geotiff(dnbr, OUTPUT_DIR / "dnbr.tif"),
    }
    print("-" * 50)
    print("Saved GeoTIFFs:")
    for k, v in paths.items():
        print(f"  {k}: {v}")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURE_DIR / "data_access_test_preview.png"
    make_figure(pre_nbr, post_nbr, dnbr, fig_path)
    print(f"Saved figure: {fig_path.relative_to(PROJECT_ROOT)}")

    qa = {
        "stage": 3,
        "description": "Data-access test on one fire using Sentinel-2 L2A tile-stratified composites",
        "timestamp": datetime.now().isoformat(),
        "fire": {
            "event_id": fire["event_id"],
            "fire_name": fire["fire_name"],
            "ignition_date": fire["ignition_date"],
            "area_ha": float(fire["burn_area_ha"]),
            "bbox": list(bbox),
        },
        "windows": {
            "pre_fire": [pre_start, pre_end],
            "post_fire": [post_start, post_end],
        },
        "scl_mask": {
            "bad_values": SCL_BAD_VALUES,
            "note": "0=no data, 9=cloud high prob, 10=thin cirrus",
        },
        "tiles": {
            "pre_available": pre_tile_counts,
            "post_available": post_tile_counts,
        },
        "scenes_used": {
            "pre": [{"id": i.id, "tile": get_tile_id(i), "cloud_cover": i.properties.get("eo:cloud_cover")} for i in pre_used],
            "post": [{"id": i.id, "tile": get_tile_id(i), "cloud_cover": i.properties.get("eo:cloud_cover")} for i in post_used],
        },
        "scene_counts": {
            "pre_candidates": len(pre_items),
            "post_candidates": len(post_items),
            "pre_used": len(pre_used),
            "post_used": len(post_used),
        },
        "index_stats": {
            "pre_ndvi": summarize(pre_ndvi, "pre_ndvi"),
            "post_ndvi": summarize(post_ndvi, "post_ndvi"),
            "pre_nbr": summarize(pre_nbr, "pre_nbr"),
            "post_nbr": summarize(post_nbr, "post_nbr"),
            "dnbr": summarize(dnbr, "dnbr"),
        },
        "outputs": paths,
        "figure": str(fig_path.relative_to(PROJECT_ROOT)),
    }
    QA_PATH.parent.mkdir(parents=True, exist_ok=True)
    QA_PATH.write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(f"Saved QA report: {QA_PATH.relative_to(PROJECT_ROOT)}")

    print("-" * 50)
    print("Summary:")
    print(f"  Pre-fire NBR mean:  {qa['index_stats']['pre_nbr'].get('mean', float('nan')):.4f}")
    print(f"  Post-fire NBR mean: {qa['index_stats']['post_nbr'].get('mean', float('nan')):.4f}")
    print(f"  dNBR mean:          {qa['index_stats']['dnbr'].get('mean', float('nan')):.4f}")
    print(f"  Pre-fire valid fraction:  {qa['index_stats']['pre_nbr'].get('valid_fraction', 0):.4f}")
    print(f"  Post-fire valid fraction: {qa['index_stats']['post_nbr'].get('valid_fraction', 0):.4f}")
    print("Stage 3 complete.")


if __name__ == "__main__":
    main()
