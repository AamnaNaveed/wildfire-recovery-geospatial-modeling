"""
Stage 5, part 1: Classify fires by ecosystem.

For each fire in the California catalog, sample ESA WorldCover land cover
within a 3 km buffer around the ignition point, determine the majority
class, and map it to one of three ecosystem groups (forest, shrubland,
grassland).

Then select 15 fires (5 per ecosystem group) stratified across years
and geographic diversity.

Outputs:
  reports/fire_classification.csv  - all 145 fires with their class
  data/interim/selected_fires.csv  - the 15 selected fires
  reports/fire_selection.json      - selection rationale and stats

Per PROJECT_PLAN.md section 6 and section 15, row 5.
"""

from pathlib import Path
import json
import glob
from datetime import datetime

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds


# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "interim" / "fire_catalog.csv"
WORLDCOVER_DIR = PROJECT_ROOT / "data" / "raw"
CLASSIFICATION_CSV = PROJECT_ROOT / "reports" / "fire_classification.csv"
SELECTED_CSV = PROJECT_ROOT / "data" / "interim" / "selected_fires.csv"
SELECTION_JSON = PROJECT_ROOT / "reports" / "fire_selection.json"


# --- WorldCover class mapping ---
WORLDCOVER_LABELS = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}

ECOSYSTEM_MAP = {
    10: "forest",
    20: "shrubland",
    30: "grassland",
}

SAMPLE_BUFFER_M = 3000
N_PER_ECOSYSTEM = 5


def load_catalog():
    return pd.read_csv(CATALOG_PATH)


def find_tile_for_fire(fire, tiles):
    """Return the tile path whose bounds contain the fire's ignition point."""
    lon = float(fire["longitude"])
    lat = float(fire["latitude"])
    for tile_path in tiles:
        with rasterio.open(tile_path) as src:
            b = src.bounds
            if b.left <= lon <= b.right and b.bottom <= lat <= b.top:
                return tile_path
    return None


def sample_worldcover_window(fire, tile_path, buffer_m=SAMPLE_BUFFER_M):
    """
    Read a small window from the WorldCover tile around the fire.
    Uses rasterio.windows.from_bounds — fast and memory-safe.
    Returns (majority_class, class_counts, sample_size).
    """
    lon = float(fire["longitude"])
    lat = float(fire["latitude"])

    # Buffer in degrees. At CA latitudes: ~111 km/deg lat, ~87 km/deg lon.
    dlat = buffer_m / 111_000
    dlon = buffer_m / 87_000

    minx = lon - dlon
    maxx = lon + dlon
    miny = lat - dlat
    maxy = lat + dlat

    with rasterio.open(tile_path) as src:
        # Clip the window to the tile's own bounds
        b = src.bounds
        minx = max(minx, b.left)
        maxx = min(maxx, b.right)
        miny = max(miny, b.bottom)
        maxy = min(maxy, b.top)

        if minx >= maxx or miny >= maxy:
            return None, {}, 0

        # Build a pixel-aligned window
        window = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
        window = window.round_offsets().round_lengths()

        # Read only that window
        data = src.read(1, window=window)
        nodata = src.nodata

    arr = data.ravel()
    if nodata is not None:
        arr = arr[arr != nodata]
    arr = arr[arr > 0]
    if arr.size == 0:
        return None, {}, 0

    # Count classes
    values, counts = np.unique(arr.astype(int), return_counts=True)
    class_counts = dict(zip(values.tolist(), counts.tolist()))

    # Keep only our three ecosystem classes
    kept = {k: v for k, v in class_counts.items() if k in ECOSYSTEM_MAP}
    if not kept:
        return None, class_counts, int(arr.size)

    majority = max(kept, key=kept.get)
    return majority, class_counts, int(arr.size)


def classify_all_fires(catalog, tiles):
    rows = []
    total = len(catalog)

    for i, (_, fire) in enumerate(catalog.iterrows(), 1):
        tile = find_tile_for_fire(fire, tiles)
        if tile is None:
            majority, counts, n = None, {}, 0
            tile_name = None
        else:
            majority, counts, n = sample_worldcover_window(fire, tile)
            tile_name = Path(tile).name

        ecosystem = ECOSYSTEM_MAP.get(majority) if majority else None

        row = {
            "event_id": fire["event_id"],
            "fire_name": fire["fire_name"],
            "year": fire["year"],
            "burn_area_ha": fire["burn_area_ha"],
            "latitude": fire["latitude"],
            "longitude": fire["longitude"],
            "ignition_date": fire["ignition_date"],
            "tile_used": tile_name,
            "majority_class": majority,
            "majority_label": WORLDCOVER_LABELS.get(majority, "unclassified") if majority else "unclassified",
            "ecosystem": ecosystem,
            "sample_pixels": n,
        }
        # Record fraction of each relevant class
        for cls in ECOSYSTEM_MAP:
            row[f"frac_class_{cls}"] = round(counts.get(cls, 0) / n, 4) if n > 0 else 0.0
        rows.append(row)

        if i % 25 == 0 or i == total:
            print(f"    Classified {i}/{total}")

    return pd.DataFrame(rows)


def select_fires(classified, n_per_eco=N_PER_ECOSYSTEM):
    selected = []
    selection_info = {}

    for eco in ["forest", "shrubland", "grassland"]:
        pool = classified[classified["ecosystem"] == eco].copy()
        pool = pool.sort_values(["year", "burn_area_ha"], ascending=[True, False])

        chosen = []
        used_coords = []

        # Try one per year first
        for year in [2018, 2019, 2020, 2021, 2022]:
            if len(chosen) >= n_per_eco:
                break
            year_pool = pool[pool["year"] == year]
            for _, cand in year_pool.iterrows():
                too_close = False
                for lat, lon in used_coords:
                    dlat = abs(cand["latitude"] - lat) * 111
                    dlon = abs(cand["longitude"] - lon) * 87
                    dist_km = (dlat ** 2 + dlon ** 2) ** 0.5
                    if dist_km < 50:
                        too_close = True
                        break
                if too_close:
                    continue
                chosen.append(cand)
                used_coords.append((cand["latitude"], cand["longitude"]))
                break

        # Fill remaining slots from the rest of the pool
        if len(chosen) < n_per_eco:
            for _, cand in pool.iterrows():
                if len(chosen) >= n_per_eco:
                    break
                if cand["event_id"] in [c["event_id"] for c in chosen]:
                    continue
                chosen.append(cand)

        selection_info[eco] = {
            "pool_size": int(len(pool)),
            "selected": len(chosen),
            "years": [int(c["year"]) for c in chosen],
        }
        selected.extend(chosen)

    return pd.DataFrame(selected), selection_info


def main():
    print("Stage 5, part 1: Classify fires by ecosystem")
    print("-" * 50)

    catalog = load_catalog()
    print(f"Loaded {len(catalog)} California fires")

    tiles = sorted(glob.glob(str(WORLDCOVER_DIR / "ESA_WorldCover_*.tif")))
    print(f"Found {len(tiles)} WorldCover tiles")
    for t in tiles:
        print(f"  {Path(t).name}")

    print("-" * 50)
    print("Classifying fires by WorldCover majority in 3 km buffer ...")
    classified = classify_all_fires(catalog, tiles)

    CLASSIFICATION_CSV.parent.mkdir(parents=True, exist_ok=True)
    classified.to_csv(CLASSIFICATION_CSV, index=False)
    print(f"Saved classification: {CLASSIFICATION_CSV.relative_to(PROJECT_ROOT)}")

    print("-" * 50)
    print("Classification summary:")
    eco_counts = classified["ecosystem"].value_counts(dropna=False)
    for eco, count in eco_counts.items():
        label = eco if pd.notna(eco) else "unclassified"
        print(f"  {label}: {count}")

    print("-" * 50)
    print(f"Selecting {N_PER_ECOSYSTEM} fires per ecosystem ...")
    selected, info = select_fires(classified)
    print(f"  Selected {len(selected)} fires total")
    for eco, stats in info.items():
        print(f"  {eco}: pool={stats['pool_size']}, selected={stats['selected']}, years={stats['years']}")

    SELECTED_CSV.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(SELECTED_CSV, index=False)
    print(f"Saved selection: {SELECTED_CSV.relative_to(PROJECT_ROOT)}")

    selection_json = {
        "stage": 5,
        "part": "classify_and_select",
        "timestamp": datetime.now().isoformat(),
        "catalog_size": int(len(catalog)),
        "ecosystem_counts": {str(k): int(v) for k, v in eco_counts.items()},
        "selection_info": info,
        "selected_fires": selected[["event_id", "fire_name", "year", "ecosystem", "burn_area_ha"]].to_dict(orient="records"),
        "total_selected": int(len(selected)),
        "worldcover_tiles": [Path(t).name for t in tiles],
        "sample_buffer_m": SAMPLE_BUFFER_M,
    }
    SELECTION_JSON.parent.mkdir(parents=True, exist_ok=True)
    SELECTION_JSON.write_text(json.dumps(selection_json, indent=2), encoding="utf-8")
    print(f"Saved selection report: {SELECTION_JSON.relative_to(PROJECT_ROOT)}")

    print("-" * 50)
    print("Selected fires:")
    print(selected[["event_id", "fire_name", "year", "ecosystem", "burn_area_ha"]].to_string(index=False))

    print("-" * 50)
    print("Stage 5 part 1 complete.")


if __name__ == "__main__":
    main()
