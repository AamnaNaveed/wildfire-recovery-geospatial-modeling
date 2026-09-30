"""
Stage 5, part 3: Load MTBS fire perimeters and match to selected fires.

Loads the nationwide bsp_perims_DD shapefile, matches each selected fire
by event_id, deduplicates polygons (MTBS ships multiple near-identical
perimeters per fire), and computes the perimeter bounding box (with a
small buffer) for use as the processing window in the Sentinel-2 pipeline.

Outputs:
  data/interim/fire_perimeters.csv  - event_id + perimeter bbox + area
  reports/perimeter_loading.json    - diagnostic report

Per PROJECT_PLAN.md section 6 and section 15, row 5.
"""

from pathlib import Path
import json
from datetime import datetime

import geopandas as gpd
import pandas as pd


# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PERIM_SHP = PROJECT_ROOT / "data" / "interim" / "perimeters_national" / "bsp_perims_DD.shp"
SELECTED_CSV = PROJECT_ROOT / "data" / "interim" / "selected_fires.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "interim" / "fire_perimeters.csv"
REPORT_JSON = PROJECT_ROOT / "reports" / "perimeter_loading.json"


BBOX_BUFFER_DEG = 0.02


def find_event_id_column(gdf):
    candidates = ["event_id", "EVENT_ID", "Event_ID", "EVENTID", "eventid"]
    for col in candidates:
        if col in gdf.columns:
            return col
    for col in gdf.columns:
        if "event" in col.lower() and "id" in col.lower():
            return col
    raise ValueError(f"No event_id column found. Columns: {list(gdf.columns)}")


def main():
    print("Stage 5, part 3: Load MTBS perimeters")
    print("-" * 50)

    if not PERIM_SHP.exists():
        raise FileNotFoundError(f"Perimeter shapefile not found: {PERIM_SHP}")

    print(f"Loading {PERIM_SHP.name} ...")
    print("  (This is a 1 GB shapefile; may take 30-90 seconds)")
    perims = gpd.read_file(PERIM_SHP)
    print(f"  Loaded {len(perims):,} features")
    print(f"  CRS: {perims.crs}")

    id_col = find_event_id_column(perims)
    print(f"  Using event_id column: '{id_col}'")

    print("Loading selected fires ...")
    selected = pd.read_csv(SELECTED_CSV)
    print(f"  Selected fires: {len(selected)}")

    print("Matching perimeters to selected fires ...")
    perims = perims.copy()
    perims["_event_id_norm"] = perims[id_col].astype(str).str.strip()
    selected_ids = set(selected["event_id"].astype(str).str.strip())

    matched = perims[perims["_event_id_norm"].isin(selected_ids)].copy()
    print(f"  Matched {len(matched)} polygons across {matched['_event_id_norm'].nunique()} unique fires")

    # Deduplicate: keep one polygon per event_id
    print("Deduplicating (MTBS ships multiple near-identical perimeters per fire) ...")
    matched = matched.sort_values(["event_id", "map_id"]).drop_duplicates(subset="event_id", keep="first")
    print(f"  Kept {len(matched)} polygons (one per fire)")

    # Reproject to EPSG:4326
    matched = matched.to_crs("EPSG:4326")

    # Fix invalid geometries
    print("Fixing invalid geometries ...")
    matched["geometry"] = matched["geometry"].make_valid()

    # Compute bbox per fire
    print("Computing bboxes ...")
    rows = []
    for _, row in matched.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        minx, miny, maxx, maxy = geom.bounds

        rows.append({
            "event_id": row["_event_id_norm"],
            "map_id": row.get("map_id"),
            "perim_area_ha_mtbs": float(row.get("burnbndac", 0)),
            "perim_minx": minx,
            "perim_miny": miny,
            "perim_maxx": maxx,
            "perim_maxy": maxy,
            "bbox_minx": minx - BBOX_BUFFER_DEG,
            "bbox_miny": miny - BBOX_BUFFER_DEG,
            "bbox_maxx": maxx + BBOX_BUFFER_DEG,
            "bbox_maxy": maxy + BBOX_BUFFER_DEG,
            "bbox_width_deg": round(maxx - minx, 4),
            "bbox_height_deg": round(maxy - miny, 4),
        })

    result = pd.DataFrame(rows).sort_values("event_id").reset_index(drop=True)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV.relative_to(PROJECT_ROOT)}")

    unmatched = selected[~selected["event_id"].astype(str).str.strip().isin(set(result["event_id"]))]
    report = {
        "stage": "5c",
        "description": "Load MTBS perimeters and match to selected fires (deduplicated)",
        "timestamp": datetime.now().isoformat(),
        "n_selected": int(len(selected)),
        "n_unique_fires_matched": int(len(result)),
        "n_unmatched": int(len(unmatched)),
        "unmatched_ids": unmatched["event_id"].tolist() if len(unmatched) else [],
        "bbox_buffer_deg": BBOX_BUFFER_DEG,
        "source_shapefile": str(PERIM_SHP.relative_to(PROJECT_ROOT)),
        "fires": result.to_dict(orient="records"),
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Saved report: {REPORT_JSON.relative_to(PROJECT_ROOT)}")

    print("-" * 50)
    print("Per-fire perimeter summary:")
    cols = ["event_id", "perim_area_ha_mtbs", "bbox_width_deg", "bbox_height_deg"]
    print(result[cols].to_string(index=False))

    print("-" * 50)
    print(f"Unique fires matched: {len(result)}/{len(selected)}")
    if len(unmatched):
        print(f"Unmatched: {unmatched['event_id'].tolist()}")
    print("Stage 5 part 3 complete.")


if __name__ == "__main__":
    main()
