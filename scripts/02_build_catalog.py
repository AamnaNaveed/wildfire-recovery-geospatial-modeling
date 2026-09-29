"""
Stage 2: Build the fire event catalog.

Reads the MTBS Fire Occurrence Data (shapefile), filters to California
wildfires 2018-2022 with burn areas between 1,000 and 250,000 hectares,
and writes the filtered catalog to CSV.

Per PROJECT_PLAN.md section 6 and section 15, row 2.
"""

from pathlib import Path
import json
from datetime import datetime

import geopandas as gpd
import pandas as pd
import yaml


# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yml"
SOURCE_SHP = PROJECT_ROOT / "data" / "interim" / "fod_points" / "bsp_FODpoints_DD.shp"
OUTPUT_CSV = PROJECT_ROOT / "data" / "interim" / "fire_catalog.csv"
MANIFEST_PATH = PROJECT_ROOT / "reports" / "fire_catalog_manifest.json"


# --- Filter criteria (from PROJECT_PLAN.md section 6) ---
FILTERS = {
    "map_prog": "MTBS",
    "incid_type": "Wildfire",
    "year_min": 2018,
    "year_max": 2022,
    "area_ha_min": 1000,
    "area_ha_max": 250000,
    # California bounding box (approximate)
    "bbox_lon_min": -124.5,
    "bbox_lon_max": -114.0,
    "bbox_lat_min": 32.5,
    "bbox_lat_max": 42.0,
}

ACRES_TO_HA = 0.404686


def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def apply_filters(gdf):
    """Apply all filter criteria. Returns (filtered_gdf, filter_report)."""
    report = {"initial_rows": int(len(gdf))}

    # 1. Filter map_prog
    gdf = gdf[gdf["map_prog"] == FILTERS["map_prog"]].copy()
    report["after_map_prog"] = int(len(gdf))

    # 2. Filter incid_type
    gdf = gdf[gdf["incid_type"] == FILTERS["incid_type"]].copy()
    report["after_incid_type"] = int(len(gdf))

    # 3. Parse ignition year
    gdf["ignition_date"] = pd.to_datetime(gdf["ig_date"], errors="coerce")
    gdf["year"] = gdf["ignition_date"].dt.year

    # 4. Filter years
    gdf = gdf[
        (gdf["year"] >= FILTERS["year_min"]) & (gdf["year"] <= FILTERS["year_max"])
    ].copy()
    report["after_year_filter"] = int(len(gdf))

    # 5. Convert lat/lon/area to numeric (shapefile loads them as strings)
    gdf["burnbndlat"] = pd.to_numeric(gdf["burnbndlat"], errors="coerce")
    gdf["burnbndlon"] = pd.to_numeric(gdf["burnbndlon"], errors="coerce")
    gdf["burnbndac"] = pd.to_numeric(gdf["burnbndac"], errors="coerce")
    gdf = gdf.dropna(subset=["burnbndlat", "burnbndlon", "burnbndac"]).copy()
    report["after_numeric_coerce"] = int(len(gdf))

    # 6. Filter geography (California bounding box)
    gdf = gdf[
        (gdf["burnbndlon"] >= FILTERS["bbox_lon_min"])
        & (gdf["burnbndlon"] <= FILTERS["bbox_lon_max"])
        & (gdf["burnbndlat"] >= FILTERS["bbox_lat_min"])
        & (gdf["burnbndlat"] <= FILTERS["bbox_lat_max"])
    ].copy()
    report["after_bbox_filter"] = int(len(gdf))

    # 7. Convert acres to hectares and filter by area
    gdf["area_ha"] = gdf["burnbndac"] * ACRES_TO_HA
    gdf = gdf[
        (gdf["area_ha"] >= FILTERS["area_ha_min"])
        & (gdf["area_ha"] <= FILTERS["area_ha_max"])
    ].copy()
    report["after_area_filter"] = int(len(gdf))

    return gdf, report


def build_catalog(gdf):
    """Select and rename the columns that belong in the catalog."""
    catalog = pd.DataFrame({
        "event_id": gdf["event_id"],
        "fire_name": gdf["incid_name"],
        "map_program": gdf["map_prog"],
        "fire_type": gdf["incid_type"],
        "ignition_date": gdf["ignition_date"].dt.strftime("%Y-%m-%d"),
        "year": gdf["year"].astype("Int64"),
        "burn_area_acres": gdf["burnbndac"].round(2),
        "burn_area_ha": gdf["area_ha"].round(2),
        "latitude": gdf["burnbndlat"].round(6),
        "longitude": gdf["burnbndlon"].round(6),
        "pre_image_id": gdf["pre_id"],
        "post_image_id": gdf["post_id"],
        "perimeter_image_id": gdf["perim_id"],
    })
    return catalog.sort_values("ignition_date").reset_index(drop=True)


def write_manifest(filter_report, catalog, config):
    manifest = {
        "stage": 2,
        "description": "Fire event catalog built from MTBS Fire Occurrence Data",
        "timestamp": datetime.now().isoformat(),
        "source_file": str(SOURCE_SHP.relative_to(PROJECT_ROOT)),
        "source_program": "MTBS (Monitoring Trends in Burn Severity)",
        "filters_applied": FILTERS,
        "filter_report": filter_report,
        "final_catalog_rows": int(len(catalog)),
        "years_in_catalog": sorted(catalog["year"].dropna().unique().tolist()),
        "area_ha_range": [
            float(catalog["burn_area_ha"].min()),
            float(catalog["burn_area_ha"].max()),
        ],
        "config_loaded": bool(config),
        "output_file": str(OUTPUT_CSV.relative_to(PROJECT_ROOT)),
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    print("Stage 2: Build fire event catalog")
    print("-" * 50)

    # Load config
    config = load_config()
    print(f"Config loaded: {bool(config)}")

    # Read shapefile
    print(f"Reading {SOURCE_SHP.relative_to(PROJECT_ROOT)} ...")
    gdf = gpd.read_file(SOURCE_SHP)
    print(f"  Loaded {len(gdf):,} rows")

    # Apply filters
    filtered, report = apply_filters(gdf)
    print("-" * 50)
    print("Filter report:")
    print(f"  Initial:            {report['initial_rows']:>6,}")
    print(f"  After map_prog:     {report['after_map_prog']:>6,}")
    print(f"  After incid_type:   {report['after_incid_type']:>6,}")
    print(f"  After year:         {report['after_year_filter']:>6,}")
    print(f"  After numeric:      {report['after_numeric_coerce']:>6,}")
    print(f"  After bbox (CA):    {report['after_bbox_filter']:>6,}")
    print(f"  After area:         {report['after_area_filter']:>6,}")

    if len(filtered) == 0:
        print("ERROR: No fires passed all filters. Check filter criteria.")
        return

    # Build catalog
    catalog = build_catalog(filtered)

    # Write CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(OUTPUT_CSV, index=False)
    print("-" * 50)
    print(f"Wrote catalog: {OUTPUT_CSV.relative_to(PROJECT_ROOT)}")
    print(f"  Rows: {len(catalog)}")
    print(f"  Years: {sorted(catalog['year'].dropna().unique().tolist())}")
    print(f"  Area range (ha): {catalog['burn_area_ha'].min():,.0f} - {catalog['burn_area_ha'].max():,.0f}")

    # Write manifest
    manifest = write_manifest(report, catalog, config)
    print(f"Wrote manifest: {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")

    # Preview
    print("-" * 50)
    print("Preview (first 5 rows):")
    print(catalog.head(5).to_string())

    print("-" * 50)
    print("Stage 2 complete.")


if __name__ == "__main__":
    main()
