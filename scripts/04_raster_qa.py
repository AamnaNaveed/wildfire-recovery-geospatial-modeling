"""
Stage 4: Raster QA.

Load terrain (NASADEM), climate (ERA5-Land), and land cover (ESA WorldCover,
local file) for the Planada fire. Align all layers to a common grid centered
on the fire. Write a QA report documenting ranges, valid-pixel counts, and
alignment checks. Produce a multi-panel preview.

Per PROJECT_PLAN.md section 15, row 4.
"""

from pathlib import Path
import json
from datetime import datetime
import zipfile

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
import pystac_client
import planetary_computer
import cdsapi
from pyproj import Transformer


# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "interim" / "fire_catalog.csv"
REFERENCE_TIF = PROJECT_ROOT / "data" / "interim" / "fire_planada" / "post_nbr.tif"
OUTPUT_DIR = PROJECT_ROOT / "data" / "interim" / "fire_planada"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
QA_PATH = PROJECT_ROOT / "reports" / "raster_qa.json"
CLIMATE_EXTRACTED = OUTPUT_DIR / "era5_land_extracted.nc"
WORLDCOVER_TIF = PROJECT_ROOT / "data" / "raw" / "ESA_WorldCover_10m_2021_v200_N36W123_Map.tif"


# --- Fire of interest ---
FIRE_EVENT_ID = "CA3735012030020180615"

# --- Half-size of the working grid around the fire, in metres ---
HALF_WINDOW_M = 10000


def load_fire():
    df = pd.read_csv(CATALOG_PATH)
    row = df[df["event_id"] == FIRE_EVENT_ID]
    if row.empty:
        raise ValueError(f"Fire {FIRE_EVENT_ID} not found in catalog")
    return row.iloc[0].to_dict()


def load_reference():
    da = rioxarray.open_rasterio(REFERENCE_TIF, masked=True).squeeze()
    return da


def fire_centroid_in_ref_crs(fire, ref):
    transformer = Transformer.from_crs("EPSG:4326", ref.rio.crs, always_xy=True)
    x, y = transformer.transform(float(fire["longitude"]), float(fire["latitude"]))
    return x, y


def make_fire_reference(fire, ref_full, half_window_m=HALF_WINDOW_M):
    cx, cy = fire_centroid_in_ref_crs(fire, ref_full)
    minx, maxx = cx - half_window_m, cx + half_window_m
    miny, maxy = cy - half_window_m, cy + half_window_m
    print(f"    Fire centroid in {ref_full.rio.crs}: ({cx:.0f}, {cy:.0f})")
    print(f"    Working window: x [{minx:.0f}, {maxx:.0f}], y [{miny:.0f}, {maxy:.0f}]")
    clipped = ref_full.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy).load()
    print(f"    Working grid shape: {clipped.shape}")
    return clipped


def working_bounds_lonlat(ref):
    """Return (west, south, east, north) of the working grid, in EPSG:4326 degrees."""
    left, bottom, right, top = ref.rio.bounds()
    transformer = Transformer.from_crs(ref.rio.crs, "EPSG:4326", always_xy=True)
    west, south = transformer.transform(left, bottom)
    east, north = transformer.transform(right, top)
    return west, south, east, north


def reproject_to_reference(da, ref, categorical=False):
    resampling = Resampling.nearest if categorical else Resampling.bilinear
    aligned = da.rio.reproject_match(ref, resampling=resampling)
    return aligned.load()


def open_cog_small(href, overview_levels=(2, 1, 3, 0)):
    last_err = None
    for level in overview_levels:
        try:
            da = rioxarray.open_rasterio(
                href, masked=True, overview_level=level
            ).squeeze()
            return da.load()
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Could not open {href}: {last_err}")


def search_stac(collection, bbox, datetime_range=None):
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    kwargs = {"collections": [collection], "bbox": bbox}
    if datetime_range:
        kwargs["datetime"] = datetime_range
    return list(catalog.search(**kwargs).items())


def fetch_nasadem(bbox, ref):
    items = search_stac("nasadem", bbox)
    if not items:
        raise ValueError("No NASADEM items found")
    item = items[0]
    print(f"    NASADEM item: {item.id}")

    da = open_cog_small(item.assets["elevation"].href)
    print(f"    NASADEM loaded shape: {da.shape}")

    dem = reproject_to_reference(da, ref, categorical=False)

    res_x = abs(ref.rio.resolution()[0])
    res_y = abs(ref.rio.resolution()[1])

    dy, dx = np.gradient(dem.values)
    slope_rad = np.arctan(np.sqrt((dx / res_x) ** 2 + (dy / res_y) ** 2))
    slope_deg = np.degrees(slope_rad)

    aspect_rad = np.arctan2(-dx, dy)
    aspect_deg = (np.degrees(aspect_rad) + 360) % 360

    slope = xr.DataArray(slope_deg, coords=dem.coords, dims=dem.dims).rio.write_crs(dem.rio.crs)
    aspect = xr.DataArray(aspect_deg, coords=dem.coords, dims=dem.dims).rio.write_crs(dem.rio.crs)

    return dem, slope, aspect, item.id


def fetch_climate(working_bounds_lonlat_bbox, extracted_path=CLIMATE_EXTRACTED):
    """
    Fetch ERA5-Land for the FULL working grid, not just the fire bbox.
    Caches to extracted_path. The bbox arg is (W, S, E, N) in degrees.
    """
    if extracted_path.exists():
        print(f"    Using extracted NetCDF: {extracted_path.name}")
        return xr.open_dataset(extracted_path)

    print("    Requesting ERA5-Land from CDS (this may take 1-3 minutes) ...")
    client = cdsapi.Client()

    west, south, east, north = working_bounds_lonlat_bbox
    # Add a small buffer to be sure we cover the full working grid
    buf = 0.05
    area = [north + buf, west - buf, south - buf, east + buf]

    request = {
        "variable": ["2m_temperature", "total_precipitation"],
        "year": "2018",
        "month": "06",
        "day": [f"{d:02d}" for d in range(1, 31)],
        "time": ["00:00", "12:00"],
        "area": area,
        "format": "netcdf",
    }

    tmp_path = extracted_path.parent / "era5_land_download.nc"
    extracted_path.parent.mkdir(parents=True, exist_ok=True)
    client.retrieve("reanalysis-era5-land", request, str(tmp_path))

    if zipfile.is_zipfile(tmp_path):
        print("    CDS returned a ZIP; extracting the NetCDF ...")
        with zipfile.ZipFile(tmp_path) as z:
            nc_names = [n for n in z.namelist() if n.endswith(".nc")]
            if not nc_names:
                raise ValueError("No .nc inside ZIP")
            with z.open(nc_names[0]) as src:
                data = src.read()
            extracted_path.write_bytes(data)
        print(f"    Wrote {extracted_path.name} ({extracted_path.stat().st_size} bytes)")
    else:
        tmp_path.rename(extracted_path)

    return xr.open_dataset(extracted_path)


def _find_dim(ds, candidates):
    for name in candidates:
        if name in ds.dims:
            return name
    raise ValueError(f"None of {candidates} found in dims {list(ds.dims)}")


def _find_coord(ds, candidates):
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    raise ValueError(f"None of {candidates} found in coords {list(ds.coords)}")


def process_climate(ds, ref):
    var_names = list(ds.data_vars)
    print(f"    Climate variables in file: {var_names}")

    t2m_name = "t2m" if "t2m" in ds else var_names[0]
    tp_name = "tp" if "tp" in ds else var_names[1]

    time_dim = _find_dim(ds, ["valid_time", "time"])
    lat_dim = _find_coord(ds, ["latitude", "lat"])
    lon_dim = _find_coord(ds, ["longitude", "lon"])

    temp_mean = ds[t2m_name].mean(dim=time_dim) - 273.15
    precip_total = ds[tp_name].sum(dim=time_dim) * 1000.0

    lat_values = ds[lat_dim].values
    if lat_values[0] > lat_values[-1]:
        temp_mean = temp_mean.isel({lat_dim: slice(None, None, -1)})
        precip_total = precip_total.isel({lat_dim: slice(None, None, -1)})

    rename_map = {lat_dim: "y", lon_dim: "x"}
    temp_mean = temp_mean.rename(rename_map)
    precip_total = precip_total.rename(rename_map)

    temp_mean = temp_mean.rio.write_crs("EPSG:4326")
    precip_total = precip_total.rio.write_crs("EPSG:4326")

    temp_aligned = reproject_to_reference(temp_mean, ref, categorical=False)
    precip_aligned = reproject_to_reference(precip_total, ref, categorical=False)

    return temp_aligned, precip_aligned


def fetch_landcover(working_bounds_lonlat_bbox, ref, tif_path=WORLDCOVER_TIF):
    """Clip WorldCover to the FULL working grid extent, not just the fire bbox."""
    if not tif_path.exists():
        raise FileNotFoundError(f"WorldCover tile not found: {tif_path}")
    print(f"    Local WorldCover tile: {tif_path.name}")

    west, south, east, north = working_bounds_lonlat_bbox
    da_full = rioxarray.open_rasterio(tif_path, masked=True)
    if "band" in da_full.dims:
        da_full = da_full.squeeze("band", drop=True)

    da_clipped = da_full.rio.clip_box(
        minx=west, miny=south, maxx=east, maxy=north
    )
    da_clipped = da_clipped.load()
    print(f"    Clipped WorldCover shape: {da_clipped.shape}")

    lc = reproject_to_reference(da_clipped, ref, categorical=True)
    return lc, tif_path.name


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


def make_figure(dem, slope, aspect, temp, precip, lc, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    im0 = axes[0, 0].imshow(dem.values, cmap="terrain")
    axes[0, 0].set_title("Elevation (m)")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(slope.values, cmap="magma")
    axes[0, 1].set_title("Slope (deg)")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[0, 2].imshow(aspect.values, cmap="hsv", vmin=0, vmax=360)
    axes[0, 2].set_title("Aspect (deg)")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    im3 = axes[1, 0].imshow(temp.values, cmap="RdBu_r")
    axes[1, 0].set_title("Mean 2m Temp (C) - June 2018")
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

    im4 = axes[1, 1].imshow(precip.values, cmap="Blues")
    axes[1, 1].set_title("Total Precip (mm) - June 2018")
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)

    im5 = axes[1, 2].imshow(lc.values, cmap="tab20")
    axes[1, 2].set_title("Land Cover (ESA WorldCover class)")
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046)

    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main():
    print("Stage 4: Raster QA")
    print("-" * 50)

    fire = load_fire()
    print(f"Fire: {fire['fire_name']} ({fire['event_id']})")

    ref_full = load_reference()
    print(f"Full reference shape: {ref_full.shape}, bounds: {ref_full.rio.bounds()}")

    print("Defining fire-centered working grid ...")
    ref = make_fire_reference(fire, ref_full)

    # Get the working grid extent in lon/lat so climate and land cover
    # are fetched for the FULL grid, not just the fire bbox
    wb = working_bounds_lonlat(ref)
    print(f"Working grid bounds (lon/lat): W={wb[0]:.4f}, S={wb[1]:.4f}, E={wb[2]:.4f}, N={wb[3]:.4f}")
    working_bbox_ll = (wb[0], wb[1], wb[2], wb[3])

    print("-" * 50)
    print("Fetching terrain (NASADEM) ...")
    dem, slope, aspect, dem_id = fetch_nasadem(working_bbox_ll, ref)
    print(f"    DEM aligned shape: {dem.shape}")

    print("-" * 50)
    print("Fetching climate (ERA5-Land) ...")
    climate_ds = fetch_climate(working_bbox_ll)
    temp, precip = process_climate(climate_ds, ref)
    print(f"    Temperature aligned shape: {temp.shape}")
    print(f"    Precipitation aligned shape: {precip.shape}")

    print("-" * 50)
    print("Fetching land cover (ESA WorldCover, local) ...")
    lc, lc_id = fetch_landcover(working_bbox_ll, ref)
    print(f"    Land cover aligned shape: {lc.shape}")

    print("-" * 50)
    print("Building QA report ...")

    shapes = {
        "reference": tuple(ref.shape),
        "dem": tuple(dem.shape),
        "slope": tuple(slope.shape),
        "aspect": tuple(aspect.shape),
        "temp": tuple(temp.shape),
        "precip": tuple(precip.shape),
        "landcover": tuple(lc.shape),
    }
    all_aligned = len(set(shapes.values())) == 1

    qa = {
        "stage": 4,
        "description": "Raster QA for one fire (Planada) - terrain, climate, land cover",
        "timestamp": datetime.now().isoformat(),
        "fire": {
            "event_id": fire["event_id"],
            "fire_name": fire["fire_name"],
            "bbox_lonlat": list(working_bbox_ll),
        },
        "reference_grid": {
            "source": str(REFERENCE_TIF.relative_to(PROJECT_ROOT)),
            "note": "Clipped to a 20x20 km window centered on the fire",
            "shape": tuple(ref.shape),
            "crs": str(ref.rio.crs),
            "resolution": ref.rio.resolution(),
            "bounds": tuple(ref.rio.bounds()),
            "bounds_lonlat": list(working_bbox_ll),
        },
        "alignment": {
            "all_same_shape": all_aligned,
            "shapes": shapes,
        },
        "sources": {
            "terrain": {"collection": "nasadem", "item": dem_id},
            "climate": {"dataset": "reanalysis-era5-land", "months": "2018-06"},
            "landcover": {"source": "local_tile", "file": lc_id},
        },
        "layers": {
            "elevation": summarize(dem, "elevation_m"),
            "slope": summarize(slope, "slope_deg"),
            "aspect": summarize(aspect, "aspect_deg"),
            "temperature": summarize(temp, "temp_celsius"),
            "precipitation": summarize(precip, "precip_mm_total"),
            "landcover": summarize(lc, "landcover_class"),
        },
    }

    QA_PATH.parent.mkdir(parents=True, exist_ok=True)
    QA_PATH.write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(f"Saved QA report: {QA_PATH.relative_to(PROJECT_ROOT)}")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURE_DIR / "raster_qa_preview.png"
    make_figure(dem, slope, aspect, temp, precip, lc, fig_path)
    print(f"Saved figure: {fig_path.relative_to(PROJECT_ROOT)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dem.rio.to_raster(OUTPUT_DIR / "dem.tif")
    slope.rio.to_raster(OUTPUT_DIR / "slope.tif")
    aspect.rio.to_raster(OUTPUT_DIR / "aspect.tif")
    temp.rio.to_raster(OUTPUT_DIR / "temp_june2018.tif")
    precip.rio.to_raster(OUTPUT_DIR / "precip_june2018.tif")
    lc.rio.to_raster(OUTPUT_DIR / "landcover.tif")
    print("Saved aligned layers to data/interim/fire_planada/")

    print("-" * 50)
    print("QA Summary:")
    for name, stats in qa["layers"].items():
        print(f"  {name}: valid={stats['valid_fraction']:.3f}, "
              f"min={stats.get('min', 0):.2f}, max={stats.get('max', 0):.2f}")
    print(f"  All layers aligned: {all_aligned}")
    print("Stage 4 complete.")


if __name__ == "__main__":
    main()
