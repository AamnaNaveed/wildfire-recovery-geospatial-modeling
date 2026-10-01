"""
Stage 6a: Fetch auxiliary layers (terrain, climate, landcover) for all fires.

For each fire in data/interim/fires/{event_id}/, load the optical grid
(from post_nbr.tif) as the reference and fetch:
  - NASADEM elevation -> derive slope, aspect
  - ERA5-Land temperature and precipitation for the fire's year
  - ESA WorldCover land cover (clipped to fire bbox)
Align all to the fire's optical grid and save as GeoTIFFs in the fire folder.

Per PROJECT_PLAN.md section 4 and section 15, row 4.
"""

from pathlib import Path
import json
import glob
import gc
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
import rasterio
import pystac_client
import planetary_computer
import cdsapi


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIRES_DIR = PROJECT_ROOT / "data" / "interim" / "fires"
PERIMETERS_CSV = PROJECT_ROOT / "data" / "interim" / "fire_perimeters.csv"
CATALOG_PATH = PROJECT_ROOT / "data" / "interim" / "fire_catalog.csv"
WORLDCOVER_DIR = PROJECT_ROOT / "data" / "raw"
CLIMATE_CACHE_DIR = PROJECT_ROOT / "data" / "interim" / "climate_cache"
REPORTS_DIR = PROJECT_ROOT / "reports"

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
CLIMATE_BUFFER_DEG = 0.05


def log(msg):
    print(msg, flush=True)


def load_fires():
    """Return dict of event_id -> fire row + perim row."""
    sel = pd.read_csv(PROJECT_ROOT / "data" / "interim" / "selected_fires.csv")
    perim = pd.read_csv(PERIMETERS_CSV)
    cat = pd.read_csv(CATALOG_PATH)
    merged = sel.merge(perim, on="event_id", how="left")
    merged = merged.merge(
        cat[["event_id", "pre_image_id", "post_image_id", "ignition_date"]],
        on="event_id", how="left",
    )
    return merged.to_dict(orient="records")


def search_stac(collection, bbox):
    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    return list(catalog.search(collections=[collection], bbox=bbox).items())


def reproject_to_ref(da, ref, categorical=False):
    resampling = Resampling.nearest if categorical else Resampling.bilinear
    return da.rio.reproject_match(ref, resampling=resampling).load()


def fetch_terrain(event_id, fire, ref):
    """Fetch NASADEM elevation, derive slope and aspect, align to ref grid."""
    bbox = (
        float(fire["bbox_minx"]), float(fire["bbox_miny"]),
        float(fire["bbox_maxx"]), float(fire["bbox_maxy"]),
    )
    items = search_stac("nasadem", bbox)
    if not items:
        raise ValueError(f"No NASADEM for {event_id}")
    item = items[0]

    # Open elevation at overview
    for level in (2, 1, 3, 0):
        try:
            da = rioxarray.open_rasterio(
                item.assets["elevation"].href, masked=True, overview_level=level
            ).squeeze()
            da = da.load()
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f"Could not open NASADEM for {event_id}")

    dem = reproject_to_ref(da, ref, categorical=False)

    res_x = abs(ref.rio.resolution()[0])
    res_y = abs(ref.rio.resolution()[1])
    dy, dx = np.gradient(dem.values)
    slope_rad = np.arctan(np.sqrt((dx / res_x) ** 2 + (dy / res_y) ** 2))
    slope_deg = np.degrees(slope_rad).astype("float32")
    aspect_rad = np.arctan2(-dx, dy)
    aspect_deg = ((np.degrees(aspect_rad) + 360) % 360).astype("float32")

    slope = xr.DataArray(slope_deg, coords=dem.coords, dims=dem.dims).rio.write_crs(ref.rio.crs)
    aspect = xr.DataArray(aspect_deg, coords=dem.coords, dims=dem.dims).rio.write_crs(ref.rio.crs)
    return dem.astype("float32"), slope, aspect, item.id


def fetch_climate(event_id, fire, ref):
    """
    Fetch ERA5-Land for the fire's year, take annual mean temp and annual
    total precip as climate summaries. Cached per fire.
    """
    year = int(fire["year"])
    cache = CLIMATE_CACHE_DIR / f"{event_id}_era5_{year}.nc"
    CLIMATE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not cache.exists():
        log(f"      Requesting ERA5-Land for {year} ...")
        client = cdsapi.Client()
        # area = [N, W, S, E] in degrees
        area = [
            float(fire["bbox_maxy"]) + CLIMATE_BUFFER_DEG,
            float(fire["bbox_minx"]) - CLIMATE_BUFFER_DEG,
            float(fire["bbox_miny"]) - CLIMATE_BUFFER_DEG,
            float(fire["bbox_maxx"]) + CLIMATE_BUFFER_DEG,
        ]
        request = {
            "variable": ["2m_temperature", "total_precipitation"],
            "year": str(year),
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": ["00:00", "12:00"],
            "area": area,
            "format": "netcdf",
        }
        tmp = cache.parent / f"{event_id}_tmp.nc"
        client.retrieve("reanalysis-era5-land", request, str(tmp))
        if zipfile.is_zipfile(tmp):
            with zipfile.ZipFile(tmp) as z:
                nc_names = [n for n in z.namelist() if n.endswith(".nc")]
                with z.open(nc_names[0]) as src:
                    cache.write_bytes(src.read())
            tmp.unlink()
        else:
            tmp.rename(cache)
    else:
        log(f"      Using cached climate for {year}")

    ds = xr.open_dataset(cache)
    var_names = list(ds.data_vars)
    t2m_name = "t2m" if "t2m" in ds else var_names[0]
    tp_name = "tp" if "tp" in ds else var_names[1]

    # Dim names
    time_dim = "valid_time" if "valid_time" in ds.dims else "time"
    lat_dim = "latitude" if "latitude" in ds.coords else "lat"
    lon_dim = "longitude" if "longitude" in ds.coords else "lon"

    temp_mean = ds[t2m_name].mean(dim=time_dim) - 273.15
    precip_total = ds[tp_name].sum(dim=time_dim) * 1000.0

    if float(ds[lat_dim][0]) > float(ds[lat_dim][-1]):
        temp_mean = temp_mean.isel({lat_dim: slice(None, None, -1)})
        precip_total = precip_total.isel({lat_dim: slice(None, None, -1)})

    rename_map = {lat_dim: "y", lon_dim: "x"}
    temp_mean = temp_mean.rename(rename_map).rio.write_crs("EPSG:4326")
    precip_total = precip_total.rename(rename_map).rio.write_crs("EPSG:4326")

    temp = reproject_to_ref(temp_mean, ref, categorical=False).astype("float32")
    precip = reproject_to_ref(precip_total, ref, categorical=False).astype("float32")
    return temp, precip


def find_worldcover_for_fire(fire):
    """Find the WorldCover tile(s) that cover the fire's bbox."""
    tiles = sorted(glob.glob(str(WORLDCOVER_DIR / "ESA_WorldCover_*.tif")))
    covering = []
    lon_c = (float(fire["bbox_minx"]) + float(fire["bbox_maxx"])) / 2
    lat_c = (float(fire["bbox_miny"]) + float(fire["bbox_maxy"])) / 2
    for t in tiles:
        with rasterio.open(t) as src:
            b = src.bounds
            if b.left <= lon_c <= b.right and b.bottom <= lat_c <= b.top:
                covering.append(t)
    return covering


def fetch_landcover(event_id, fire, ref):
    """Clip WorldCover to fire bbox (in EPSG:4326), reproject to ref grid."""
    tiles = find_worldcover_for_fire(fire)
    if not tiles:
        raise ValueError(f"No WorldCover tile for {event_id}")

    # Use the first covering tile
    tile_path = tiles[0]
    da_full = rioxarray.open_rasterio(tile_path, masked=True)
    if "band" in da_full.dims:
        da_full = da_full.squeeze("band", drop=True)

    minx, miny = float(fire["bbox_minx"]), float(fire["bbox_miny"])
    maxx, maxy = float(fire["bbox_maxx"]), float(fire["bbox_maxy"])
    da_clipped = da_full.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy).load()

    lc = reproject_to_ref(da_clipped, ref, categorical=True).astype("float32")
    return lc, Path(tile_path).name


def save_geotiff(da, path):
    da = da.astype("float32")
    path.parent.mkdir(parents=True, exist_ok=True)
    if da.rio.crs is None:
        raise ValueError(f"{da.name} has no CRS")
    da.rio.to_raster(path)


def process_one_fire(fire):
    event_id = fire["event_id"]
    fire_name = fire["fire_name"]
    fire_dir = FIRES_DIR / event_id

    if not fire_dir.exists():
        raise ValueError(f"Fire folder missing: {fire_dir}")

    ref_path = fire_dir / "post_nbr.tif"
    if not ref_path.exists():
        raise ValueError(f"Missing post_nbr.tif for {event_id}")

    log(f"  [{event_id}] {fire_name}")
    ref = rioxarray.open_rasterio(ref_path, masked=True).squeeze()
    log(f"    ref shape: {ref.shape}, crs: {ref.rio.crs}")

    # Terrain
    log(f"    Fetching terrain (NASADEM) ...")
    dem, slope, aspect, dem_id = fetch_terrain(event_id, fire, ref)
    log(f"      dem shape: {dem.shape}")

    # Climate
    log(f"    Fetching climate (ERA5-Land) ...")
    temp, precip = fetch_climate(event_id, fire, ref)
    log(f"      temp={float(np.nanmean(temp.values)):.2f}C, precip={float(np.nanmean(precip.values)):.1f}mm")

    # Landcover
    log(f"    Fetching landcover (ESA WorldCover) ...")
    lc, lc_tile = fetch_landcover(event_id, fire, ref)
    log(f"      lc tile: {lc_tile}, shape: {lc.shape}")

    # Save
    save_geotiff(dem, fire_dir / "dem.tif")
    save_geotiff(slope, fire_dir / "slope.tif")
    save_geotiff(aspect, fire_dir / "aspect.tif")
    save_geotiff(temp, fire_dir / "temp.tif")
    save_geotiff(precip, fire_dir / "precip.tif")
    save_geotiff(lc, fire_dir / "landcover.tif")

    meta = {
        "event_id": event_id,
        "fire_name": fire_name,
        "ref_shape": list(ref.shape),
        "ref_crs": str(ref.rio.crs),
        "dem_item": dem_id,
        "landcover_tile": lc_tile,
        "stats": {
            "elevation_mean": float(np.nanmean(dem.values)),
            "slope_mean": float(np.nanmean(slope.values)),
            "temp_mean": float(np.nanmean(temp.values)),
            "precip_total_mean": float(np.nanmean(precip.values)),
            "landcover_mode": int(pd.Series(lc.values.flatten()).mode().iloc[0]) if lc.size else None,
        },
        "processed_at": datetime.now().isoformat(),
    }
    (fire_dir / "aux_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main():
    log("Stage 6a: Fetch auxiliary layers for all fires")
    log("-" * 50)

    fires = load_fires()
    log(f"Fires to process: {len(fires)}")

    metas = []
    errors = []
    for fire in fires:
        try:
            meta = process_one_fire(fire)
            metas.append(meta)
        except Exception as e:
            log(f"    ERROR on {fire['event_id']}: {type(e).__name__}: {e}")
            errors.append({"event_id": fire["event_id"], "error": str(e)})
        finally:
            gc.collect()

    report = {
        "stage": "6a",
        "timestamp": datetime.now().isoformat(),
        "n_fires": len(fires),
        "n_succeeded": len(metas),
        "n_failed": len(errors),
        "errors": errors,
        "fires": metas,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "aux_layers_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"Saved report: {report_path.relative_to(PROJECT_ROOT)}")

    log("-" * 50)
    log(f"Stage 6a complete: {len(metas)} succeeded, {len(errors)} failed")


if __name__ == "__main__":
    main()