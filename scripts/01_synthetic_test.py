"""
Stage 1 synthetic raster test.

Creates a small synthetic raster with two bands (NIR, Red), computes NDVI,
writes the result to GeoTIFF, and produces a JSON QA report.

No real data is downloaded. Everything here is synthetic.
"""

from pathlib import Path
import json
from datetime import datetime

import numpy as np
import rasterio
from rasterio.transform import from_origin
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yml"
REPORT_DIR = PROJECT_ROOT / "reports"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
DATA_DIR = PROJECT_ROOT / "data" / "interim"


def make_synthetic_bands(size: int = 100, seed: int = 42):
    """Create two fake reflectance bands: NIR and Red."""
    rng = np.random.default_rng(seed)

    # Base pattern: smooth gradient so the raster looks structured
    y, x = np.mgrid[0:size, 0:size]
    gradient = (x + y) / (2 * size)

    # Add noise
    noise = rng.normal(0, 0.05, size=(size, size))

    # Red band: low reflectance
    red = np.clip(0.10 + 0.15 * gradient + noise, 0.0, 1.0)

    # NIR band: high reflectance, roughly anti-correlated with red
    nir = np.clip(0.40 + 0.30 * (1 - gradient) + noise, 0.0, 1.0)

    return red.astype("float32"), nir.astype("float32")


def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """NDVI = (NIR - Red) / (NIR + Red)."""
    denominator = nir + red
    # Avoid divide-by-zero
    denominator = np.where(denominator == 0, np.nan, denominator)
    return ((nir - red) / denominator).astype("float32")


def write_geotiff(path: Path, red: np.ndarray, nir: np.ndarray, ndvi: np.ndarray):
    """Write a 3-band GeoTIFF: Red, NIR, NDVI."""
    height, width = red.shape
    transform = from_origin(west=0, north=height, xsize=1, ysize=1)

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 3,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "compress": "lzw",
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(red, 1)
        dst.write(nir, 2)
        dst.write(ndvi, 3)
        dst.set_band_description(1, "Red")
        dst.set_band_description(2, "NIR")
        dst.set_band_description(3, "NDVI")


def read_qa(path: Path) -> dict:
    """Read back the GeoTIFF and gather QA statistics."""
    with rasterio.open(path) as src:
        red = src.read(1)
        nir = src.read(2)
        ndvi = src.read(3)

        return {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "driver": src.driver,
            "crs": str(src.crs),
            "shape": [src.height, src.width],
            "count": src.count,
            "dtype": src.dtypes[0],
            "transform": list(src.transform)[:6],
            "band_descriptions": src.descriptions,
            "stats": {
                "red": {
                    "min": float(np.nanmin(red)),
                    "max": float(np.nanmax(red)),
                    "mean": float(np.nanmean(red)),
                },
                "nir": {
                    "min": float(np.nanmin(nir)),
                    "max": float(np.nanmax(nir)),
                    "mean": float(np.nanmean(nir)),
                },
                "ndvi": {
                    "min": float(np.nanmin(ndvi)),
                    "max": float(np.nanmax(ndvi)),
                    "mean": float(np.nanmean(ndvi)),
                    "valid_pixels": int(np.sum(~np.isnan(ndvi))),
                    "total_pixels": int(ndvi.size),
                },
            },
        }


def main():
    print("Stage 1: synthetic raster test")
    print("-" * 40)

    # Load config
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        print(f"Loaded config: {CONFIG_PATH.name}")
    else:
        config = {}
        print(f"Config not found at {CONFIG_PATH}, continuing with defaults.")

    # Build synthetic bands
    red, nir = make_synthetic_bands(size=100, seed=42)
    print(f"Created synthetic bands: shape {red.shape}")

    # Compute NDVI
    ndvi = compute_ndvi(red, nir)
    print(f"Computed NDVI: mean={np.nanmean(ndvi):.4f}")

    # Write GeoTIFF
    raster_path = DATA_DIR / "synthetic_test.tif"
    write_geotiff(raster_path, red, nir, ndvi)
    print(f"Wrote raster: {raster_path.relative_to(PROJECT_ROOT)}")

    # Read back and QA
    qa = read_qa(raster_path)
    qa["timestamp"] = datetime.now().isoformat()
    qa["source"] = "synthetic"
    qa["config_loaded"] = bool(config)

    # Write QA report
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "synthetic_test_qa.json"
    report_path.write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(f"Wrote QA report: {report_path.relative_to(PROJECT_ROOT)}")

    # Summary
    print("-" * 40)
    print("Summary:")
    print(f"  NDVI min:  {qa['stats']['ndvi']['min']:.4f}")
    print(f"  NDVI max:  {qa['stats']['ndvi']['max']:.4f}")
    print(f"  NDVI mean: {qa['stats']['ndvi']['mean']:.4f}")
    print(f"  Valid pixels: {qa['stats']['ndvi']['valid_pixels']} / {qa['stats']['ndvi']['total_pixels']}")
    print("Stage 1 synthetic test complete.")


if __name__ == "__main__":
    main()