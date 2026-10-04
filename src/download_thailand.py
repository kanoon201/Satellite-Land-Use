"""Download cloud-free Sentinel-2 composites and ESA WorldCover labels for the Thai study areas.

For every area in config.AOIS this writes into data/thailand/:
    <area>/<area>_S2_L2A.tif      12-band median composite, surface reflectance x 10000
    <area>/<area>_S2_L1C.tif      13-band median composite, top-of-atmosphere x 10000
    <area>/<area>_worldcover.tif  ESA WorldCover class code per pixel
    <area>/<area>_clearobs.tif    number of cloud-free observations per pixel
    <area>/<area>_meta.json       scene counts, cloud statistics and grid definition

All rasters of an area share one 10 m pixel grid, so pixel (r, c) in the image
and in the label map is the same place on the ground.

Usage:  python src/download_thailand.py [--areas bangkok nan] [--levels L2A L1C]
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import ee
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import Affine

import config as c

TILE = 512      # pixels per request side; keeps each request far below the 48 MB limit
WORKERS = 6
RETRIES = 5


def aoi_origin(lon, lat):
    """Upper-left corner (x, y) of the area in THAI_CRS, snapped to the Sentinel-2 grid."""
    x, y = Transformer.from_crs("EPSG:4326", c.THAI_CRS, always_xy=True).transform(lon, lat)
    half = c.AOI_SIZE_PX * c.PIXEL_SIZE / 2
    # 60 m is a common multiple of the 10/20/60 m band grids
    return round((x - half) / 60) * 60, round((y + half) / 60) * 60


def aoi_region(x0, y0):
    side = c.AOI_SIZE_PX * c.PIXEL_SIZE
    return ee.Geometry.Rectangle([x0, y0 - side, x0 + side, y0], proj=c.THAI_CRS, geodesic=False)


def s2_collection(level, region):
    return (
        ee.ImageCollection(c.S2_COLLECTIONS[level])
        .filterBounds(region)
        .filterDate(c.S2_START, c.S2_END)
        .linkCollection(ee.ImageCollection(c.CLOUD_SCORE_COLLECTION), ["cs_cdf"])
    )


def s2_composite(level, region):
    bands = c.BANDS if level == "L2A" else c.BANDS_L1C

    def prepare(img):
        clear = img.select("cs_cdf").gte(c.CLEAR_THRESHOLD)
        # bilinear brings the 20 m and 60 m bands onto the 10 m grid without blocky pixels
        return img.select(bands).updateMask(clear).resample("bilinear")

    return s2_collection(level, region).map(prepare).median().unmask(0).toUint16(), bands


def clear_count(region):
    """Number of cloud-free observations per pixel over the period."""
    clear = s2_collection("L2A", region).map(lambda img: img.select("cs_cdf").gte(c.CLEAR_THRESHOLD))
    return clear.sum().unmask(0).toUint8().rename("clear_obs")


def fetch_tile(image, bands, x, y, width, height):
    request = {
        "expression": image,
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {
            "dimensions": {"width": width, "height": height},
            "affineTransform": {
                "scaleX": c.PIXEL_SIZE, "shearX": 0, "translateX": x,
                "shearY": 0, "scaleY": -c.PIXEL_SIZE, "translateY": y,
            },
            "crsCode": c.THAI_CRS,
        },
    }
    for attempt in range(RETRIES):
        try:
            tile = ee.data.computePixels(request)
            return np.stack([tile[b] for b in bands])
        except Exception:
            if attempt == RETRIES - 1:
                raise
            time.sleep(2 ** attempt * 3)


def fetch_raster(image, bands, x0, y0, dtype):
    n = c.AOI_SIZE_PX
    out = np.zeros((len(bands), n, n), dtype=dtype)
    jobs = [(r, col) for r in range(0, n, TILE) for col in range(0, n, TILE)]

    def run(job):
        r, col = job
        h, w = min(TILE, n - r), min(TILE, n - col)
        data = fetch_tile(image, bands, x0 + col * c.PIXEL_SIZE, y0 - r * c.PIXEL_SIZE, w, h)
        out[:, r:r + h, col:col + w] = data
        return job

    with ThreadPoolExecutor(WORKERS) as pool:
        for i, _ in enumerate(pool.map(run, jobs), 1):
            print(f"    tile {i}/{len(jobs)}", flush=True)
    return out


def save(path, array, bands, x0, y0):
    profile = dict(
        driver="GTiff", count=array.shape[0], height=array.shape[1], width=array.shape[2],
        dtype=array.dtype, crs=c.THAI_CRS, nodata=0, compress="deflate", tiled=True,
        transform=Affine(c.PIXEL_SIZE, 0, x0, 0, -c.PIXEL_SIZE, y0),
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array)
        dst.descriptions = tuple(bands)


def scene_stats(level, region):
    coll = s2_collection(level, region)
    return {
        "scenes": coll.size().getInfo(),
        "mean_cloudy_pixel_percentage": coll.aggregate_mean("CLOUDY_PIXEL_PERCENTAGE").getInfo(),
    }


def download_area(name, levels):
    lon, lat = c.AOIS[name]
    x0, y0 = aoi_origin(lon, lat)
    region = aoi_region(x0, y0)
    print(f"[{name}] centre ({lon}, {lat})  upper-left ({x0}, {y0}) {c.THAI_CRS}", flush=True)

    meta = {
        "area": name, "centre_lon_lat": [lon, lat], "crs": c.THAI_CRS,
        "upper_left_xy": [x0, y0], "pixel_size_m": c.PIXEL_SIZE, "size_px": c.AOI_SIZE_PX,
        "period": [c.S2_START, c.S2_END], "clear_threshold": c.CLEAR_THRESHOLD, "levels": {},
    }

    for level in levels:
        path = c.thai_path(name, f"S2_{level}.tif")
        meta["levels"][level] = scene_stats(level, region)
        if path.exists():
            print(f"  {path.name} exists, skipped", flush=True)
            continue
        print(f"  Sentinel-2 {level}: {meta['levels'][level]['scenes']} scenes", flush=True)
        image, bands = s2_composite(level, region)
        save(path, fetch_raster(image, bands, x0, y0, "uint16"), bands, x0, y0)

    worldcover = ee.ImageCollection(c.WORLDCOVER_COLLECTION).first().select("Map").unmask(0).toUint8()
    extras = {
        "worldcover": (worldcover, ["Map"]),
        "clearobs": (clear_count(region), ["clear_obs"]),
    }
    for key, (image, bands) in extras.items():
        path = c.thai_path(name, f"{key}.tif")
        if path.exists():
            print(f"  {path.name} exists, skipped", flush=True)
            continue
        print(f"  {key}", flush=True)
        save(path, fetch_raster(image, bands, x0, y0, "uint8"), bands, x0, y0)

    c.thai_path(name, "meta.json").write_text(json.dumps(meta, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--areas", nargs="+", default=list(c.AOIS), choices=list(c.AOIS))
    parser.add_argument("--levels", nargs="+", default=["L2A", "L1C"], choices=["L2A", "L1C"])
    args = parser.parse_args()

    ee.Initialize(project=c.EE_PROJECT)
    for name in args.areas:
        (c.THAILAND_DIR / name).mkdir(parents=True, exist_ok=True)
    for name in args.areas:
        download_area(name, args.levels)


if __name__ == "__main__":
    main()
