from pathlib import Path
from tqdm import tqdm
import rasterio

path = Path("/NAS/data/dop/BB/tif")


def extract_from_file_name(file_name: str) -> tuple[tuple[float, float]]:
    split_file_name = file_name.split("_")
    coords = list(map(float, split_file_name[2:4]))

    return coords


correct_crs = rasterio.CRS.from_epsg(25833)
for file in tqdm(list(path.rglob("dop20_*.tif"))):
    with rasterio.open(file, "r+") as src:
        profile = src.profile
        crs = profile["crs"]

        if crs == correct_crs:
            continue

        easting, northing = extract_from_file_name(file.name)
        left = easting * 1000
        bottom = northing * 1000
        right = (easting * 1000) + 1000
        top = (northing * 1000) + 1000
        new_bounds = rasterio.coords.BoundingBox(left, bottom, right, top)
        src.crs = correct_crs
        src.transform = rasterio.transform.from_bounds(left, bottom, right, top, src.width, src.height)
