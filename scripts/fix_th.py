import rasterio
import pathlib
import tqdm

INPUT = pathlib.Path("/NAS/data/dop/TH/codede/")

for tif_path in INPUT.rglob("*.tif"):
    with rasterio.open(tif_path) as tif:
        print(tif.width, tif.height)
