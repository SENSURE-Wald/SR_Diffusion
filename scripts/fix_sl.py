import pathlib
import rasterio
from tqdm import tqdm
import numpy as np

INPUT = pathlib.Path("/NAS/data/dop/SL/tif")
OUTPUT = pathlib.Path("/NAS/data/dop/SL/combined")
file_list = list(INPUT.rglob("*rgb*.tif"))
empties_array = []

for tif_file in tqdm(file_list):
    # If a combined file already exists we can skip.
    output_path = OUTPUT / tif_file.name.replace("rgb_", "")
    if output_path.exists():
        continue

    with rasterio.open(tif_file, "r") as src:
        arr = src.read()
        profile = src.profile

        # File is completely zero?
        if not np.any(arr):
            empties_array.append(tif_file)
            # move the file
            continue

        ir_file = tif_file.parent / tif_file.name.replace("rgb", "ir")

        with rasterio.open(ir_file) as ir_tif:
            ir = ir_tif.read(1)
            arr[3, :, :] = ir

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(arr)
