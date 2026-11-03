import rasterio
import numpy as np
import argparse
import pathlib
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("INPUT", type=pathlib.Path)
args = parser.parse_args()
empty_arrays = []
file_list = list(args.INPUT.rglob("*.tif"))
for tif_file in tqdm(file_list):
    with rasterio.open(tif_file) as src:
        arr = src.read()
        profile = src.profile

        if not np.any(arr):
            empty_arrays.append(tif_file)


print(f"We have {len(empty_arrays) / len(file_list)} empty files.")
