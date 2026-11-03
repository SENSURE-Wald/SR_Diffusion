import pathlib
import os
from tqdm import tqdm

metadata_path = pathlib.Path("/NAS/data/dop/SA/metadata")
tif_path = pathlib.Path("/NAS/data/dop/SA/tif")
metadata_dict = {}

for metadata in metadata_path.rglob("*.meta"):
    split_file_name = metadata.name.split("_")
    coords = tuple(map(int, split_file_name[2:4]))

    if coords in metadata_dict:
        print("Duplicate tile?")

    metadata_dict[(coords[0], coords[1])] = metadata.name.split(".")[0]


for tif in tqdm(tif_path.rglob("*.tif")):
    without_suffix = tif.name.split(".")[0]
    split_file_name = without_suffix.split("_")
    coords = (int(split_file_name[-2]) // 1000, int(split_file_name[-1]) // 1000)

    if coords not in metadata_dict:
        print("Missing metadata?")

    new_file_name = f"{metadata_dict[(coords[0], coords[1])]}.tif"
    output_path = tif.parent / new_file_name

    os.rename(tif, output_path)
