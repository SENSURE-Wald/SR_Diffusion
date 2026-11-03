import pathlib
import os
from tqdm import tqdm
import json

metadata_path = pathlib.Path("/NAS/data/dop/BW/v_dop_20_bildflugkacheln.json")
missing_tiles_path = metadata_path.parent / "missing_tiles"
tif_path = pathlib.Path("/NAS/data/dop/BW/tif")
metadata_dict = {}

with open(metadata_path, "r") as f:
    meta_dict = json.load(f)["features"]

meta_data = {}

for feature in meta_dict:
    properties = feature["properties"]
    tile_id = str(properties["dop_kachel"])
    date = properties["befliegungsdatum"]

    meta_data[tile_id] = date

missing_tiles = []

for tif in tqdm(tif_path.rglob("*.tif")):
    try:
        tile_id = tif.name.split(".tif")[0]
        # print(tile_id)
        date = meta_data[tile_id]
        # os.rename(tif, output_path)
        magic_number = tile_id[0:2]
        coords = (int(tile_id[2:5]), int(tile_id[5:]))
        output_name = f"dop20_{magic_number}_{coords[0]}_{coords[1]}_{date}.tif"
        output_path = tif.parent / output_name
        os.rename(tif, output_path)
    except KeyError as _:
        missing_tiles.append(tile_id)
        os.rename(tif, missing_tiles_path / tif.name)
