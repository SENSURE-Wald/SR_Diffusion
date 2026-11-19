import argparse
from tqdm import tqdm
import pathlib
import rasterio
import datetime

parser = argparse.ArgumentParser()

parser.add_argument("INPUT", type=pathlib.Path)
parser.add_argument("-s", "--size", default=800, type=int)

args = parser.parse_args()

for file in tqdm(list(args.INPUT.glob("*.tif"))):
    with rasterio.open(file, "r+") as tif:
        if "recording_date" in tif.tags():
            continue

        split_file_name = file.name.split("_")
        coords = list(map(float, split_file_name[2:4]))
        date_str = split_file_name[-1].split(".")[0]
        date = datetime.datetime.strptime(date_str, "%Y-%m-%d")

        tif.update_tags(recording_date=date.isoformat())
