from tqdm import tqdm
import pathlib
import zipfile
import argparse
import requests
import io

parser = argparse.ArgumentParser()
parser.add_argument("INPUT", type=pathlib.Path)
parser.add_argument("OUTPUT", type=pathlib.Path)

args = parser.parse_args()

with open(args.INPUT, "r") as f:
    for link in tqdm(f.readlines()):
        split_link = link.split("/")
        tile_name = split_link[-1].split(".")[0].replace("_tiff", "")
        tif_path, metadata_path = args.OUTPUT / (tile_name + ".tif"), args.OUTPUT / (tile_name + "_akt.csv")
        if tif_path.exists() and metadata_path.exists():
            continue
        response = requests.get(link.strip())
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content), "r") as zip_file:
            zip_file.extractall(args.OUTPUT)
