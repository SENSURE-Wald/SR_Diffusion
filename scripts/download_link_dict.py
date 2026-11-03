from tqdm import tqdm
import pathlib
import zipfile
import argparse
import requests
import io
import json
import rasterio
from urllib.parse import urlparse, parse_qs

parser = argparse.ArgumentParser()
parser.add_argument("INPUT", type=pathlib.Path)
parser.add_argument("OUTPUT", type=pathlib.Path)

args = parser.parse_args()

with open(args.INPUT, "r") as f:
    dict = json.load(f)

for link in tqdm(dict.values()):
    link_txt = link.strip()
    file_name = link_txt.split("/")[-1]

    params = parse_qs(urlparse(link).query)
    left, bottom, _, _ = map(lambda x: int(float(x) / 100), params["BBOX"][0].split(","))

    if "rgb" in params["LAYERS"][0]:
        layer = "rgb"

    elif "ir" in params["LAYERS"][0]:
        layer = "ir"

    else:
        raise ValueError("Invalid layer.")

    file_name = f"dop20_32_{left}_{bottom}_{layer}_sl.tif"
    output_path: pathlib.Path = args.OUTPUT / file_name

    if output_path.is_file():
        continue
    response = requests.get(link)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(response.content)
    # with rasterio.open(io.BytesIO(response.content)) as src:
    #    left, bottom, _, _ = src.bounds
    #    import sys

    #    print(left, bottom, link)
    #    output_path: pathlib.Path = args.OUTPUT / file_name

    # if output_path.is_file():
    #    continue

    # sys.exit(1)
